import numpy as np
import matplotlib.pyplot as plt
# import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import time
import argparse
import draccus
import logging
logging.basicConfig(level=logging.DEBUG)
import multiprocessing as mp
import collections
import yaml
import random
import os
from scripts.numpy_logger import NumpyCSVLogger

from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig

try:
    from ruckig import Ruckig, InputParameter, OutputParameter, Result
    RUCKIG_AVAILABLE = True
except ImportError:
    RUCKIG_AVAILABLE = False
    logging.warning("Ruckig not installed. Ruckig smoothing will be disabled.")

def make_camera_config(cfg: dict):
    t = cfg.get('type')
    if t == 'opencv':
        return OpenCVCameraConfig(
            index_or_path=cfg['index_or_path'],
            width=cfg.get('width', 640),
            height=cfg.get('height', 480),
            fps=cfg.get('fps', 30),
        )
    elif t == 'orbbec':
        return OrbbecCameraConfig(
            index_or_path=cfg['index_or_path'],
            width=cfg.get('width', 640),
            height=cfg.get('height', 480),
            fps=cfg.get('fps', 30),
        )
    else:
        raise ValueError(f"Unsupported camera type: {t!r}")

# ---------- 子进程：推理循环 ----------
def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    config,
    checkpoint_dir,
    args,
):

    # 1. 只在该进程里加载一次模型 / CUDA
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    while True:
        item = in_q.get()
        if item is None:            # 收到结束标识
            del policy
            break
        # item expected to be (idx, obs, anchor) where anchor may be None
        if isinstance(item, tuple) and len(item) == 3:
            idx, obs, anchor = item
        elif isinstance(item, tuple) and len(item) == 2:
            idx, obs = item
            anchor = None
        else:
            # unexpected message, skip
            continue
        start_time = time.time()
        result = policy.infer(obs)
        infer_time = time.time() - start_time
        print(f"Step {idx}: infer time = {infer_time:.4f} seconds")
        actions = result.get("actions")
        # perform horizon-level smoothing and optional QP optimization in worker
        try:
            if actions is not None:
                arr = np.asarray(actions, dtype=float)
                if arr.ndim == 1:
                    arr = arr[None, :]
                H, D = arr.shape
                # assume last column is gripper; exclude it from horizon-level processing
                if D >= 2:
                    body = arr[:, :-1]
                    grip = arr[:, -1:]
                else:
                    body = arr
                    grip = None

                # smooth only the body (joints)
                if getattr(args, "horizon_smooth", "none") != "none" and body.size > 0:
                    try:
                        body = smooth_horizon(body, method=args.horizon_smooth, window=args.horizon_window, ema_alpha=args.horizon_ema_alpha)
                    except Exception:
                        logging.exception("worker horizon smoothing failed")

                # QP optimization only on body
                if getattr(args, "qp_lambda_acc", 0.0) and args.qp_lambda_acc > 0 and body.size > 0:
                    try:
                        body = optimize_horizon_qp(body, lambda_acc=args.qp_lambda_acc, velocity_limit=args.qp_velocity_limit, anchor=anchor)
                    except Exception as e:
                        logging.exception("worker qp optimization failed: %s", e)

                # recombine
                if grip is None:
                    actions = body
                else:
                    actions = np.concatenate([body, grip], axis=1)
        except Exception as e:
            logging.exception("worker horizon postprocessing failed, falling back to raw actions: %s", e)
        out_q.put((idx, actions))

def _apply_transition(old_actions, new_actions, h_fn):
    """
    通用 transition 应用器：对前 n_interp 个旧动作与新动作按权重 h(t) 做插值。
    h_fn: 接受 t in (0,1) 返回权重 h(t) 的函数，interp = (1-h)*old + h*new。
    返回 list[np.ndarray]
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        h = float(h_fn(t))
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def linear_transition(old_actions, new_actions):
    """线性插值：h(t)=t"""
    return _apply_transition(old_actions, new_actions, lambda t: t)


def cubic_transition(old_actions, new_actions):
    """三次 Hermite ease-in/out：h(t)=3t^2-2t^3"""
    return _apply_transition(old_actions, new_actions, lambda t: 3 * t**2 - 2 * t**3)


def quintic_transition(old_actions, new_actions):
    """五次平滑：h(t)=10t^3-15t^4+6t^5"""
    return _apply_transition(old_actions, new_actions, lambda t: 10 * t**3 - 15 * t**4 + 6 * t**5)

def ema_transition(old_actions, new_actions, alpha=0.7):
    """
    指数加权平滑(EMA)，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    alpha: 新动作权重,0~1
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        interp_action = alpha * new_actions[i_interp] + (1 - alpha) * old_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def smooth_horizon(actions: np.ndarray, method: str = "none", window: int = 3, ema_alpha: float = 0.9):
    """
    Smooth predicted horizon (actions: shape (H, D)). Returns smoothed array same shape.
    method: 'none'|'moving'|'median'|'ema'
    window: integer window size for moving/median (should be odd for median/centered moving)
    ema_alpha: smoothing factor for EMA (0-1)
    """
    if method == "none" or window <= 1 and method in ("moving", "median"):
        return actions
    actions = np.asarray(actions, dtype=float)
    H, D = actions.shape
    if method == "moving":
        k = max(1, int(window))
        if k == 1:
            return actions
        kernel = np.ones(k, dtype=float) / k
        sm = np.zeros_like(actions)
        for d in range(D):
            sm[:, d] = np.convolve(actions[:, d], kernel, mode="same")
        return sm
    elif method == "median":
        k = max(1, int(window))
        if k == 1:
            return actions
        pad = k // 2
        padded = np.pad(actions, ((pad, pad), (0, 0)), mode="edge")
        sm = np.zeros_like(actions)
        for t in range(H):
            sm[t] = np.median(padded[t:t + k], axis=0)
        return sm
    elif method == "ema":
        alpha = float(ema_alpha)
        sm = np.zeros_like(actions)
        s = actions[0].copy()
        for t in range(H):
            s = alpha * actions[t] + (1.0 - alpha) * s
            sm[t] = s
        return sm
    else:
        raise ValueError(f"Unknown horizon smoothing method: {method!r}")

def set_seeds(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # JAX PRNG is handled when creating keys in the code that uses it.

def _build_D2(H: int) -> np.ndarray:
    """Construct second-difference matrix D2 of shape (H-2, H).
    (D2 a)[t] = a[t+2] - 2 a[t+1] + a[t]
    """
    if H < 3:
        return np.zeros((0, H), dtype=float)
    D2 = np.zeros((H - 2, H), dtype=float)
    for i in range(H - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def optimize_horizon_qp(new_actions: np.ndarray, lambda_acc: float = 0.1, velocity_limit: float = 0.0, anchor: np.ndarray | None = None) -> np.ndarray:
    """
    Light-weight QP-style optimizer: minimize 0.5||a - a_ref||^2 + 0.5 * lambda_acc * ||D2 a||^2
    Solves per-joint linear system: (I + lambda_acc * D2^T D2) a = a_ref
    new_actions: (H, D)
    velocity_limit: if >0, post-clamp per-step deltas to [-velocity_limit, velocity_limit]
    anchor: optional previous executed action (1D array of length D) used to bias first element
    Returns optimized actions (H, D)
    """
    a_ref = np.asarray(new_actions, dtype=float)
    H, D = a_ref.shape
    if H == 0:
        return a_ref
    D2 = _build_D2(H)
    if lambda_acc <= 0 or D2.size == 0:
        sol = a_ref.copy()
    else:
        M = np.eye(H, dtype=float)
        # add regularizer: lambda_acc * D2^T D2
        reg = lambda_acc * (D2.T @ D2)
        A = M + reg
        # For numerical stability, add small diag jitter
        A += np.eye(H) * 1e-8
        sol = np.zeros_like(a_ref)
        # Solve per joint
        for j in range(D):
            b = a_ref[:, j]
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            sol[:, j] = x
    # Optional simple post-processing: clamp per-step velocity
    if velocity_limit and velocity_limit > 0.0:
        # if anchor provided, use it as previous value; else use sol[0]
        prev = None
        if anchor is not None:
            prev = np.asarray(anchor, dtype=float)
        else:
            prev = sol[0].copy()
        # enforce on each joint independently
        for j in range(D):
            val = prev[j]
            for t in range(H):
                delta = sol[t, j] - val
                if delta > velocity_limit:
                    delta = velocity_limit
                elif delta < -velocity_limit:
                    delta = -velocity_limit
                val = val + delta
                sol[t, j] = val
    return sol

def main():
    parser = argparse.ArgumentParser(description="Inference script for AgileX follower robot")
    parser.add_argument("--port", type=str, required=True, help="port name")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="path to checkpoint directory")
    parser.add_argument("--fps", type=int, required=False, default=30, help="frames per second")
    parser.add_argument("--task", type=str, required=True, help="task prompt")
    parser.add_argument("--id", type=str, required=False, help="robot id", default="left")
    parser.add_argument("--cameras", type=str, required=False, help="camera config yaml", default=None)
    parser.add_argument("--max_relative_target", type=int, required=False, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    parser.add_argument("--action_steps", type=int, required=False, default=20, help="number of action steps to execute before next inference")
    parser.add_argument("--smooth_type", type=str, default="cubic", choices=["linear", "cubic", "quintic", "ema"], help="动作平滑策略: linear/cubic/quintic/ema")
    parser.add_argument("--ema_alpha", type=float, default=0.5, help="EMA平滑时新动作权重alpha,0~1")
    parser.add_argument("--align_mode", type=str, default="step", choices=["step", "euclidean"], help="新动作对齐方式: step(步数) 或 euclidean(欧氏距离)")
    # Horizon-level smoothing of the predicted action sequence (uses full predicted horizon)
    parser.add_argument("--horizon_smooth", type=str, default="ema", choices=["none", "moving", "median", "ema"], help="对预测 horizon 进行时序平滑: none/moving/median/ema")
    parser.add_argument("--horizon_window", type=int, default=30, help="窗口大小用于 moving/median 平滑（越大越平滑）。奇数优先")
    parser.add_argument("--horizon_ema_alpha", type=float, default=0.7, help="horizon EMA alpha 用于 horizon_smooth=ema")
    # QP-style online optimizer options
    parser.add_argument("--qp_lambda_acc", type=float, default=0.0, help="二阶差分加速惩罚系数（>=0），qp优化时使用。0 表示禁用")
    parser.add_argument("--qp_velocity_limit", type=float, default=0.0, help="可选的每步最大速度（动作单位/step），>0 则启用简单束缚后处理")
    # speed or pose
    parser.add_argument("--mode", type=str, required=False, default="pose", help="inference mode")
    # jitter seed
    parser.add_argument("--seed", type=int, required=False, default=10002)
    # Ruckig smoothing options
    parser.add_argument("--ruckig_enable", action="store_true", help="启用 Ruckig 实时平滑")
    parser.add_argument("--ruckig_lookahead", type=int, default=4, help="Ruckig 目标 waypoint 间隔步数")
    parser.add_argument("--ruckig_max_vel", type=float, default=200000.0, help="Ruckig 最大关节速度 (Piper单位/s)，建议根据实际动作范围调整")
    parser.add_argument("--ruckig_max_acc", type=float, default=500000.0, help="Ruckig 最大关节加速度 (Piper单位/s²)")
    parser.add_argument("--ruckig_max_jerk", type=float, default=5000000.0, help="Ruckig 最大 Jerk (Piper单位/s³)")
    # 日志路径
    parser.add_argument("--log_dir", type=str, default="/home/test/test_tra", help="日志文件保存目录")
    args = parser.parse_args()

    set_seeds(args.seed)

    # 创建日志记录器
    import os
    from datetime import datetime
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    logger_action = NumpyCSVLogger(os.path.join(args.log_dir, f"action_sent_{log_timestamp}.csv"), mode="w")
    logger_end_pose = NumpyCSVLogger(os.path.join(args.log_dir, f"end_pose_{log_timestamp}.csv"), mode="w")
    logger_joint_state = NumpyCSVLogger(os.path.join(args.log_dir, f"joint_state_{log_timestamp}.csv"), mode="w")
    print_log = True
    print(f"Logging to {args.log_dir} with timestamp {log_timestamp}")

    # 解析摄像头配置
    if args.cameras is not None:
        raw = yaml.safe_load(args.cameras)
        cameras = {name: make_camera_config(cfg) for name, cfg in raw.items()}
    else:
        cameras = {}


    robot_config = AlohaAgileXFollowerConfig(
        port=args.port,
        id=args.id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
    )
    # 选择配置和 checkpoint
    robot = AlohaAgileXFollower(robot_config)

    # Load pretrained policy
    # policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # ==== 1. 启动推理子进程 ====
    ctx = mp.get_context("spawn")        # "spawn" 更安全，尤其 CUDA
    in_q: mp.Queue = ctx.Queue(maxsize=4)   # 根据实时性调节 maxsize
    out_q: mp.Queue = ctx.Queue(maxsize=4)
    config = _config.get_config("pi05_agileX")
    checkpoint_dir = args.checkpoint_dir #"/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")

    proc = ctx.Process(
        target=inference_worker,
        args=(in_q, out_q, config, checkpoint_dir, args)
    )
    proc.daemon = True
    proc.start()

    robot.connect()
    try:
        prev_main = None
        i, sent_idx, recv_idx = 0, 0, 0
        kMaxTimeStamps = 600000

        # rows = []
        step = 1
        step_time = step/(args.fps)
        prompt = args.task
        tokenizer = PaligemmaTokenizer()
        tokenized, mask = tokenizer.tokenize(prompt)

        action_queue = collections.deque()  # 存储当前动作序列
        waiting_for_infer = False
        action_step_counter = 0  # 记录已执行的动作步数
        first = True

        # ==== Ruckig 平滑初始化 ====
        ruckig_enabled = args.ruckig_enable and RUCKIG_AVAILABLE
        ruckig_instance = None
        ruckig_inp = None
        ruckig_out = None
        ruckig_initialized = False
        ruckig_step_counter = 0  # 稀疏采样计数器
        last_sent_action = None  # 用于降级时保持静止
        DOF = 6  # 关节数，不含 gripper

        if ruckig_enabled:
            ruckig_instance = Ruckig(DOF, step_time)
            ruckig_inp = InputParameter(DOF)
            ruckig_out = OutputParameter(DOF)
            # 设置约束
            ruckig_inp.max_velocity = [args.ruckig_max_vel] * DOF
            ruckig_inp.max_acceleration = [args.ruckig_max_acc] * DOF
            ruckig_inp.max_jerk = [args.ruckig_max_jerk] * DOF
            logging.info(f"Ruckig smoothing enabled: lookahead={args.ruckig_lookahead}, "
                         f"max_vel={args.ruckig_max_vel}, max_acc={args.ruckig_max_acc}, max_jerk={args.ruckig_max_jerk}")

        # robot.send_action_np(np.array([-7980, 20113, -2285, -7921, 37285,  1023,     0.]))
        # time.sleep(1)
        while i < kMaxTimeStamps:
            t0 = time.perf_counter()

            # 1. 只有在执行了action_steps步后才采集观测并推理
            if not waiting_for_infer and (action_step_counter >= args.action_steps or first):
                first = False
                obs = robot.get_observation()
                # Ensure obs["state"] is a numpy array
                state7 = np.asarray(obs.get("state"))
                if args.mode == "speed":
                    # compute delta = current_main - prev_main (or zeros for first frame)
                    if prev_main is None:
                        delta = np.zeros_like(state7)
                    else:
                        try:
                            delta = state7 - prev_main
                        except Exception:
                            delta = np.zeros_like(state7)
                    obs["state"] = np.concatenate([state7, delta], axis=-1)
                    prev_main = state7.copy()
                else:
                    obs["state"] = state7
                obs["tokenized_prompt"] = tokenized[None]
                obs["tokenized_prompt_mask"] = mask[None]
                obs["token_ar_mask"] = None
                obs["token_loss_mask"] = None
                # send anchor (first pending action) to worker so it can align/anchor optimization
                anchor = None
                if len(action_queue) > 0:
                    try:
                        anchor = np.asarray(action_queue[0], dtype=float)
                    except Exception:
                        anchor = None
                try:
                    in_q.put_nowait((sent_idx, obs, anchor))
                    sent_idx += 1
                    waiting_for_infer = True
                    action_step_counter = 0
                except mp.queues.Full:
                    logging.debug("inference queue full, dropping frame")

            # 2. 如果有新推理结果，立即清空并更新 action_queue
            try:
                idx, action_vals = out_q.get_nowait()
                recv_idx = idx
                logging.debug(f"got result #{recv_idx}")

                # 1. 记录未执行的旧动作
                old_actions = list(action_queue)
                action_queue.clear()

                # 2. 新推理动作起点
                if args.align_mode == "step":
                    start_idx = action_step_counter
                elif args.align_mode == "euclidean" and len(old_actions) > 0 and len(action_vals) > 0:
                    # 取旧队列第一个动作，与新动作序列做欧氏距离最小匹配
                    old_action = old_actions[0]
                    dists = np.linalg.norm(action_vals - old_action, axis=1)
                    start_idx = int(np.argmin(dists))
                else:
                    start_idx = 0
                new_actions = action_vals[start_idx:]
                # 对新推理得到的 horizon 做时序平滑（因为 horizon 是完整的未来序列，可以使用中心/非因果平滑）
                try:
                    if args.horizon_smooth != "none" and len(new_actions) > 0:
                        arr = np.asarray(new_actions, dtype=float)
                        if arr.ndim == 1:
                            arr = arr[None, :]
                        H, D = arr.shape
                        if D >= 2:
                            body = arr[:, :-1]
                            grip = arr[:, -1:]
                        else:
                            body = arr
                            grip = None
                        if body.size > 0:
                            try:
                                body = smooth_horizon(body, method=args.horizon_smooth, window=args.horizon_window, ema_alpha=args.horizon_ema_alpha)
                            except Exception:
                                logging.exception("main horizon smoothing failed")
                        new_actions = np.concatenate([body, grip], axis=1) if grip is not None else body
                except Exception as e:
                    logging.exception("horizon smoothing failed, falling back to raw predictions: %s", e)

                # 可选：对整个 horizon 做 QP-style 优化以最小化二阶差分（加速度）
                try:
                    if args.qp_lambda_acc and args.qp_lambda_acc > 0 and len(new_actions) > 0:
                        # anchor 使用当前队列第一个动作（若有）以保证前端对齐
                        anchor = None
                        if len(old_actions) > 0:
                            try:
                                anchor = np.asarray(old_actions[0], dtype=float)
                            except Exception:
                                anchor = None
                        arr = np.asarray(new_actions, dtype=float)
                        if arr.ndim == 1:
                            arr = arr[None, :]
                        H, D = arr.shape
                        if D >= 2:
                            body = arr[:, :-1]
                            grip = arr[:, -1:]
                        else:
                            body = arr
                            grip = None
                        if body.size > 0:
                            try:
                                body = optimize_horizon_qp(body, lambda_acc=args.qp_lambda_acc, velocity_limit=args.qp_velocity_limit, anchor=anchor)
                            except Exception as e:
                                logging.exception("main qp optimization failed: %s", e)
                        new_actions = np.concatenate([body, grip], axis=1) if grip is not None else body
                except Exception as e:
                    logging.exception("qp horizon optimization failed, falling back to pre-qped predictions: %s", e)

                # 3. 平滑衔接（可通过参数切换）
                if args.smooth_type == "linear":
                    smooth_actions = linear_transition(old_actions, new_actions)
                elif args.smooth_type == "cubic":
                    smooth_actions = cubic_transition(old_actions, new_actions)
                elif args.smooth_type == "quintic":
                    smooth_actions = quintic_transition(old_actions, new_actions)
                elif args.smooth_type == "ema":
                    smooth_actions = ema_transition(old_actions, new_actions, alpha=args.ema_alpha)
                else:
                    raise ValueError(f"Unknown smooth_type: {args.smooth_type}")
                for a in smooth_actions:
                    action_queue.append(a)

                waiting_for_infer = False
            except mp.queues.Empty:
                pass

            # 3. 如果 action_queue 有动作，发给 robot
            if action_queue:
                if ruckig_enabled:
                    # ==== 稀疏采样 + Ruckig 平滑 ====
                    lookahead = args.ruckig_lookahead

                    # 首次初始化：从机械臂读取当前状态
                    if not ruckig_initialized:
                        try:
                            state = robot.get_joint_state()["state"]
                            ruckig_inp.current_position = list(state[:DOF])
                            ruckig_inp.current_velocity = [0.0] * DOF
                            ruckig_inp.current_acceleration = [0.0] * DOF
                            ruckig_initialized = True
                            logging.info("Ruckig initialized with robot state")
                        except Exception as e:
                            logging.warning(f"Failed to init Ruckig from robot state: {e}, using first action")
                            first_action = action_queue[0]
                            ruckig_inp.current_position = list(np.asarray(first_action[:DOF], dtype=float))
                            ruckig_inp.current_velocity = [0.0] * DOF
                            ruckig_inp.current_acceleration = [0.0] * DOF
                            ruckig_initialized = True

                    # # 稀疏采样：在即将到达目标时（lookahead-1 步）更新新目标

                    # # 这样 Ruckig 会从当前运动状态平滑过渡到新目标，无需估计 target_velocity
                    # if ruckig_step_counter == lookahead - 1 or ruckig_step_counter == 0:
                    #     # 首次（step 0）或即将到达目标时（step lookahead-1），设置/更新目标
                    #     # 目标选取：取队列中 lookahead-1 位置的动作
                    #     target_idx = min(lookahead - 1, len(action_queue) - 1)
                    #     target_action = action_queue[target_idx]
                    #     target_pos = np.asarray(target_action[:DOF], dtype=float)
                    #     ruckig_inp.target_position = list(target_pos)
                    #     # target_velocity 设为 0，Ruckig 会使用当前运动状态重新规划平滑轨迹
                    #     ruckig_inp.target_velocity = [0.0] * DOF
                    #     ruckig_inp.target_acceleration = [0.0] * DOF

                    # 每步都更新目标：取 lookahead 步后的动作作为目标
                    
                    # 如果队列长度不足，取队列最后一个
                    target_idx = min(lookahead - 1, len(action_queue) - 1)
                    target_action = action_queue[target_idx]
                    target_pos = np.asarray(target_action[:DOF], dtype=float)
                    ruckig_inp.target_position = list(target_pos)
                    
                    # 基于运动学估算 target_velocity: v_target = v_current + a_current * Δt
                    duration = step_time * lookahead
                    current_vel = np.array(ruckig_inp.current_velocity)
                    current_acc = np.array(ruckig_inp.current_acceleration)
                    target_vel = current_vel + current_acc * duration
                    # 限制在约束范围内
                    target_vel = np.clip(target_vel, -args.ruckig_max_vel, args.ruckig_max_vel)
                    ruckig_inp.target_velocity = list(target_vel)
                    ruckig_inp.target_acceleration = [0.0] * DOF

                    # Ruckig 更新
                    try:
                        result = ruckig_instance.update(ruckig_inp, ruckig_out)
                        # 发送平滑后的位置
                        smoothed_pos = np.array(ruckig_out.new_position)
                        gripper_val = action_queue[0][DOF]  # gripper 直接透传
                        action_to_send = np.concatenate([smoothed_pos, [gripper_val]])
                        if print_log:
                            logger_action.log(time.perf_counter(), action_to_send[:7])
                            # 记录末端位姿和关节状态
                            try:
                                end_pose_msg = robot.piper.GetArmEndPoseMsgs()
                                pose = end_pose_msg.end_pose
                                logger_end_pose.log(time.perf_counter(), 
                                    pose.X_axis / 1000.0, pose.Y_axis / 1000.0, pose.Z_axis / 1000.0,
                                    pose.RX_axis / 1000.0, pose.RY_axis / 1000.0, pose.RZ_axis / 1000.0)
                                joint_msg = robot.piper.GetArmJointMsgs()
                                js = joint_msg.joint_state
                                logger_joint_state.log(time.perf_counter(),
                                    js.joint_1 / 1000.0, js.joint_2 / 1000.0, js.joint_3 / 1000.0,
                                    js.joint_4 / 1000.0, js.joint_5 / 1000.0, js.joint_6 / 1000.0)
                            except Exception:
                                pass
                        robot.send_action_np(action_to_send[:7])
                        last_sent_action = action_to_send
                        action_step_counter += 1

                        # 更新 Ruckig 状态
                        ruckig_out.pass_to_input(ruckig_inp)

                        # 基于距离的动态 popleft：找到 smoothed_pos 最接近的 action
                        search_range = min(max(1, lookahead // 2), len(action_queue))
                        distances = []
                        for idx in range(search_range):
                            action_pos = np.asarray(action_queue[idx][:DOF], dtype=float)
                            dist = np.linalg.norm(smoothed_pos - action_pos)
                            distances.append(dist)
                        closest_idx = int(np.argmin(distances))
                        
                        # 记录距离日志（用于调试）
                        logging.debug(f"Distances: {[f'{d:.1f}' for d in distances]}, closest_idx={closest_idx}")
                        
                        # 如果 closest_idx > 0，消耗 closest_idx 个 action
                        # 如果 closest_idx == 0，不消耗（smoothed_pos 还没超过 queue[0]）
                        if closest_idx > 0:
                            for _ in range(closest_idx):
                                action_queue.popleft()
                        # 更新稀疏采样计数器
                        ruckig_step_counter = (ruckig_step_counter + 1) % lookahead
                    except Exception as e:
                        logging.exception(f"Ruckig update failed: {e}, falling back to raw action")
                        # 降级：直接发送原始动作
                        action_to_send = action_queue.popleft()
                        if print_log:
                            logger_action.log(time.perf_counter(), action_to_send[:7])
                        robot.send_action_np(action_to_send[:7])
                        last_sent_action = action_to_send
                        action_step_counter += 1
                        ruckig_step_counter = (ruckig_step_counter + 1) % lookahead
                else:
                    # ==== 原始逻辑（无 Ruckig）====
                    action_to_send = action_queue.popleft()
                    if print_log:
                        logger_action.log(time.perf_counter(), action_to_send[:7])
                    robot.send_action_np(action_to_send[:7])
                    last_sent_action = action_to_send
                    action_step_counter += 1
                    # print(f'publish an action:{time.perf_counter()},action counter:{action_step_counter}')

            # 2.5 统计
            i += 1
            dt_s = time.perf_counter() - t0
            # print(f"loop {i} dt={dt_s:.3f} s")
            time.sleep(max(step_time - dt_s,0))
    finally:
        # ==== 3. 结束 ====
        in_q.put(None)      # 通知子进程退出
        proc.join()
        robot.disconnect()
        # 关闭日志
        logger_action.close()
        logger_end_pose.close()
        logger_joint_state.close()
        print(f"Logs saved to {args.log_dir}")

if __name__ == "__main__":
    main()