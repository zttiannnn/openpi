import numpy as np
import matplotlib.pyplot as plt
# import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import time
import argparse
import draccus
import logging
# logging.basicConfig(level=logging.DEBUG)
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
    from ruckig import Ruckig, InputParameter, OutputParameter, Result, Trajectory
    RUCKIG_AVAILABLE = True
except ImportError:
    RUCKIG_AVAILABLE = False
    logging.warning("Ruckig not installed. Ruckig smoothing will be disabled.")

from dataclasses import dataclass

@dataclass
class ActionPoint:
    """动作点，区分关键点和 Ruckig 插值点"""
    action: np.ndarray    # 7-DoF 动作 (6 joints + gripper)
    is_keypoint: bool     # True = 采样关键点, False = Ruckig 插值点

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

def log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel):
    """记录末端位姿、关节角度、关节速度"""
    # 记录末端位姿 (only pose, no velocity)
    try:
        end_pose_msg = robot.piper.GetArmEndPoseMsgs()
        if end_pose_msg is not None:
            pose = getattr(end_pose_msg, "end_pose", end_pose_msg)
            if hasattr(pose, "X_axis"):
                # 日志格式: timestamp, X, Y, Z, RX, RY, RZ (7列)
                logger_end_pose.log(time.perf_counter(), 
                    pose.X_axis / 1000.0, pose.Y_axis / 1000.0, pose.Z_axis / 1000.0,
                    pose.RX_axis / 1000.0, pose.RY_axis / 1000.0, pose.RZ_axis / 1000.0)
    except Exception:
        pass

    # 记录关节角度反馈
    try:
        joint_msg = robot.piper.GetArmJointMsgs()
        if joint_msg is not None:
            js = getattr(joint_msg, "joint_state", joint_msg)
            if hasattr(js, "joint_1"):
                # 日志格式: timestamp, J1, J2, J3, J4, J5, J6 (7列)
                logger_joint_state.log(time.perf_counter(),
                    js.joint_1 / 1000.0, js.joint_2 / 1000.0, js.joint_3 / 1000.0,
                    js.joint_4 / 1000.0, js.joint_5 / 1000.0, js.joint_6 / 1000.0)
    except Exception:
        pass
    
    # 记录关节速度 (从 GetMotorStates 或 GetArmHighSpdInfoMsgs)
    try:
        joint_vel = [0.0] * 6
        if hasattr(robot.piper, "GetMotorStates"):
            motor_states = robot.piper.GetMotorStates()
        else:
            motor_states = robot.piper.GetArmHighSpdInfoMsgs()
        
        if motor_states is not None:
            # 返回格式: (time_stamp, Hz, motor_1, motor_2, ..., motor_6)
            if isinstance(motor_states, (list, tuple)) and len(motor_states) > 2:
                for i in range(6):
                    if i + 2 < len(motor_states):
                        motor_info = motor_states[i + 2]
                        if hasattr(motor_info, "motor_speed"):
                            joint_vel[i] = motor_info.motor_speed / 1000.0  # 0.001rad/s -> rad/s
            else:
                # 如果是对象，直接访问属性
                for i in range(6):
                    joint_key = f"motor_{i+1}"
                    if hasattr(motor_states, joint_key):
                        motor_info = getattr(motor_states, joint_key)
                        if hasattr(motor_info, "motor_speed"):
                            joint_vel[i] = motor_info.motor_speed / 1000.0
        
        # 日志格式: timestamp, J1_vel, J2_vel, J3_vel, J4_vel, J5_vel, J6_vel (7列)
        logger_joint_vel.log(time.perf_counter(),
            joint_vel[0], joint_vel[1], joint_vel[2],
            joint_vel[3], joint_vel[4], joint_vel[5])
    except Exception:
        pass

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


def estimate_keypoint_velocity(raw_actions: np.ndarray, keypoint_idx: int, dt: float, dof: int = 6) -> np.ndarray:
    """
    用关键点前后的普通点差分估计该关键点处的速度。
    
    Args:
        raw_actions: 原始动作序列 (H, D)
        keypoint_idx: 关键点在原始序列中的索引
        dt: 控制周期
        dof: 关节自由度（不含 gripper）
    
    Returns:
        速度向量 (dof,)
    """
    H = len(raw_actions)
    if keypoint_idx == 0:
        # 首个关键点：向前差分
        if H > 1:
            vel = (raw_actions[1, :dof] - raw_actions[0, :dof]) / dt
        else:
            vel = np.zeros(dof)
    elif keypoint_idx >= H - 1:
        # 末尾关键点：向后差分
        vel = (raw_actions[H - 1, :dof] - raw_actions[H - 2, :dof]) / dt
    else:
        # 中间关键点：中心差分（用前后普通点）
        vel = (raw_actions[keypoint_idx + 1, :dof] - raw_actions[keypoint_idx - 1, :dof]) / (2 * dt)
    return vel


def offline_spline_plan(
    raw_actions: np.ndarray,
    sample_rate: int,
    dt: float,
    dof: int = 6,
    start_velocity: np.ndarray = None,
) -> list:
    """
    离线三次样条插值规划。
    
    使用三次样条拟合关键点，保证 C2 连续（位置、速度、加速度连续）。
    当提供 start_velocity 时，使用 clamped 边界条件确保速度连续。
    
    Args:
        raw_actions: 原始动作序列 (H, 7)，前 6 列为关节，第 7 列为 gripper
        sample_rate: 采样频率（每 N 个动作采 1 个关键点）
        dt: 控制周期
        dof: 关节自由度（不含 gripper）
        start_velocity: 起点速度 (dof,)，用于 clamped 边界条件
    
    Returns:
        List[ActionPoint]，规划后的动作序列
    """
    from scipy.interpolate import CubicSpline
    
    raw_actions = np.asarray(raw_actions, dtype=float)
    H = len(raw_actions)
    if H < 2:
        return [ActionPoint(action=raw_actions[0], is_keypoint=True)] if H == 1 else []
    
    # 采样关键点索引
    keypoint_indices = list(range(0, H, sample_rate))
    # 确保最后一个点也是关键点
    if keypoint_indices[-1] != H - 1:
        keypoint_indices.append(H - 1)
    
    num_keypoints = len(keypoint_indices)
    if num_keypoints < 2:
        return [ActionPoint(action=raw_actions[0], is_keypoint=True)]
    
    # 关键点时间和位置
    keypoint_times = np.array([idx * dt for idx in keypoint_indices])
    keypoint_positions = raw_actions[keypoint_indices, :dof]  # (num_keypoints, dof)
    
    # 估计末端速度（用后两个关键点差分）
    end_velocity = (keypoint_positions[-1] - keypoint_positions[-2]) / (keypoint_times[-1] - keypoint_times[-2])
    
    # 为每个关节创建三次样条
    splines = []
    for j in range(dof):
        # 确定边界条件
        if start_velocity is not None:
            # 使用 clamped 边界条件：指定首尾速度
            bc_type = ((1, start_velocity[j]), (1, end_velocity[j]))
        else:
            # 使用 natural 边界条件：首尾加速度为 0
            bc_type = 'natural'
        
        spline = CubicSpline(keypoint_times, keypoint_positions[:, j], bc_type=bc_type)
        splines.append(spline)
    
    # 计算输出点数（从 t=0 到 t=最后一个关键点时间）
    total_time = keypoint_times[-1]
    num_output_points = int(np.ceil(total_time / dt)) + 1
    
    result = []
    for i in range(num_output_points):
        t = i * dt
        if t > total_time:
            t = total_time
        
        # 对每个关节采样样条
        pos = np.array([splines[j](t) for j in range(dof)])
        
        # Gripper 透传：找到对应时刻的原始动作索引
        raw_idx = min(int(t / dt), H - 1)
        gripper_val = raw_actions[raw_idx, dof]
        
        action = np.concatenate([pos, [gripper_val]])
        
        # 判断是否为关键点（用时间匹配）
        is_kp = any(abs(t - kt) < 1e-6 for kt in keypoint_times)
        result.append(ActionPoint(action=action, is_keypoint=is_kp))
    
    logging.info(f"Offline Spline plan: {H} raw -> {len(result)} planned, "
                 f"{num_keypoints} keypoints, sample_rate={sample_rate}")
    
    return result


def offline_ruckig_plan(
    raw_actions: np.ndarray,
    sample_rate: int,
    dt: float,
    max_vel: float,
    max_acc: float,
    max_jerk: float,
    dof: int = 6,
) -> list:
    """
    离线点到点 Ruckig 规划。
    
    Args:
        raw_actions: 原始动作序列 (H, 7)，前 6 列为关节，第 7 列为 gripper
        sample_rate: 采样频率（每 N 个动作采 1 个关键点）
        dt: 控制周期
        max_vel, max_acc, max_jerk: Ruckig 约束参数
        dof: 关节自由度（不含 gripper）
    
    Returns:
        List[ActionPoint]，规划后的动作序列
    """
    if not RUCKIG_AVAILABLE:
        logging.warning("Ruckig not available, returning raw actions as ActionPoints")
        return [ActionPoint(action=a, is_keypoint=(i % sample_rate == 0)) 
                for i, a in enumerate(raw_actions)]
    
    raw_actions = np.asarray(raw_actions, dtype=float)
    H = len(raw_actions)
    if H < 2:
        return [ActionPoint(action=raw_actions[0], is_keypoint=True)] if H == 1 else []
    
    # 采样关键点索引
    keypoint_indices = list(range(0, H, sample_rate))
    # 确保最后一个点也是关键点
    if keypoint_indices[-1] != H - 1:
        keypoint_indices.append(H - 1)
    
    num_keypoints = len(keypoint_indices)
    if num_keypoints < 2:
        return [ActionPoint(action=raw_actions[0], is_keypoint=True)]
    
    # 估计每个关键点的速度
    keypoint_velocities = []
    for kp_idx in keypoint_indices:
        vel = estimate_keypoint_velocity(raw_actions, kp_idx, dt, dof)
        keypoint_velocities.append(vel)
    
    # 逐段 Ruckig 规划
    result = []
    ruckig = Ruckig(dof, dt)
    inp = InputParameter(dof)
    out = OutputParameter(dof)
    
    # 设置约束
    inp.max_velocity = [max_vel] * dof
    inp.max_acceleration = [max_acc] * dof
    inp.max_jerk = [max_jerk] * dof
    
    for seg_idx in range(num_keypoints - 1):
        start_kp_idx = keypoint_indices[seg_idx]
        end_kp_idx = keypoint_indices[seg_idx + 1]
        
        # 起点状态
        inp.current_position = list(raw_actions[start_kp_idx, :dof])
        inp.current_velocity = list(keypoint_velocities[seg_idx])
        inp.current_acceleration = [0.0] * dof
        
        # 终点状态
        inp.target_position = list(raw_actions[end_kp_idx, :dof])
        inp.target_velocity = list(keypoint_velocities[seg_idx + 1])
        inp.target_acceleration = [0.0] * dof
        
        # 设置最小持续时间（作为保底，避免过快）
        segment_steps = end_kp_idx - start_kp_idx
        inp.minimum_duration = segment_steps * dt * 0.9
        
        try:
            # 计算轨迹
            traj = Trajectory(dof)
            calc_result = ruckig.calculate(inp, traj)
            
            if calc_result == Result.ErrorInvalidInput:
                logging.warning(f"Ruckig segment {seg_idx} invalid input, using linear interpolation")
                # 降级：线性插值
                for j in range(segment_steps):
                    t_ratio = j / segment_steps if segment_steps > 0 else 0
                    interp_pos = (1 - t_ratio) * raw_actions[start_kp_idx, :dof] + t_ratio * raw_actions[end_kp_idx, :dof]
                    gripper_val = raw_actions[start_kp_idx + j, dof] if (start_kp_idx + j) < H else raw_actions[-1, dof]
                    action = np.concatenate([interp_pos, [gripper_val]])
                    result.append(ActionPoint(action=action, is_keypoint=(j == 0)))
                continue
            
            # 按 dt 采样轨迹
            duration = traj.duration
            num_samples = max(1, int(np.ceil(duration / dt)))
            
            for j in range(num_samples):
                t = j * dt
                if t > duration:
                    t = duration
                
                # 获取该时刻的位置
                new_pos, new_vel, new_acc = traj.at_time(t)
                
                # 非首段跳过 j=0（因为它和上一段的 duration 点重合）
                if seg_idx > 0 and j == 0:
                    continue
                
                # Gripper 透传：按比例取原始值
                raw_idx = start_kp_idx + int(j * segment_steps / num_samples) if num_samples > 1 else start_kp_idx
                raw_idx = min(raw_idx, H - 1)
                gripper_val = raw_actions[raw_idx, dof]
                
                action = np.concatenate([np.array(new_pos), [gripper_val]])
                is_kp = (seg_idx == 0 and j == 0)  # 只有首段第一个点是关键点
                result.append(ActionPoint(action=action, is_keypoint=is_kp))
            
            # 添加 duration 时刻的点（确保边界对齐）
            last_sample_time = (num_samples - 1) * dt
            if last_sample_time < duration - 1e-6:
                end_pos, end_vel, end_acc = traj.at_time(duration)
                gripper_val = raw_actions[min(end_kp_idx, H - 1), dof]
                action = np.concatenate([np.array(end_pos), [gripper_val]])
                result.append(ActionPoint(action=action, is_keypoint=False))
                
        except Exception as e:
            logging.exception(f"Ruckig segment {seg_idx} planning failed: {e}, using linear interpolation")
            # 降级：线性插值
            for j in range(segment_steps):
                t_ratio = j / segment_steps if segment_steps > 0 else 0
                interp_pos = (1 - t_ratio) * raw_actions[start_kp_idx, :dof] + t_ratio * raw_actions[end_kp_idx, :dof]
                gripper_val = raw_actions[start_kp_idx + j, dof] if (start_kp_idx + j) < H else raw_actions[-1, dof]
                action = np.concatenate([interp_pos, [gripper_val]])
                result.append(ActionPoint(action=action, is_keypoint=(j == 0)))
    
    # 最后一个关键点
    result.append(ActionPoint(action=raw_actions[-1].copy(), is_keypoint=True))
    
    logging.info(f"Offline Ruckig plan: {len(raw_actions)} raw -> {len(result)} planned, "
                 f"{num_keypoints} keypoints, sample_rate={sample_rate}")
    
    return result

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
    # 离线 Ruckig 规划选项
    parser.add_argument("--ruckig_offline", action="store_true", help="启用离线点到点 Ruckig 规划（与 ruckig_enable 互斥）")
    parser.add_argument("--ruckig_sample_rate", type=int, default=10, help="关键点采样频率（每 N 个动作采 1 个关键点）")
    # 离线样条插值规划选项
    parser.add_argument("--spline_offline", action="store_true", help="启用离线三次样条插值规划（与 ruckig_enable/ruckig_offline 互斥）")
    # 日志路径
    parser.add_argument("--log_dir", type=str, default="/home/test/jemotor/dsrl_pi05/openpi/trajectory_plots/csv_data/", help="日志文件保存目录")
    args = parser.parse_args()

    set_seeds(args.seed)

    # 互斥校验
    modes_enabled = sum([args.ruckig_enable, args.ruckig_offline, args.spline_offline])
    if modes_enabled > 1:
        raise ValueError("--ruckig_enable, --ruckig_offline, --spline_offline 互斥，只能选其一")

    # 创建日志记录器
    import os
    from datetime import datetime
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    logger_action = NumpyCSVLogger(os.path.join(args.log_dir, f"action_sent_{log_timestamp}.csv"), mode="w")
    logger_end_pose = NumpyCSVLogger(os.path.join(args.log_dir, f"end_pose_{log_timestamp}.csv"), mode="w")
    logger_joint_state = NumpyCSVLogger(os.path.join(args.log_dir, f"joint_state_{log_timestamp}.csv"), mode="w")
    logger_joint_vel = NumpyCSVLogger(os.path.join(args.log_dir, f"joint_vel_{log_timestamp}.csv"), mode="w")
    logger_merge_events = NumpyCSVLogger(os.path.join(args.log_dir, f"merge_events_{log_timestamp}.csv"), mode="w")
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
        ruckig_offline_enabled = args.ruckig_offline and RUCKIG_AVAILABLE
        spline_offline_enabled = args.spline_offline
        ruckig_instance = None
        ruckig_inp = None
        ruckig_out = None
        ruckig_initialized = False
        ruckig_step_counter = 0  # 稀疏采样计数器
        last_sent_action = None  # 用于降级时保持静止
        prev_sent_action = None  # 上一个发送的动作，用于计算速度
        DOF = 6  # 关节数，不含 gripper

        if ruckig_enabled:
            ruckig_instance = Ruckig(DOF, step_time)
            ruckig_inp = InputParameter(DOF)
            ruckig_out = OutputParameter(DOF)
            # 设置约束
            ruckig_inp.max_velocity = [args.ruckig_max_vel] * DOF
            ruckig_inp.max_acceleration = [args.ruckig_max_acc] * DOF
            ruckig_inp.max_jerk = [args.ruckig_max_jerk] * DOF
            lookahead = args.ruckig_lookahead
            ruckig_inp.minimum_duration = lookahead * step_time
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
                        first_item = action_queue[0]
                        # 支持 ActionPoint 对象和普通 ndarray
                        if hasattr(first_item, 'action'):
                            anchor = np.asarray(first_item.action, dtype=float)
                        else:
                            anchor = np.asarray(first_item, dtype=float)
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

                # 记录 merge 事件时间戳 (用于可视化)
                if print_log:
                    # columns: timestamp, recv_idx, queue_len_before_merge
                    logger_merge_events.log(time.perf_counter(), recv_idx, len(action_queue))

                # 1. 记录未执行的旧动作（从 ActionPoint 中提取 action）
                old_actions = []
                for item in action_queue:
                    if hasattr(item, 'action'):
                        old_actions.append(item.action)
                    else:
                        old_actions.append(item)
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
                
                # ==== 离线 Ruckig 模式：对融合结果进行规划 ====
                if ruckig_offline_enabled:
                    # 将融合结果转为 ndarray
                    smooth_arr = np.array([np.asarray(a) for a in smooth_actions], dtype=float)
                    # 调用离线 Ruckig 规划
                    planned_points = offline_ruckig_plan(
                        smooth_arr,
                        sample_rate=args.ruckig_sample_rate,
                        dt=step_time,
                        max_vel=args.ruckig_max_vel,
                        max_acc=args.ruckig_max_acc,
                        max_jerk=args.ruckig_max_jerk,
                        dof=DOF,
                    )
                    for ap in planned_points:
                        action_queue.append(ap)
                    logging.info(f"Offline Ruckig: {len(smooth_actions)} merged -> {len(planned_points)} planned")
                elif spline_offline_enabled:
                    # ==== 离线样条插值模式：直接用 spline 衔接当前状态和新动作 ====
                    # 不使用 cubic_transition，直接用 spline 的 clamped 边界条件实现平滑衔接
                    
                    # 计算当前速度（用于 clamped 边界条件）
                    if last_sent_action is not None and prev_sent_action is not None:
                        current_pos = np.asarray(last_sent_action[:DOF], dtype=float)
                        current_vel = (current_pos - np.asarray(prev_sent_action[:DOF], dtype=float)) / step_time
                    else:
                        current_pos = None
                        current_vel = None
                    
                    # 构建对齐后的动作序列：用 current_pos 替代 new_actions[start_idx]
                    if current_pos is not None and start_idx < len(new_actions):
                        # 构建 aligned_actions: [current_pos+gripper] + new_actions[start_idx+1:]
                        # Gripper 值使用原始 new_actions[start_idx] 的 gripper
                        gripper_val = new_actions[start_idx, -1] if new_actions.ndim > 1 else 0.0
                        current_action = np.concatenate([current_pos, [gripper_val]])
                        
                        if start_idx + 1 < len(new_actions):
                            aligned_actions = np.vstack([current_action, new_actions[start_idx + 1:]])
                        else:
                            # 只剩一个点，无法拟合样条，直接返回
                            action_queue.append(current_action)
                            waiting_for_infer = False
                            continue
                    else:
                        # 首次推理或无当前状态，直接使用 new_actions
                        aligned_actions = new_actions
                        current_vel = None
                    
                    # 调用样条拟合（使用 clamped 边界条件）
                    planned_points = offline_spline_plan(
                        aligned_actions,
                        sample_rate=args.ruckig_sample_rate,
                        dt=step_time,
                        dof=DOF,
                        start_velocity=current_vel,
                    )
                    
                    # 跳过第一个点（它等于 current_pos，避免重复）
                    skip_first = current_pos is not None and len(planned_points) > 1
                    for idx, ap in enumerate(planned_points):
                        if skip_first and idx == 0:
                            continue
                        action_queue.append(ap)
                    
                    logging.info(f"Offline Spline (direct merge): {len(new_actions)} raw -> {len(aligned_actions)} aligned -> {len(planned_points)} planned, skip_first={skip_first}")
                else:
                    # 普通模式：直接添加融合后的动作
                    for a in smooth_actions:
                        action_queue.append(a)

                waiting_for_infer = False
            except mp.queues.Empty:
                pass

            # ==== 离线 Ruckig 模式：处理推理结果到来时的特殊逻辑 ====
            # 当收到新推理结果且启用离线模式时，需要先执行到下一个关键点
            if ruckig_offline_enabled and waiting_for_infer and len(action_queue) > 0:
                # 检查 out_q 是否有结果（非阻塞查看，不取出）
                pending_result = None
                try:
                    pending_result = out_q.get_nowait()
                except mp.queues.Empty:
                    pending_result = None
                
                if pending_result is not None:
                    # 有新结果，需要执行到下一个关键点
                    # 找到下一个关键点位置
                    next_kp_offset = None
                    for kp_i, ap in enumerate(action_queue):
                        if hasattr(ap, 'is_keypoint') and ap.is_keypoint:
                            next_kp_offset = kp_i
                            break
                    
                    if next_kp_offset is not None and next_kp_offset > 0:
                        # 执行到关键点（立即发送）
                        logging.info(f"Offline Ruckig: executing {next_kp_offset} actions to reach keypoint")
                        for _ in range(next_kp_offset):
                            if len(action_queue) == 0:
                                break
                            ap = action_queue.popleft()
                            action_to_send = ap.action if hasattr(ap, 'action') else ap
                            if print_log:
                                logger_action.log(time.perf_counter(), action_to_send[:7])
                                log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel)
                            robot.send_action_np(action_to_send[:7])
                            last_sent_action = action_to_send
                            action_step_counter += 1
                            # 注意：这里需要 sleep 以保持控制频率
                            time.sleep(step_time)
                    
                    # 现在处理新推理结果
                    idx, action_vals = pending_result
                    recv_idx = idx
                    logging.debug(f"got result #{recv_idx} (offline mode)")
                    
                    if print_log:
                        logger_merge_events.log(time.perf_counter(), recv_idx, len(action_queue))
                    
                    # 提取剩余动作（从关键点开始）
                    old_remaining = []
                    for ap in action_queue:
                        if hasattr(ap, 'action'):
                            old_remaining.append(ap.action)
                        else:
                            old_remaining.append(ap)
                    action_queue.clear()
                    
                    # 对齐新动作起点
                    if args.align_mode == "step":
                        start_idx = action_step_counter
                    elif args.align_mode == "euclidean" and len(old_remaining) > 0 and len(action_vals) > 0:
                        old_action = old_remaining[0]
                        dists = np.linalg.norm(action_vals - old_action, axis=1)
                        start_idx = int(np.argmin(dists))
                    else:
                        start_idx = 0
                    new_actions = action_vals[start_idx:]
                    
                    # 融合
                    merged = cubic_transition(old_remaining, new_actions)
                    
                    # 离线 Ruckig 规划
                    merged_arr = np.array([np.asarray(a) for a in merged], dtype=float)
                    planned_points = offline_ruckig_plan(
                        merged_arr,
                        sample_rate=args.ruckig_sample_rate,
                        dt=step_time,
                        max_vel=args.ruckig_max_vel,
                        max_acc=args.ruckig_max_acc,
                        max_jerk=args.ruckig_max_jerk,
                        dof=DOF,
                    )
                    for ap in planned_points:
                        action_queue.append(ap)
                    
                    waiting_for_infer = False
                    action_step_counter = 0
                    logging.info(f"Offline Ruckig merge: {len(old_remaining)} old + {len(new_actions)} new -> {len(planned_points)} planned")

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
                    
                    # ===== 队列即将清空时的减速保护 =====
                    # 当队列剩余动作 < 5 时，保持最后目标不变，减速停止
                    QUEUE_LOW_THRESHOLD = 5
                    queue_len = len(action_queue)
                    
                    if queue_len < QUEUE_LOW_THRESHOLD:
                        # 队列即将清空：使用队列最后一个动作作为目标，并减速到 0
                        target_action = action_queue[-1]  # 取最后一个
                        target_pos = np.asarray(target_action[:DOF], dtype=float)
                        ruckig_inp.target_position = list(target_pos)
                        # 目标速度和加速度都设为 0，Ruckig 会自动规划减速停止
                        ruckig_inp.target_velocity = [0.0] * DOF
                        ruckig_inp.target_acceleration = [0.0] * DOF
                        logging.debug(f"Ruckig decel mode: queue={queue_len}, holding last target")
                    else:
                        # 正常模式：取 lookahead 步后的动作作为目标
                        target_idx = min(lookahead - 1, queue_len - 1)
                        target_action = action_queue[target_idx]
                        target_pos = np.asarray(target_action[:DOF], dtype=float)
                        ruckig_inp.target_position = list(target_pos)
                        
                        # 基于运动学估算 target_velocity: v_target = v_current + a_current * Δt
                        duration = step_time * lookahead
                        current_vel = np.array(ruckig_inp.current_velocity)
                        current_acc = np.array(ruckig_inp.current_acceleration)
                        # 限制速度增量，避免加速度过大时估算值爆炸
                        vel_delta = current_acc * duration
                        max_vel_delta = args.ruckig_max_vel * 0.1  # 增量不超过 max_vel 的 10%
                        vel_delta = np.clip(vel_delta, -max_vel_delta, max_vel_delta)
                        target_vel = current_vel + vel_delta
                        # 限制在约束范围内
                        target_vel = np.clip(target_vel, -args.ruckig_max_vel, args.ruckig_max_vel)
                        # 使用 target_velocity = 0 以简化（方案2）
                        ruckig_inp.target_velocity = [0.0] * DOF
                        ruckig_inp.target_acceleration = [0.0] * DOF

                    # Ruckig 更新
                    ruckig_success = False
                    try:
                        result = ruckig_instance.update(ruckig_inp, ruckig_out)
                        ruckig_success = True
                    except Exception as e:
                        # 方案3：回退到 target_velocity = 0 重试
                        logging.warning(f"Ruckig failed with estimated vel, retrying with target_vel=0: {e}")
                        ruckig_inp.target_velocity = [0.0] * DOF
                        try:
                            result = ruckig_instance.update(ruckig_inp, ruckig_out)
                            ruckig_success = True
                        except Exception as e2:
                            logging.exception(f"Ruckig retry also failed: {e2}, falling back to raw action")
                    
                    if ruckig_success:
                        # 发送平滑后的位置
                        smoothed_pos = np.array(ruckig_out.new_position)
                        gripper_val = action_queue[0][DOF]  # gripper 直接透传
                        action_to_send = np.concatenate([smoothed_pos, [gripper_val]])
                        if print_log:
                            logger_action.log(time.perf_counter(), action_to_send[:7])
                            log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel)
                        robot.send_action_np(action_to_send[:7])
                        last_sent_action = action_to_send
                        action_step_counter += 1

                        # 更新 Ruckig 状态
                        ruckig_out.pass_to_input(ruckig_inp)

                        # 每步消耗一个 action
                        action_queue.popleft()
                        # 更新稀疏采样计数器
                        ruckig_step_counter = (ruckig_step_counter + 1) % lookahead
                    else:
                        # 降级：直接发送原始动作
                        action_to_send = action_queue.popleft()
                        if print_log:
                            logger_action.log(time.perf_counter(), action_to_send[:7])
                            log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel)
                        robot.send_action_np(action_to_send[:7])
                        prev_sent_action = last_sent_action
                        last_sent_action = action_to_send
                        action_step_counter += 1
                        ruckig_step_counter = (ruckig_step_counter + 1) % lookahead
                elif ruckig_offline_enabled:
                    # ==== 离线 Ruckig 模式：直接发送预规划的动作 ====
                    ap = action_queue.popleft()
                    action_to_send = ap.action if hasattr(ap, 'action') else ap
                    if print_log:
                        logger_action.log(time.perf_counter(), action_to_send[:7])
                        log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel)
                    robot.send_action_np(action_to_send[:7])
                    prev_sent_action = last_sent_action
                    last_sent_action = action_to_send
                    action_step_counter += 1
                else:
                    # ==== 原始逻辑（无 Ruckig）====
                    action_item = action_queue.popleft()
                    # 处理 ActionPoint 对象
                    if hasattr(action_item, 'action'):
                        action_to_send = action_item.action
                    else:
                        action_to_send = action_item
                    if print_log:
                        logger_action.log(time.perf_counter(), action_to_send[:7])
                        log_robot_state(robot, logger_end_pose, logger_joint_state, logger_joint_vel)
                    robot.send_action_np(action_to_send[:7])
                    prev_sent_action = last_sent_action
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
        logger_joint_vel.close()
        logger_merge_events.close()
        print(f"Logs saved to {args.log_dir}")

if __name__ == "__main__":
    main()