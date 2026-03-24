import numpy as np
import matplotlib.pyplot as plt
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import time
import argparse
import csv
import draccus
import logging
import multiprocessing as mp
import collections
import json
import yaml
import random
import os
import pathlib
from dataclasses import asdict
from scripts.numpy_logger import NumpyCSVLogger

from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.rtc_utils import RTCActionQueue
from openpi.models_pytorch.rtc_utils import RTCConfig
from openpi.models_pytorch.rtc_utils import ExecutionResampler
from openpi.models_pytorch.rtc_utils import build_executed_prefix_reference
from openpi.models_pytorch.rtc_utils import compute_action_chunk_stats
from openpi.models_pytorch.rtc_utils import compute_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import compute_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import compute_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import aggregate_model_latency_traces
from openpi.models_pytorch.rtc_utils import blend_action_chunks
from openpi.models_pytorch.rtc_utils import build_chunk_handoff_dump_payload
from openpi.models_pytorch.rtc_utils import compute_b1_boundary_window_status
from openpi.models_pytorch.rtc_utils import format_b1_guidance_diagnostics
from openpi.models_pytorch.rtc_utils import format_b1_boundary_window_status
from openpi.models_pytorch.rtc_utils import format_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import format_action_chunk_stats
from openpi.models_pytorch.rtc_utils import format_action_array_preview
from openpi.models_pytorch.rtc_utils import format_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import format_checkpoint_action_norm_stats
from openpi.models_pytorch.rtc_utils import format_model_latency_summary
from openpi.models_pytorch.rtc_utils import format_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import get_conservative_rtc_delay_estimate
from openpi.models_pytorch.rtc_utils import get_rtc_boundary_risks
from openpi.models_pytorch.rtc_utils import ModelLatencyCollector
from openpi.models_pytorch.rtc_utils import should_dump_chunk_handoff
from openpi.shared.inference_kwargs import build_policy_infer_kwargs
from openpi.models_pytorch.rtc_utils import should_mark_rtc_delay_estimate_valid
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig

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


def print_action_diagnostics(
    *,
    actions,
    range_label: str,
    chunk_label: str,
    chunk_joint_label: str,
    preview_label: str,
    state=None,
    joint_dims: int = 6,
    print_arrays: bool = False,
    print_array_rows: int = 3,
):
    """Print range, smoothness stats, and optional array preview for one action stage."""

    if actions is None:
        return

    action_arr = np.asarray(actions)
    if action_arr.ndim == 1:
        action_arr = action_arr[None, :]

    if not np.isfinite(action_arr).all():
        print(f"  {range_label} range: [non-finite], shape={action_arr.shape}")
        print(f"  {chunk_label}: contains_nonfinite=True", flush=True)
        if print_arrays:
            print(
                format_action_array_preview(
                    action_arr,
                    label=preview_label,
                    max_rows=print_array_rows,
                ),
                flush=True,
            )
        return

    print(f"  {range_label} range: [{action_arr.min():.4f}, {action_arr.max():.4f}], shape={action_arr.shape}")
    chunk_stats = compute_action_chunk_stats(action_arr, state=state, joint_dims=joint_dims)
    print(format_action_chunk_stats(chunk_stats, label=chunk_label), flush=True)
    chunk_joint_stats = compute_action_chunk_joint_stats(action_arr, state=state, joint_dims=joint_dims)
    print(format_action_chunk_joint_stats(chunk_joint_stats, label=chunk_joint_label), flush=True)
    if print_arrays:
        print(
            format_action_array_preview(
                action_arr,
                label=preview_label,
                max_rows=print_array_rows,
            ),
            flush=True,
        )


def _peek_next_policy_action(action_queue, *, rtc_enabled: bool):
    if rtc_enabled:
        leftover = action_queue.get_processed_left_over()
        if leftover is None or len(leftover) == 0:
            return None
        return np.asarray(leftover[0], dtype=float).copy()
    if len(action_queue) == 0:
        return None
    return np.asarray(action_queue[0], dtype=float).copy()


def _pop_next_policy_action(action_queue, *, rtc_enabled: bool):
    if rtc_enabled:
        action = action_queue.get()
        if action is None:
            return None
        return np.asarray(action, dtype=float).copy()
    if len(action_queue) == 0:
        return None
    return np.asarray(action_queue.popleft(), dtype=float).copy()


def _write_model_profile_json(path: str | None, *, trace, aggregate) -> None:
    if not path:
        return
    payload = {
        "single_trace": asdict(trace) if trace is not None else None,
        "aggregate": asdict(aggregate) if aggregate is not None else None,
    }
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _write_model_profile_csv(path: str | None, *, trace, aggregate) -> None:
    if not path:
        return
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if trace is not None:
        row = {"kind": "single_trace", "mode": trace.mode}
        row.update(trace.component_ms)
        rows.append(row)
    if aggregate is not None:
        row = {"kind": "aggregate_mean", "mode": aggregate.mode, "runs": aggregate.runs}
        row.update(aggregate.mean_ms)
        rows.append(row)
        std_row = {"kind": "aggregate_std", "mode": aggregate.mode, "runs": aggregate.runs}
        std_row.update(aggregate.std_ms)
        rows.append(std_row)

    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_chunk_handoff_dump(
    path: str | os.PathLike[str],
    *,
    payload: dict[str, np.ndarray | int | str],
) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

# ---------- 子进程：推理循环 ----------
def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    config,
    checkpoint_dir,
    args,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    for candidate in (checkpoint_dir / "openpi_stats", checkpoint_dir / "assets" / "openpi_stats"):
        if candidate.exists():
            try:
                checkpoint_norm_stats = _normalize.load(candidate)
                formatted_stats = format_checkpoint_action_norm_stats(checkpoint_norm_stats, joint_dims=6)
                if formatted_stats is not None:
                    print(formatted_stats, flush=True)
            except Exception:
                logging.exception("failed to load checkpoint norm stats from %s", candidate)
            break

    # 1. 只在该进程里加载一次模型 / CUDA
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    profile_collector = ModelLatencyCollector(
        warmup_remaining=max(int(getattr(args, "profile_model_warmup_runs", 0)), 0),
        target_runs=max(int(getattr(args, "profile_model_runs", 0)), 0),
    )

    while True:
        item = in_q.get()
        if item is None:            # 收到结束标识
            del policy
            break
        # item expected to be (idx, obs, anchor) or (idx, obs, anchor, rtc_context)
        rtc_context = None
        if isinstance(item, tuple) and len(item) == 4:
            idx, obs, anchor, rtc_context = item
        elif isinstance(item, tuple) and len(item) == 3:
            idx, obs, anchor = item
        elif isinstance(item, tuple) and len(item) == 2:
            idx, obs = item
            anchor = None
        else:
            # unexpected message, skip
            continue
        start_time = time.time()

        # Prepare RTC kwargs for policy.infer
        infer_kwargs = build_policy_infer_kwargs(
            num_steps=args.num_steps,
            rtc_context=rtc_context,
            profile_model=args.profile_model_enable,
        )

        result = policy.infer(obs, **infer_kwargs)
        infer_time = time.time() - start_time
        print(f"Step {idx}: infer time = {infer_time:.4f} seconds")
        model_latency_trace = result.get("model_latency_trace")
        policy_actions = result.get("actions")
        actions = policy_actions
        # Get raw (normalized) model output for RTC prev_actions — this is pre-transform
        raw_actions = result.get("raw_actions")  # normalized, in model output space
        # ── RTC Debug ──
        if rtc_context:
            print(
                f"  RTC: mode={rtc_context.get('rtc_config').guidance_mode if rtc_context.get('rtc_config') is not None else 'unknown'}, "
                f"inference_delay={rtc_context.get('inference_delay')}, "
                f"prev_actions={'set' if rtc_context.get('prev_actions') is not None else 'None'}, "
                f"processed_leftover={'set' if rtc_context.get('processed_leftover') is not None else 'None'}, "
                f"executed_prefix_ref={'set' if rtc_context.get('executed_prefix_ref') is not None else 'None'}"
            )
        if model_latency_trace is not None:
            profile_status = profile_collector.ingest(model_latency_trace)
            if profile_status == "incomplete":
                print("  Model profiling: skipped incomplete RTC trace (guidance not active yet)", flush=True)
            elif profile_status == "warmup":
                print(
                    f"  Model profiling: warmup trace skipped ({profile_collector.warmup_remaining} remaining before record)",
                    flush=True,
                )
            elif profile_status in {"recorded_first", "recorded"}:
                if profile_status == "recorded_first" and profile_collector.single_trace is not None:
                    print("  Model profiling single trace:", flush=True)
                    print(format_model_latency_summary(profile_collector.single_trace), flush=True)
                print(
                    f"  Model profiling: recorded run {len(profile_collector.traces)}/{profile_collector.target_runs}",
                    flush=True,
                )
                if profile_collector.is_ready():
                    aggregate = aggregate_model_latency_traces(profile_collector.traces)
                    print("  Model profiling aggregate:", flush=True)
                    print(format_model_latency_summary(aggregate), flush=True)
                    _write_model_profile_json(
                        args.profile_model_output_json,
                        trace=profile_collector.single_trace,
                        aggregate=aggregate,
                    )
                    _write_model_profile_csv(
                        args.profile_model_output_csv,
                        trace=profile_collector.single_trace,
                        aggregate=aggregate,
                    )
                    profile_collector.report_written = True
        if raw_actions is not None:
            print_action_diagnostics(
                actions=raw_actions,
                range_label="raw_actions (normalized)",
                chunk_label="raw_chunk",
                chunk_joint_label="raw_chunk_joint",
                preview_label="raw_actions_preview",
                joint_dims=6,
                print_arrays=args.print_action_arrays,
                print_array_rows=args.print_action_array_rows,
            )
        if policy_actions is not None:
            print_action_diagnostics(
                actions=policy_actions,
                range_label="policy_actions (denormalized, pre_postprocess)",
                chunk_label="policy_chunk",
                chunk_joint_label="policy_chunk_joint",
                preview_label="policy_actions_preview",
                state=obs.get("state"),
                joint_dims=6,
                print_arrays=args.print_action_arrays,
                print_array_rows=args.print_action_array_rows,
            )
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
        if actions is not None:
            print_action_diagnostics(
                actions=actions,
                range_label="worker_actions (postprocess)",
                chunk_label="worker_chunk",
                chunk_joint_label="worker_chunk_joint",
                preview_label="worker_actions_preview",
                state=obs.get("state"),
                joint_dims=6,
                print_arrays=args.print_action_arrays,
                print_array_rows=args.print_action_array_rows,
            )
        b1_diagnostics = None
        rtc_cfg = infer_kwargs.get("rtc_config")
        if rtc_cfg is not None and getattr(rtc_cfg, "guidance_mode", "raw_prefix") == "executed_prefix_b1":
            b1_diagnostics = getattr(rtc_cfg, "last_b1_diagnostics", None)
            if b1_diagnostics is not None:
                print(format_b1_guidance_diagnostics(b1_diagnostics), flush=True)
        out_q.put((idx, actions, raw_actions, b1_diagnostics))

def linear_transition(old_actions, new_actions, max_transition_steps: int = 0):
    """线性插值：h(t)=t"""
    return list(
        blend_action_chunks(
            old_actions=old_actions,
            new_actions=new_actions,
            curve="linear",
            max_transition_steps=max_transition_steps,
        )
    )


def cubic_transition(old_actions, new_actions, max_transition_steps: int = 0):
    """三次 Hermite ease-in/out：h(t)=3t^2-2t^3"""
    return list(
        blend_action_chunks(
            old_actions=old_actions,
            new_actions=new_actions,
            curve="cubic",
            max_transition_steps=max_transition_steps,
        )
    )


def quintic_transition(old_actions, new_actions, max_transition_steps: int = 0):
    """五次平滑：h(t)=10t^3-15t^4+6t^5"""
    return list(
        blend_action_chunks(
            old_actions=old_actions,
            new_actions=new_actions,
            curve="quintic",
            max_transition_steps=max_transition_steps,
        )
    )

def ema_transition(old_actions, new_actions, alpha=0.7, max_transition_steps: int = 0):
    """
    指数加权平滑(EMA)，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    alpha: 新动作权重,0~1
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    return list(
        blend_action_chunks(
            old_actions=old_actions,
            new_actions=new_actions,
            curve="ema",
            ema_alpha=alpha,
            max_transition_steps=max_transition_steps,
        )
    )

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
    parser.add_argument("--config", type=str, default="pi05_agileX", help="model config name")
    parser.add_argument("--fps", type=int, required=False, default=30, help="frames per second")
    parser.add_argument("--task", type=str, required=True, help="task prompt")
    parser.add_argument("--id", type=str, required=False, help="robot id", default="left")
    parser.add_argument("--cameras", type=str, required=False, help="camera config yaml", default=None)
    parser.add_argument("--max_relative_target", type=int, required=False, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    parser.add_argument("--action_steps", type=int, required=False, default=20, help="number of action steps to execute before next inference")
    parser.add_argument("--num_steps", type=int, required=False, default=10, help="flow matching denoise steps per inference")
    parser.add_argument("--profile_model_enable", action="store_true", help="启用模型内部耗时 profiling（PyTorch no-RTC/RTC 共用口径）")
    parser.add_argument("--profile_model_warmup_runs", type=int, default=1, help="profiling 统计前跳过的 warmup 次数")
    parser.add_argument("--profile_model_runs", type=int, default=5, help="profiling 统计的有效 inference 次数")
    parser.add_argument("--profile_model_output_json", type=str, default=None, help="profiling 结果 JSON 输出路径")
    parser.add_argument("--profile_model_output_csv", type=str, default=None, help="profiling 结果 CSV 输出路径")
    parser.add_argument("--smooth_type", type=str, default="cubic", choices=["none", "linear", "cubic", "quintic", "ema"], help="动作平滑策略: none/linear/cubic/quintic/ema")
    parser.add_argument("--ema_alpha", type=float, default=0.5, help="EMA平滑时新动作权重alpha,0~1")
    parser.add_argument("--transition_steps", type=int, default=0, help="仅对 chunk 边界前 N 步做 transition。0 表示沿用全 overlap transition")
    parser.add_argument("--align_mode", type=str, default="step", choices=["step", "euclidean"], help="新动作对齐方式: step(步数) 或 euclidean(欧氏距离)")
    # Horizon-level smoothing of the predicted action sequence (uses full predicted horizon)
    parser.add_argument("--horizon_smooth", type=str, default="ema", choices=["none", "moving", "median", "ema"], help="对预测 horizon 进行时序平滑: none/moving/median/ema")
    parser.add_argument("--horizon_window", type=int, default=30, help="窗口大小用于 moving/median 平滑（越大越平滑）。奇数优先")
    parser.add_argument("--horizon_ema_alpha", type=float, default=0.7, help="horizon EMA alpha 用于 horizon_smooth=ema")
    parser.add_argument("--print_action_arrays", action="store_true", help="打印动作序列预览，便于对比模型原始输出、后处理输出和最终入队动作")
    parser.add_argument("--print_action_array_rows", type=int, default=3, help="动作序列预览时打印的 head/tail 行数")
    parser.add_argument("--execution_interp_enable", action="store_true", help="仅用于低频 RTC 验证：在发送层对低频策略动作做补帧插值")
    parser.add_argument("--execution_send_fps", type=float, default=None, help="发送层实际下发频率；默认等于 fps。建议低频验证时设为 20")
    parser.add_argument("--execution_interp_method", type=str, default="linear", choices=["linear"], help="发送层补帧方式。v1 仅支持 linear")
    parser.add_argument("--dump_chunk_handoff_enable", action="store_true", help="保存 prior/new chunk handoff 快照到 .npz，供离线可视化分析")
    parser.add_argument("--dump_chunk_handoff_dir", type=str, default=None, help="chunk handoff .npz 输出目录")
    parser.add_argument("--dump_chunk_handoff_step_min", type=int, default=None, help="仅保存 step_idx >= 该值的 handoff 快照")
    parser.add_argument("--dump_chunk_handoff_step_max", type=int, default=None, help="仅保存 step_idx <= 该值的 handoff 快照")
    parser.add_argument("--dump_chunk_handoff_stride", type=int, default=1, help="每隔 N 个 step_idx 保存一份 handoff 快照")
    # QP-style online optimizer options
    parser.add_argument("--qp_lambda_acc", type=float, default=0.0, help="二阶差分加速惩罚系数（>=0），qp优化时使用。0 表示禁用")
    parser.add_argument("--qp_velocity_limit", type=float, default=0.0, help="可选的每步最大速度（动作单位/step），>0 则启用简单束缚后处理")
    # RTC (Real-Time Chunking) options
    parser.add_argument("--rtc_enable", action="store_true", help="启用 RTC 平滑（替代 chunk 间 transition 平滑）")
    parser.add_argument("--rtc_execution_horizon", type=int, default=10, help="legacy raw RTC 的权重衰减终点；paper raw RTC 会优先使用完整 overlap 区间")
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=10.0, help="RTC 最大 guidance 权重")
    parser.add_argument("--rtc_sigma_d", type=float, default=0.2, help="RTC 先验方差缩放参数")
    parser.add_argument("--rtc_schedule", type=str, default="auto", choices=["auto", "zeros", "ones", "linear", "exp"], help="RTC 权重衰减策略；auto 会按 guidance_mode 选择默认值")
    parser.add_argument("--rtc_guidance_mode", type=str, default="executed_overlap_paper", choices=["raw_prefix", "raw_prefix_paper", "executed_overlap_paper", "executed_prefix_b1"], help="RTC guidance 目标：legacy raw leftover、论文式 raw overlap、论文式执行空间 overlap 或执行空间前缀 B1")
    parser.add_argument("--rtc_prefix_steps", type=int, default=5, help="B1 执行空间 guidance 的前缀步数")
    parser.add_argument("--rtc_anchor_weight", type=float, default=1.0, help="B1 第一帧锚定损失权重")
    parser.add_argument("--rtc_prefix_weight", type=float, default=1.0, help="B1 前缀跟随损失权重")
    parser.add_argument("--rtc_smooth_weight", type=float, default=0.25, help="B1 前缀平滑损失权重")
    parser.add_argument("--rtc_stay_weight", type=float, default=0.25, help="B1 guidance 保守系数")
    parser.add_argument("--rtc_guidance_start_fraction", type=float, default=0.5, help="B1 guidance 在 denoise 后半程开始逐步增强")
    parser.add_argument("--rtc_arm_joint_dims", type=int, default=6, help="B1 guidance 作用的机械臂关节维度数")
    parser.add_argument("--rtc_b1_guided_delta_abs_max", type=float, default=5.0, help="B1 最终 guided_delta 的逐元素绝对值上限")
    # speed or pose
    parser.add_argument("--mode", type=str, required=False, default="pose", help="inference mode")
    # jitter seed
    parser.add_argument("--seed", type=int, required=False, default=10002)
    args = parser.parse_args()

    if args.profile_model_enable:
        logging.warning(
            "Model profiling enabled: no-RTC breakdown runs through the eager path for instrumentation, "
            "so component totals are directly comparable to RTC, but absolute no-RTC totals may differ from the compiled fast path."
        )

    if args.dump_chunk_handoff_enable and not args.dump_chunk_handoff_dir:
        args.dump_chunk_handoff_dir = str(pathlib.Path("scripts/debug/chunk_handoff_dumps").resolve())

    # RTC configuration
    rtc_enabled = args.rtc_enable
    rtc_config = None
    if rtc_enabled:
        rtc_config = RTCConfig(
            enabled=True,
            guidance_mode=args.rtc_guidance_mode,
            prefix_attention_schedule=args.rtc_schedule,
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
            sigma_d=args.rtc_sigma_d,
            prefix_steps=args.rtc_prefix_steps,
            anchor_weight=args.rtc_anchor_weight,
            prefix_weight=args.rtc_prefix_weight,
            smooth_weight=args.rtc_smooth_weight,
            stay_weight=args.rtc_stay_weight,
            guidance_start_fraction=args.rtc_guidance_start_fraction,
            arm_joint_dims=args.rtc_arm_joint_dims,
            b1_guided_delta_abs_max=args.rtc_b1_guided_delta_abs_max,
        )
        if args.smooth_type != "none":
            logging.warning(
                "RTC enabled: forcing smooth_type=none because transition patching would fight with model-side RTC guidance."
            )
            args.smooth_type = "none"
        if args.transition_steps != 0:
            logging.warning("RTC enabled: forcing transition_steps=0")
            args.transition_steps = 0
        if args.qp_lambda_acc and args.qp_lambda_acc > 0:
            logging.warning(
                "RTC enabled: forcing qp_lambda_acc=0 because QP postprocessing would change the executed tail outside the model-side RTC objective."
            )
            args.qp_lambda_acc = 0.0
        if args.align_mode != "step":
            logging.warning("RTC enabled: forcing align_mode=step")
            args.align_mode = "step"
        if args.horizon_smooth != "none":
            logging.warning(
                "RTC enabled: forcing horizon_smooth=none because runtime horizon postprocessing would change the executed tail outside the model-side RTC objective."
            )
            args.horizon_smooth = "none"
        logging.info(f"RTC enabled: {rtc_config}")
        logging.info(f"smooth_type={args.smooth_type}, horizon_smooth={args.horizon_smooth}")

    set_seeds(args.seed)

    log_path = args.checkpoint_dir.rstrip('/') + "_infer_log.csv"
    logger = NumpyCSVLogger(log_path, mode="w")
    print_log = True

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
    config = _config.get_config(args.config)
    checkpoint_dir = args.checkpoint_dir #"/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")

    proc = ctx.Process(
        target=inference_worker,
        args=(in_q, out_q, config, checkpoint_dir, args)
    )
    proc.daemon = True
    proc.start()

    robot.connect()
    prev_main = None
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 600000

    # rows = []
    step = 1
    execution_send_fps = float(args.execution_send_fps if args.execution_send_fps is not None else args.fps)
    execution_interp_enabled = bool(args.execution_interp_enable and execution_send_fps > float(args.fps))
    execution_resampler = None
    if execution_interp_enabled:
        execution_resampler = ExecutionResampler(
            policy_fps=float(args.fps),
            send_fps=execution_send_fps,
            arm_joint_dims=6,
            interp_method=args.execution_interp_method,
        )
        logging.info(
            "execution interpolation enabled: policy_fps=%s send_fps=%s method=%s ratio=%s",
            args.fps,
            execution_send_fps,
            args.execution_interp_method,
            execution_resampler.send_steps_per_policy_step,
        )
    step_time = step / (execution_send_fps if execution_interp_enabled else args.fps)
    prompt = args.task
    tokenizer = PaligemmaTokenizer()
    tokenized, mask = tokenizer.tokenize(prompt)

    if rtc_enabled:
        action_queue = RTCActionQueue(enabled=True)
        rtc_delay_estimate = 0
        rtc_delay_history = collections.deque(maxlen=8)
        inflight_rtc_request = None
    else:
        action_queue = collections.deque()  # 存储当前动作序列
    waiting_for_infer = False
    action_step_counter = 0  # 记录已执行的动作步数
    first = True
    inflight_state_for_logging = None
    last_sent_action = None
    last_policy_action = None
    last_raw_chunk_for_dump = None
    last_executed_chunk_for_dump = None
    first_send_after_queue_update_pending = False

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
            if rtc_enabled:
                pending_actions = action_queue.get_processed_left_over()
                if pending_actions is not None and len(pending_actions) > 0:
                    try:
                        anchor = np.asarray(pending_actions[0], dtype=float)
                    except Exception:
                        anchor = None
            elif len(action_queue) > 0:
                try:
                    anchor = np.asarray(action_queue[0], dtype=float)
                except Exception:
                    anchor = None
            # Prepare RTC context if enabled
            rtc_ctx = None
            if rtc_enabled:
                prev_actions = action_queue.get_left_over()
                processed_leftover = action_queue.get_processed_left_over()
                action_index_before_inference = action_queue.get_action_index()
                leftover_len_at_send = 0 if prev_actions is None else len(prev_actions)
                processed_leftover_len = 0 if processed_leftover is None else len(processed_leftover)
                available_overlap = min(
                    max(leftover_len_at_send, processed_leftover_len),
                    int(config.model.action_horizon),
                )
                inflight_rtc_request = {
                    "action_index_before_inference": action_index_before_inference,
                    "leftover_len_at_send": leftover_len_at_send,
                }
                executed_prefix_reference = None
                executed_prefix_prev = None
                executed_prefix_ref = None
                request_delay_estimate = get_conservative_rtc_delay_estimate(
                    base_delay_estimate=rtc_delay_estimate,
                    delay_history=rtc_delay_history,
                    available_overlap=available_overlap,
                )
                if rtc_config.guidance_mode == "executed_prefix_b1":
                    executed_prefix_reference = build_executed_prefix_reference(
                        processed_leftover=processed_leftover,
                        current_state=state7,
                        inference_delay_estimate=request_delay_estimate,
                        prefix_steps=rtc_config.prefix_steps,
                        action_horizon=config.model.action_horizon,
                        arm_joint_dims=rtc_config.arm_joint_dims,
                    )
                    if executed_prefix_reference is not None:
                        executed_prefix_prev = executed_prefix_reference.executed_prefix_prev
                        executed_prefix_ref = executed_prefix_reference.executed_prefix_ref
                rtc_ctx = {
                    "rtc_config": rtc_config,
                    "prev_actions": prev_actions,
                    "processed_leftover": processed_leftover,
                    "inference_delay": request_delay_estimate,
                    "executed_prefix_prev": executed_prefix_prev,
                    "executed_prefix_ref": executed_prefix_ref,
                }
                logging.debug(
                    "RTC request #%s: action_index_before_inference=%s, leftover_len_at_send=%s, processed_leftover_len=%s, available_overlap=%s, delay_estimate=%s, prev_actions=%s, processed_leftover=%s, executed_prefix_ref=%s, delay_history=%s",
                    sent_idx,
                    action_index_before_inference,
                    leftover_len_at_send,
                    processed_leftover_len,
                    available_overlap,
                    request_delay_estimate,
                    "set" if prev_actions is not None else "none",
                    "set" if processed_leftover is not None else "none",
                    "set" if executed_prefix_ref is not None else "none",
                    list(rtc_delay_history),
                )
            try:
                if rtc_ctx is not None:
                    in_q.put_nowait((sent_idx, obs, anchor, rtc_ctx))
                else:
                    in_q.put_nowait((sent_idx, obs, anchor))
                sent_idx += 1
                waiting_for_infer = True
                action_step_counter = 0
                inflight_state_for_logging = np.asarray(obs.get("state")).copy()
            except mp.queues.Full:
                if rtc_enabled:
                    inflight_rtc_request = None
                logging.debug("inference queue full, dropping frame")

        # 2. 如果有新推理结果，立即清空并更新 action_queue
        try:
            result_tuple = out_q.get_nowait()
            b1_diagnostics = None
            # Worker returns (idx, actions, raw_actions, b1_diagnostics)
            if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
                idx, action_vals, raw_action_vals, b1_diagnostics = result_tuple
            else:
                idx, action_vals, raw_action_vals = result_tuple
            recv_idx = idx
            logging.debug(f"got result #{recv_idx}")

            if rtc_enabled:
                if action_vals is None:
                    logging.warning("RTC result missing actions; skipping queue update")
                    waiting_for_infer = False
                    inflight_rtc_request = None
                else:
                    if raw_action_vals is None:
                        raw_action_vals = action_vals

                    action_index_before_inference = 0
                    leftover_len_at_send = 0
                    if inflight_rtc_request is not None:
                        action_index_before_inference = inflight_rtc_request["action_index_before_inference"]
                        leftover_len_at_send = inflight_rtc_request["leftover_len_at_send"]

                    old_raw_actions_arr = action_queue.get_left_over()
                    old_actions_arr = action_queue.get_processed_left_over()
                    real_delay = max(action_queue.get_action_index() - action_index_before_inference, 0)
                    rtc_delay_estimate = real_delay
                    if should_mark_rtc_delay_estimate_valid(leftover_len_at_send):
                        rtc_delay_history.append(real_delay)
                    new_actions = np.asarray(action_vals)
                    raw_action_vals = np.asarray(raw_action_vals)
                    if not (np.isfinite(new_actions).all() and np.isfinite(raw_action_vals).all()):
                        logging.warning(
                            "RTC result #%s contains non-finite values; dropping chunk and keeping the existing queue.",
                            recv_idx,
                        )
                        waiting_for_infer = False
                        inflight_rtc_request = None
                        if action_queue.empty():
                            action_step_counter = max(action_step_counter, args.action_steps)
                        continue
                    rtc_stats = compute_rtc_runtime_stats(
                        prev_raw_left_over=old_raw_actions_arr,
                        prev_processed_left_over=old_actions_arr,
                        new_raw_actions=raw_action_vals,
                        new_processed_actions=new_actions,
                        real_delay=real_delay,
                        leftover_len_at_send=leftover_len_at_send,
                        rtc_execution_horizon=args.rtc_execution_horizon,
                        action_index_before_inference=action_index_before_inference,
                        guidance_mode=rtc_config.guidance_mode,
                        prefix_attention_schedule=rtc_config.prefix_attention_schedule,
                    )

                    print(format_rtc_runtime_stats(rtc_stats, chunk_idx=recv_idx), flush=True)
                    if b1_diagnostics is not None:
                        print(format_b1_guidance_diagnostics(b1_diagnostics, real_delay=real_delay), flush=True)
                        b1_boundary_status = compute_b1_boundary_window_status(b1_diagnostics, real_delay=real_delay)
                        print(format_b1_boundary_window_status(b1_boundary_status), flush=True)
                        if not b1_boundary_status.boundary_hit:
                            print(
                                f"  RTC B1 boundary risk: real boundary fell outside the guided window by "
                                f"{b1_boundary_status.miss_steps} step(s)",
                                flush=True,
                            )
                    elif rtc_config.guidance_mode != "executed_prefix_b1":
                        warning_reasons = get_rtc_boundary_risks(rtc_stats)
                        if warning_reasons:
                            print(f"  RTC boundary risk: {'; '.join(warning_reasons)}", flush=True)

                    action_queue.merge(
                        raw_action_vals,
                        new_actions,
                        real_delay=real_delay,
                        action_index_before_inference=action_index_before_inference,
                    )
                    first_send_after_queue_update_pending = not action_queue.empty()
                    rtc_queue_head = action_queue.get_processed_left_over()
                    rtc_boundary_stats = compute_boundary_jump_stats(
                        previous_action=last_policy_action,
                        next_action=rtc_queue_head[0] if rtc_queue_head is not None and len(rtc_queue_head) > 0 else None,
                        joint_dims=6,
                    )
                    print(format_boundary_jump_stats(rtc_boundary_stats, label="queue_boundary"), flush=True)
                    current_raw_chunk_for_dump = None if raw_action_vals is None else np.asarray(raw_action_vals, dtype=np.float32)
                    current_executed_chunk_for_dump = None if new_actions is None else np.asarray(new_actions, dtype=np.float32)
                    if (
                        last_raw_chunk_for_dump is not None
                        and last_executed_chunk_for_dump is not None
                        and should_dump_chunk_handoff(
                            recv_idx,
                            enabled=args.dump_chunk_handoff_enable,
                            step_min=args.dump_chunk_handoff_step_min,
                            step_max=args.dump_chunk_handoff_step_max,
                            stride=args.dump_chunk_handoff_stride,
                        )
                    ):
                        guided_window_start = -1 if b1_diagnostics is None else int(b1_diagnostics.guided_window_start)
                        dump_payload = build_chunk_handoff_dump_payload(
                            step_idx=recv_idx,
                            mode="rtc",
                            real_delay=real_delay,
                            guided_window_start=guided_window_start,
                            prior_start_index=0,
                            new_start_index=action_index_before_inference,
                            handoff_index=real_delay,
                            raw_prior=last_raw_chunk_for_dump,
                            raw_new=current_raw_chunk_for_dump,
                            executed_prior=last_executed_chunk_for_dump,
                            executed_new=current_executed_chunk_for_dump,
                        )
                        dump_path = pathlib.Path(args.dump_chunk_handoff_dir) / f"step_{recv_idx:06d}_mode_rtc.npz"
                        _write_chunk_handoff_dump(dump_path, payload=dump_payload)
                        print(f"  chunk_handoff_dump: saved {dump_path}", flush=True)
                    last_raw_chunk_for_dump = current_raw_chunk_for_dump
                    last_executed_chunk_for_dump = current_executed_chunk_for_dump
                    waiting_for_infer = False
                    inflight_rtc_request = None
            else:
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
                new_actions = np.asarray(action_vals[start_idx:])
                print_action_diagnostics(
                    actions=new_actions,
                    range_label="main_actions (aligned_from_worker)",
                    chunk_label="main_aligned_chunk",
                    chunk_joint_label="main_aligned_chunk_joint",
                    preview_label="main_aligned_actions_preview",
                    state=inflight_state_for_logging,
                    joint_dims=6,
                    print_arrays=args.print_action_arrays,
                    print_array_rows=args.print_action_array_rows,
                )
                # 对齐/裁剪后仍会再做一次主线程后处理；单独保留日志，便于区分 worker 与 main 的影响。
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
                print_action_diagnostics(
                    actions=new_actions,
                    range_label="main_actions (post_main_postprocess)",
                    chunk_label="main_postprocess_chunk",
                    chunk_joint_label="main_postprocess_chunk_joint",
                    preview_label="main_postprocess_actions_preview",
                    state=inflight_state_for_logging,
                    joint_dims=6,
                    print_arrays=args.print_action_arrays,
                    print_array_rows=args.print_action_array_rows,
                )

                # 3. 平滑衔接（可通过参数切换）
                if args.smooth_type == "none":
                    # No chunk-to-chunk transition smoothing (e.g. when RTC handles it)
                    smooth_actions = list(new_actions)
                elif args.smooth_type == "linear":
                    smooth_actions = linear_transition(old_actions, new_actions, max_transition_steps=args.transition_steps)
                elif args.smooth_type == "cubic":
                    smooth_actions = cubic_transition(old_actions, new_actions, max_transition_steps=args.transition_steps)
                elif args.smooth_type == "quintic":
                    smooth_actions = quintic_transition(old_actions, new_actions, max_transition_steps=args.transition_steps)
                elif args.smooth_type == "ema":
                    smooth_actions = ema_transition(
                        old_actions,
                        new_actions,
                        alpha=args.ema_alpha,
                        max_transition_steps=args.transition_steps,
                    )
                else:
                    raise ValueError(f"Unknown smooth_type: {args.smooth_type}")
                boundary_stats = compute_boundary_jump_stats(
                    previous_action=last_policy_action,
                    next_action=smooth_actions[0] if len(smooth_actions) > 0 else None,
                    joint_dims=6,
                )
                print(format_boundary_jump_stats(boundary_stats, label="queue_boundary"), flush=True)
                current_raw_chunk_for_dump = None
                if raw_action_vals is not None:
                    current_raw_chunk_for_dump = np.asarray(raw_action_vals[start_idx:], dtype=np.float32)
                current_executed_chunk_for_dump = np.asarray(smooth_actions, dtype=np.float32)
                if (
                    last_raw_chunk_for_dump is not None
                    and last_executed_chunk_for_dump is not None
                    and should_dump_chunk_handoff(
                        recv_idx,
                        enabled=args.dump_chunk_handoff_enable,
                        step_min=args.dump_chunk_handoff_step_min,
                        step_max=args.dump_chunk_handoff_step_max,
                        stride=args.dump_chunk_handoff_stride,
                    )
                ):
                    consumed_prior_steps = max(len(last_executed_chunk_for_dump) - len(old_actions), 0)
                    dump_payload = build_chunk_handoff_dump_payload(
                        step_idx=recv_idx,
                        mode="no_rtc",
                        real_delay=0,
                        guided_window_start=-1,
                        prior_start_index=0,
                        new_start_index=consumed_prior_steps,
                        handoff_index=0,
                        raw_prior=last_raw_chunk_for_dump,
                        raw_new=current_raw_chunk_for_dump,
                        executed_prior=last_executed_chunk_for_dump,
                        executed_new=current_executed_chunk_for_dump,
                    )
                    dump_path = pathlib.Path(args.dump_chunk_handoff_dir) / f"step_{recv_idx:06d}_mode_no_rtc.npz"
                    _write_chunk_handoff_dump(dump_path, payload=dump_payload)
                    print(f"  chunk_handoff_dump: saved {dump_path}", flush=True)
                last_raw_chunk_for_dump = current_raw_chunk_for_dump
                last_executed_chunk_for_dump = current_executed_chunk_for_dump
                print_action_diagnostics(
                    actions=np.asarray(smooth_actions),
                    range_label="queue_actions (post_transition)",
                    chunk_label="queue_chunk",
                    chunk_joint_label="queue_chunk_joint",
                    preview_label="queue_actions_preview",
                    state=inflight_state_for_logging,
                    joint_dims=6,
                    print_arrays=args.print_action_arrays,
                    print_array_rows=args.print_action_array_rows,
                )
                for a in smooth_actions:
                    action_queue.append(a)
                first_send_after_queue_update_pending = len(action_queue) > 0

                waiting_for_infer = False
        except mp.queues.Empty:
            pass

        # 3. 如果 action_queue 有动作，发给 robot
        action_to_send = None
        has_action = False
        policy_action_consumed = False
        consumed_policy_action = None
        if execution_interp_enabled:
            if execution_resampler is None:
                raise RuntimeError("execution interpolation enabled without a resampler")
            if execution_resampler.needs_policy_action():
                next_policy_action = _pop_next_policy_action(action_queue, rtc_enabled=rtc_enabled)
                if next_policy_action is not None:
                    next_policy_action = np.asarray(next_policy_action[:7], dtype=float)
                    execution_resampler.start_policy_step(next_policy_action)
                    policy_action_consumed = True
                    consumed_policy_action = next_policy_action.copy()
                elif execution_resampler.current_policy_action is not None:
                    action_to_send = execution_resampler.hold_current()
                    has_action = True
            if not has_action and execution_resampler.current_policy_action is not None:
                upcoming_policy_action = _peek_next_policy_action(action_queue, rtc_enabled=rtc_enabled)
                if upcoming_policy_action is not None:
                    upcoming_policy_action = np.asarray(upcoming_policy_action[:7], dtype=float)
                action_to_send = execution_resampler.sample(next_policy_action=upcoming_policy_action)
                has_action = True
        else:
            if rtc_enabled:
                action_to_send = action_queue.get()
                has_action = action_to_send is not None
            else:
                has_action = bool(action_queue)
                action_to_send = action_queue.popleft() if has_action else None
            if has_action:
                action_to_send = np.asarray(action_to_send[:7], dtype=float)
                consumed_policy_action = action_to_send.copy()
        if has_action:
            if not np.isfinite(action_to_send).all():
                logging.warning("dropping non-finite action before sending to robot")
                if rtc_enabled and action_queue.empty() and not waiting_for_infer:
                    action_step_counter = max(action_step_counter, args.action_steps)
                continue
            first_send_after_queue_update = False
            if execution_interp_enabled:
                first_send_after_queue_update = first_send_after_queue_update_pending and policy_action_consumed
            else:
                first_send_after_queue_update = first_send_after_queue_update_pending
            if first_send_after_queue_update:
                first_send_stats = compute_boundary_jump_stats(
                    previous_action=last_sent_action,
                    next_action=action_to_send,
                    joint_dims=6,
                )
                print(format_boundary_jump_stats(first_send_stats, label="first_send_after_queue_update"), flush=True)
                actual_state = None
                try:
                    actual_joint_state = robot.get_joint_state()
                    if actual_joint_state is not None:
                        actual_state = np.asarray(actual_joint_state.get("state"), dtype=float)
                except Exception:
                    logging.exception("failed to read actual joint state before first send after queue update")
                actual_state_stats = compute_boundary_jump_stats(
                    previous_action=actual_state,
                    next_action=action_to_send,
                    joint_dims=6,
                )
                print(
                    format_boundary_jump_stats(
                        actual_state_stats,
                        label="actual_state_to_first_sent_action",
                    ),
                    flush=True,
                )
                first_send_after_queue_update_pending = False
            if execution_interp_enabled and policy_action_consumed:
                send_boundary_stats = compute_boundary_jump_stats(
                    previous_action=last_sent_action,
                    next_action=action_to_send,
                    joint_dims=6,
                )
                print(format_boundary_jump_stats(send_boundary_stats, label="send_boundary"), flush=True)
            if print_log:
                logger.log(action_to_send[:7])
            robot.send_action_np(action_to_send[:7])
            last_sent_action = action_to_send.copy()
            if consumed_policy_action is not None:
                last_policy_action = consumed_policy_action.copy()
                action_step_counter += 1
            # print(f'publish an action:{time.perf_counter()},action counter:{action_step_counter}')

        # 2.5 统计
        i += 1
        dt_s = time.perf_counter() - t0
        # print(f"loop {i} dt={dt_s:.3f} s")
        time.sleep(max(step_time - dt_s,0))

    # ==== 3. 结束 ====
    in_q.put(None)      # 通知子进程退出
    proc.join()
    robot.disconnect()

if __name__ == "__main__":
    main()
