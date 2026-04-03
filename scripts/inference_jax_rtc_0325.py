import argparse
import collections
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import random
import time

import numpy as np
import yaml

from openpi.models import rtc_utils_jax
from openpi.models_pytorch.rtc_utils import build_chunk_handoff_dump_payload
from openpi.models_pytorch.rtc_utils import compute_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import compute_action_chunk_stats
from openpi.models_pytorch.rtc_utils import compute_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import compute_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import format_action_array_preview
from openpi.models_pytorch.rtc_utils import format_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import format_action_chunk_stats
from openpi.models_pytorch.rtc_utils import format_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import format_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import get_rtc_boundary_risks
from openpi.models_pytorch.rtc_utils import should_dump_chunk_handoff
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config as _policy_config
from openpi.shared.inference_kwargs import build_policy_infer_kwargs
from openpi.training import config as _config
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig


RTC_ON_ROBOT_CHECKLIST = (
    "First RTC comparison: check overlap_l2_after, queue_boundary abs_max, executed chunk handoff PNGs, "
    "and infer_ms. Success means executed-overlap metrics improve vs raw mode without unacceptable latency regression."
)


class JAXRTCActionQueue:
    """Track raw and processed chunks for the JAX RTC runtime."""

    def __init__(self):
        self.queue: np.ndarray | None = None
        self.raw_queue: np.ndarray | None = None
        self.last_index = 0

    def get(self) -> np.ndarray | None:
        if self.queue is None or self.last_index >= len(self.queue):
            return None
        action = self.queue[self.last_index].copy()
        self.last_index += 1
        return action

    def empty(self) -> bool:
        return self.qsize() <= 0

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return len(self.queue) - self.last_index

    def get_action_index(self) -> int:
        return self.last_index

    def get_raw_left_over(self) -> np.ndarray | None:
        if self.raw_queue is None:
            return None
        return self.raw_queue[self.last_index :].copy()

    def get_processed_left_over(self) -> np.ndarray | None:
        if self.queue is None:
            return None
        return self.queue[self.last_index :].copy()

    def merge(
        self,
        *,
        raw_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int,
    ) -> None:
        indexes_diff = self.last_index - int(action_index_before_inference)
        if indexes_diff != int(real_delay):
            raise ValueError(
                f"Action queue delay mismatch: indexes diff {indexes_diff} != real delay {real_delay}"
            )

        real_delay = max(int(real_delay), 0)
        self.raw_queue = np.asarray(raw_actions, dtype=float)[real_delay:].copy()
        self.queue = np.asarray(processed_actions, dtype=float)[real_delay:].copy()
        self.last_index = 0


def make_camera_config(cfg: dict):
    camera_type = cfg.get("type")
    if camera_type == "opencv":
        return OpenCVCameraConfig(
            index_or_path=cfg["index_or_path"],
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )
    if camera_type == "orbbec":
        return OrbbecCameraConfig(
            index_or_path=cfg["index_or_path"],
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )
    raise ValueError(f"Unsupported camera type: {camera_type!r}")


def emit(message: str) -> None:
    print(message, flush=True)


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
) -> None:
    if actions is None:
        return

    action_arr = np.asarray(actions)
    if action_arr.ndim == 1:
        action_arr = action_arr[None, :]

    if not np.isfinite(action_arr).all():
        emit(f"  {range_label} range: [non-finite], shape={action_arr.shape}")
        emit(f"  {chunk_label}: contains_nonfinite=True")
        if print_arrays:
            emit(
                format_action_array_preview(
                    action_arr,
                    label=preview_label,
                    max_rows=print_array_rows,
                )
            )
        return

    emit(f"  {range_label} range: [{action_arr.min():.4f}, {action_arr.max():.4f}], shape={action_arr.shape}")
    chunk_stats = compute_action_chunk_stats(action_arr, state=state, joint_dims=joint_dims)
    emit(format_action_chunk_stats(chunk_stats, label=chunk_label))
    chunk_joint_stats = compute_action_chunk_joint_stats(action_arr, state=state, joint_dims=joint_dims)
    emit(format_action_chunk_joint_stats(chunk_joint_stats, label=chunk_joint_label))
    if print_arrays:
        emit(
            format_action_array_preview(
                action_arr,
                label=preview_label,
                max_rows=print_array_rows,
            )
        )


def set_seeds(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def build_policy_observation(robot, *, tokenized_prompt: np.ndarray, tokenized_prompt_mask: np.ndarray) -> dict:
    obs = robot.get_observation()
    obs["state"] = np.asarray(obs["state"])
    obs["tokenized_prompt"] = tokenized_prompt[None]
    obs["tokenized_prompt_mask"] = tokenized_prompt_mask[None]
    obs["token_ar_mask"] = None
    obs["token_loss_mask"] = None
    return obs


def pad_prev_actions(prev_actions: np.ndarray | None, *, action_horizon: int, action_dim: int) -> np.ndarray:
    padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
    if prev_actions is None:
        return padded
    prev_actions = np.asarray(prev_actions, dtype=np.float32)
    if prev_actions.ndim == 1:
        prev_actions = prev_actions[None, :]
    copy_horizon = min(prev_actions.shape[0], action_horizon)
    copy_dim = min(prev_actions.shape[1], action_dim)
    padded[:copy_horizon, :copy_dim] = prev_actions[:copy_horizon, :copy_dim]
    return padded


def pad_processed_leftover(
    processed_leftover: np.ndarray | None,
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
    if processed_leftover is None:
        return padded
    processed_leftover = np.asarray(processed_leftover, dtype=np.float32)
    if processed_leftover.ndim == 1:
        processed_leftover = processed_leftover[None, :]
    copy_horizon = min(processed_leftover.shape[0], action_horizon)
    copy_dim = min(processed_leftover.shape[1], action_dim)
    padded[:copy_horizon, :copy_dim] = processed_leftover[:copy_horizon, :copy_dim]
    return padded


def build_rtc_context(
    *,
    rtc_config,
    prev_actions: np.ndarray | None,
    processed_leftover: np.ndarray | None,
    processed_leftover_len: int | None = None,
    inference_delay: int,
    execution_horizon: int,
) -> dict:
    rtc_context = {
        "rtc_config": rtc_config,
        "prev_actions": prev_actions,
        "inference_delay": int(inference_delay),
        "execution_horizon": int(execution_horizon),
    }
    if rtc_config is not None and rtc_utils_jax.resolve_rtc_mode(rtc_config) == "executed_overlap_paper":
        rtc_context["processed_leftover"] = processed_leftover
        rtc_context["processed_leftover_len"] = 0 if processed_leftover_len is None else int(processed_leftover_len)
    return rtc_context


def resolve_runtime_guidance_mode(rtc_mode: str) -> str:
    return "raw_prefix_paper" if rtc_mode == "raw_paper" else rtc_mode


def resolve_execution_horizon(*, delay_estimate: int, s_min: int, action_horizon: int) -> int:
    trigger = max(int(delay_estimate), int(s_min))
    return min(max(trigger, 1), max(int(action_horizon) - 1, 1))


def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    config,
    checkpoint_dir: str,
    args,
) -> None:
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    executed_transform_spec = None
    if args.rtc_mode == "executed_overlap_paper":
        executed_transform_spec = getattr(policy._model, "rtc_executed_prefix_transform_spec", None)
        if executed_transform_spec is None:
            raise RuntimeError(
                "executed_overlap_paper RTC mode requires policy._model.rtc_executed_prefix_transform_spec, "
                "but the loaded model did not provide one"
            )

    while True:
        item = in_q.get()
        if item is None:
            del policy
            return

        idx, obs, rtc_context = item
        if rtc_context and rtc_context.get("rtc_config") is not None:
            rtc_mode = rtc_utils_jax.resolve_rtc_mode(rtc_context["rtc_config"])
            if rtc_mode == "executed_overlap_paper":
                rtc_context = dict(rtc_context)
                rtc_context["executed_transform_spec"] = executed_transform_spec
        infer_kwargs = build_policy_infer_kwargs(
            num_steps=args.num_steps,
            rtc_context=rtc_context,
            profile_model=False,
        )

        start_time = time.perf_counter()
        result = policy.infer(obs, **infer_kwargs)
        infer_ms = (time.perf_counter() - start_time) * 1000.0
        out_q.put((idx, result["actions"], result["raw_actions"], infer_ms))


def request_chunk(
    *,
    in_q: mp.Queue,
    out_q: mp.Queue,
    request_idx: int,
    obs: dict,
    rtc_context: dict | None,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    in_q.put((request_idx, obs, rtc_context))
    return out_q.get()


def _write_chunk_handoff_dump(
    path: str | os.PathLike[str],
    *,
    payload: dict[str, np.ndarray | int | str],
) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _write_chunk_handoff_plots(
    *,
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    joint_dims: int = 6,
) -> None:
    from scripts import plot_chunk_handoff as _plot_chunk_handoff

    npz_path = pathlib.Path(input_path)
    plot_dir = pathlib.Path(output_dir)
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = {
            "step_idx": int(data["step_idx"]),
            "mode": str(data["mode"]),
            "real_delay": int(data["real_delay"]),
            "guided_window_start": int(data["guided_window_start"]),
            "prior_start_index": int(data["prior_start_index"]) if "prior_start_index" in data else 0,
            "new_start_index": int(data["new_start_index"]) if "new_start_index" in data else None,
            "handoff_index": int(data["handoff_index"]),
        }
        handoff_abs = metadata["handoff_index"] if metadata["new_start_index"] is None else int(metadata["new_start_index"]) + int(metadata["handoff_index"])
        metadata["handoff_abs_index"] = handoff_abs
        for space in ("raw", "executed"):
            prior_key = f"{space}_prior"
            new_key = f"{space}_new"
            _plot_chunk_handoff._plot_space(
                npz_path=npz_path,
                output_dir=plot_dir,
                space=space,
                prior=data[prior_key],
                new=data[new_key],
                prior_start_index=int(metadata["prior_start_index"]),
                new_start_index=metadata["new_start_index"],
                handoff_index=int(metadata["handoff_index"]),
                metadata=metadata,
                joint_dims=joint_dims,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone JAX RTC inference for pi05_agileX_thor_jax")
    parser.add_argument("--port", type=str, required=True, help="robot port name")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="path to checkpoint directory")
    parser.add_argument("--config", type=str, default="pi05_agileX_thor_jax", help="train config name")
    parser.add_argument("--fps", type=int, default=30, help="control frequency")
    parser.add_argument("--task", type=str, required=True, help="task prompt")
    parser.add_argument("--id", type=str, default="left", help="robot id")
    parser.add_argument("--cameras", type=str, default=None, help="camera config yaml")
    parser.add_argument("--max_relative_target", type=int, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    parser.add_argument("--num_steps", type=int, default=5, help="flow denoising steps per inference")
    parser.add_argument("--rtc_beta", type=float, default=5.0, help="paper RTC guidance clipping beta")
    parser.add_argument(
        "--rtc_mode",
        type=str,
        default="raw_paper",
        choices=("raw_paper", "executed_overlap_paper"),
        help="RTC guidance mode",
    )
    parser.add_argument("--rtc_s_min", type=int, default=25, help="minimum execution horizon before starting next inference")
    parser.add_argument("--rtc_initial_delay", type=int, default=0, help="initial conservative delay estimate")
    parser.add_argument("--rtc_delay_buffer_size", type=int, default=10, help="buffer size for conservative delay estimate")
    parser.add_argument("--print_action_arrays", action="store_true", help="print compact raw/executed action previews for each chunk")
    parser.add_argument("--print_action_array_rows", type=int, default=3, help="number of head/tail rows to show when printing action previews")
    parser.add_argument("--dump_chunk_handoff_enable", action="store_true", help="save prior/new chunk handoff snapshots to .npz for offline visualization")
    parser.add_argument("--dump_chunk_handoff_dir", type=str, default=None, help="output directory for chunk handoff .npz dumps")
    parser.add_argument("--dump_chunk_handoff_step_min", type=int, default=None, help="only dump handoff snapshots for step_idx >= this value")
    parser.add_argument("--dump_chunk_handoff_step_max", type=int, default=None, help="only dump handoff snapshots for step_idx <= this value")
    parser.add_argument("--dump_chunk_handoff_stride", type=int, default=1, help="save one handoff snapshot every N chunks")
    parser.add_argument("--plot_chunk_handoff_enable", action="store_true", help="render PNG plots alongside the handoff .npz dumps")
    parser.add_argument("--plot_chunk_handoff_dir", type=str, default=None, help="output directory for rendered chunk handoff plots")
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument("--max_steps", type=int, default=600000)
    args = parser.parse_args()

    if args.plot_chunk_handoff_enable:
        args.dump_chunk_handoff_enable = True
    if args.dump_chunk_handoff_enable and not args.dump_chunk_handoff_dir:
        args.dump_chunk_handoff_dir = str(pathlib.Path("scripts/debug/jax_chunk_handoff_dumps").resolve())
    if args.plot_chunk_handoff_enable and not args.plot_chunk_handoff_dir:
        args.plot_chunk_handoff_dir = str(pathlib.Path("scripts/debug/jax_chunk_handoff_plots").resolve())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    set_seeds(args.seed)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if (checkpoint_dir / "model.safetensors").exists():
        raise ValueError(
            "inference_jax_rtc_0325.py is intended for JAX checkpoints; found model.safetensors in checkpoint_dir"
        )

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
    robot = AlohaAgileXFollower(robot_config)

    config = _config.get_config(args.config)
    action_horizon = int(config.model.action_horizon)
    action_dim = int(config.model.action_dim)
    rtc_model_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=float(args.rtc_beta), mode=args.rtc_mode)

    ctx = mp.get_context("spawn")
    in_q: mp.Queue = ctx.Queue(maxsize=2)
    out_q: mp.Queue = ctx.Queue(maxsize=2)
    proc = ctx.Process(target=inference_worker, args=(in_q, out_q, config, str(checkpoint_dir), args))
    proc.daemon = True
    proc.start()

    tokenizer = PaligemmaTokenizer()
    tokenized_prompt, tokenized_prompt_mask = tokenizer.tokenize(args.task)

    robot.connect()
    try:
        initial_obs = build_policy_observation(
            robot,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        idx, initial_actions, initial_raw_actions, infer_ms = request_chunk(
            in_q=in_q,
            out_q=out_q,
            request_idx=0,
            obs=initial_obs,
            rtc_context=None,
        )
        if idx != 0:
            raise RuntimeError(f"Unexpected initial response id: {idx}")

        warmup_execution_horizon = resolve_execution_horizon(
            delay_estimate=max(int(args.rtc_initial_delay), 0),
            s_min=args.rtc_s_min,
            action_horizon=action_horizon,
        )
        warmup_prev_actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
        warmup_processed_leftover_unpadded = (
            initial_actions[warmup_execution_horizon:] if args.rtc_mode == "executed_overlap_paper" else None
        )
        warmup_processed_leftover = pad_processed_leftover(
            warmup_processed_leftover_unpadded,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        warmup_obs = build_policy_observation(
            robot,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        warmup_context = build_rtc_context(
            rtc_config=rtc_model_config,
            prev_actions=warmup_prev_actions,
            processed_leftover=warmup_processed_leftover,
            processed_leftover_len=(
                0 if warmup_processed_leftover_unpadded is None else len(warmup_processed_leftover_unpadded)
            ),
            inference_delay=max(int(args.rtc_initial_delay), 0),
            execution_horizon=int(warmup_execution_horizon),
        )
        warmup_idx, _warmup_actions, _warmup_raw_actions, warmup_infer_ms = request_chunk(
            in_q=in_q,
            out_q=out_q,
            request_idx=-1,
            obs=warmup_obs,
            rtc_context=warmup_context,
        )
        if warmup_idx != -1:
            raise RuntimeError(f"Unexpected RTC warmup response id: {warmup_idx}")
        emit(
            f"RTC warmup ready: mode={args.rtc_mode} infer_ms={warmup_infer_ms:.2f} "
            f"execution_horizon={warmup_execution_horizon} delay_est={warmup_context['inference_delay']} "
            f"processed_leftover_len={warmup_context.get('processed_leftover_len', 'n/a')}"
        )

        action_queue = JAXRTCActionQueue()
        action_queue.merge(
            raw_actions=initial_raw_actions,
            processed_actions=initial_actions,
            real_delay=0,
            action_index_before_inference=0,
        )
        delay_history = collections.deque(
            [max(int(args.rtc_initial_delay), 0)],
            maxlen=max(int(args.rtc_delay_buffer_size), 1),
        )
        inflight_request = None
        next_request_idx = 1
        step_time = 1.0 / float(args.fps)
        last_policy_action = None
        last_raw_chunk_for_dump = np.asarray(initial_raw_actions, dtype=np.float32)
        last_executed_chunk_for_dump = np.asarray(initial_actions, dtype=np.float32)

        emit(
            f"Initial chunk ready: mode={args.rtc_mode} infer_ms={infer_ms:.2f} queue_len={action_queue.qsize()} "
            f"action_horizon={action_horizon} action_dim={action_dim}"
        )
        emit(f"RTC compare checklist: {RTC_ON_ROBOT_CHECKLIST}")
        print_action_diagnostics(
            actions=initial_raw_actions,
            range_label="initial_raw_actions",
            chunk_label="initial_raw_chunk",
            chunk_joint_label="initial_raw_chunk_joint",
            preview_label="initial_raw_actions_preview",
            joint_dims=6,
            print_arrays=args.print_action_arrays,
            print_array_rows=args.print_action_array_rows,
        )
        print_action_diagnostics(
            actions=initial_actions,
            range_label="initial_queue_actions",
            chunk_label="initial_queue_chunk",
            chunk_joint_label="initial_queue_chunk_joint",
            preview_label="initial_queue_actions_preview",
            joint_dims=6,
            print_arrays=args.print_action_arrays,
            print_array_rows=args.print_action_array_rows,
        )

        for step_idx in range(int(args.max_steps)):
            loop_start = time.perf_counter()

            if inflight_request is not None:
                try:
                    idx, processed_actions, raw_actions, infer_ms = out_q.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if idx != inflight_request["idx"]:
                        logging.warning("Dropping unexpected RTC response id=%s (expected %s)", idx, inflight_request["idx"])
                    else:
                        old_raw_actions_arr = action_queue.get_raw_left_over()
                        old_actions_arr = action_queue.get_processed_left_over()
                        real_delay = max(
                            action_queue.get_action_index() - inflight_request["action_index_before_inference"],
                            0,
                        )
                        processed_actions = np.asarray(processed_actions, dtype=np.float32)
                        raw_actions = np.asarray(raw_actions, dtype=np.float32)
                        rtc_stats = compute_rtc_runtime_stats(
                            prev_raw_left_over=old_raw_actions_arr,
                            prev_processed_left_over=old_actions_arr,
                            new_raw_actions=raw_actions,
                            new_processed_actions=processed_actions,
                            real_delay=real_delay,
                            leftover_len_at_send=inflight_request["leftover_len_at_send"],
                            rtc_execution_horizon=inflight_request["execution_horizon"],
                            action_index_before_inference=inflight_request["action_index_before_inference"],
                            guidance_mode=resolve_runtime_guidance_mode(args.rtc_mode),
                            prefix_attention_schedule="exp",
                        )
                        action_queue.merge(
                            raw_actions=raw_actions,
                            processed_actions=processed_actions,
                            real_delay=real_delay,
                            action_index_before_inference=inflight_request["action_index_before_inference"],
                        )
                        delay_history.append(real_delay)
                        emit(
                            f"RTC chunk #{idx} ready: mode={args.rtc_mode} infer_ms={infer_ms:.2f} "
                            f"delay_est={inflight_request['delay_estimate']} real_delay={real_delay} "
                            f"execution_horizon={inflight_request['execution_horizon']} queue_len={action_queue.qsize()}"
                        )
                        emit(format_rtc_runtime_stats(rtc_stats, chunk_idx=idx))
                        warning_reasons = get_rtc_boundary_risks(rtc_stats)
                        if warning_reasons:
                            emit(f"  RTC boundary risk: {'; '.join(warning_reasons)}")
                        print_action_diagnostics(
                            actions=raw_actions,
                            range_label="raw_actions",
                            chunk_label="raw_chunk",
                            chunk_joint_label="raw_chunk_joint",
                            preview_label="raw_actions_preview",
                            state=inflight_request.get("state_for_logging"),
                            joint_dims=6,
                            print_arrays=args.print_action_arrays,
                            print_array_rows=args.print_action_array_rows,
                        )
                        print_action_diagnostics(
                            actions=processed_actions,
                            range_label="queue_actions",
                            chunk_label="queue_chunk",
                            chunk_joint_label="queue_chunk_joint",
                            preview_label="queue_actions_preview",
                            state=inflight_request.get("state_for_logging"),
                            joint_dims=6,
                            print_arrays=args.print_action_arrays,
                            print_array_rows=args.print_action_array_rows,
                        )
                        rtc_queue_head = action_queue.get_processed_left_over()
                        rtc_boundary_stats = compute_boundary_jump_stats(
                            previous_action=last_policy_action,
                            next_action=rtc_queue_head[0] if rtc_queue_head is not None and len(rtc_queue_head) > 0 else None,
                            joint_dims=6,
                        )
                        emit(format_boundary_jump_stats(rtc_boundary_stats, label="queue_boundary"))
                        current_raw_chunk_for_dump = np.asarray(raw_actions, dtype=np.float32)
                        current_executed_chunk_for_dump = np.asarray(processed_actions, dtype=np.float32)
                        if (
                            last_raw_chunk_for_dump is not None
                            and last_executed_chunk_for_dump is not None
                            and should_dump_chunk_handoff(
                                idx,
                                enabled=args.dump_chunk_handoff_enable,
                                step_min=args.dump_chunk_handoff_step_min,
                                step_max=args.dump_chunk_handoff_step_max,
                                stride=args.dump_chunk_handoff_stride,
                            )
                        ):
                            dump_payload = build_chunk_handoff_dump_payload(
                                step_idx=idx,
                                mode=args.rtc_mode,
                                real_delay=real_delay,
                                guided_window_start=inflight_request["delay_estimate"],
                                prior_start_index=0,
                                new_start_index=inflight_request["action_index_before_inference"],
                                handoff_index=real_delay,
                                raw_prior=last_raw_chunk_for_dump,
                                raw_new=current_raw_chunk_for_dump,
                                executed_prior=last_executed_chunk_for_dump,
                                executed_new=current_executed_chunk_for_dump,
                            )
                            dump_path = pathlib.Path(args.dump_chunk_handoff_dir) / f"step_{idx:06d}_mode_{args.rtc_mode}.npz"
                            _write_chunk_handoff_dump(dump_path, payload=dump_payload)
                            emit(f"  chunk_handoff_dump: saved {dump_path}")
                            if args.plot_chunk_handoff_enable:
                                try:
                                    _write_chunk_handoff_plots(
                                        input_path=dump_path,
                                        output_dir=args.plot_chunk_handoff_dir,
                                        joint_dims=6,
                                    )
                                    emit(f"  chunk_handoff_plot: saved {args.plot_chunk_handoff_dir}")
                                except Exception as exc:
                                    logging.warning("Failed to render chunk handoff plots for %s: %s", dump_path, exc)
                        last_raw_chunk_for_dump = current_raw_chunk_for_dump
                        last_executed_chunk_for_dump = current_executed_chunk_for_dump
                        inflight_request = None

            delay_estimate = max(delay_history) if delay_history else 0
            execution_horizon = resolve_execution_horizon(
                delay_estimate=delay_estimate,
                s_min=args.rtc_s_min,
                action_horizon=action_horizon,
            )
            if delay_estimate + execution_horizon > action_horizon:
                logging.warning(
                    "RTC overlap is tight: delay_estimate=%s execution_horizon=%s action_horizon=%s",
                    delay_estimate,
                    execution_horizon,
                    action_horizon,
                )

            if inflight_request is None and not action_queue.empty() and action_queue.get_action_index() >= execution_horizon:
                obs = build_policy_observation(
                    robot,
                    tokenized_prompt=tokenized_prompt,
                    tokenized_prompt_mask=tokenized_prompt_mask,
                )
                current_leftover = action_queue.get_raw_left_over()
                current_processed_leftover = action_queue.get_processed_left_over()
                prev_actions = pad_prev_actions(
                    current_leftover,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
                processed_leftover = pad_processed_leftover(
                    current_processed_leftover,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
                rtc_context = build_rtc_context(
                    rtc_config=rtc_model_config,
                    prev_actions=prev_actions,
                    processed_leftover=processed_leftover,
                    processed_leftover_len=(
                        0 if current_processed_leftover is None else len(current_processed_leftover)
                    ),
                    inference_delay=int(delay_estimate),
                    execution_horizon=int(execution_horizon),
                )
                action_index_before_inference = action_queue.get_action_index()
                leftover_len_at_send = 0 if current_leftover is None else len(current_leftover)
                try:
                    in_q.put_nowait((next_request_idx, obs, rtc_context))
                except queue.Full:
                    logging.warning("Inference queue full; skipping RTC request at step %s", step_idx)
                else:
                    inflight_request = {
                        "idx": next_request_idx,
                        "action_index_before_inference": action_index_before_inference,
                        "leftover_len_at_send": leftover_len_at_send,
                        "delay_estimate": int(delay_estimate),
                        "execution_horizon": int(execution_horizon),
                        "state_for_logging": np.asarray(obs.get("state")).copy(),
                    }
                    emit(
                        f"RTC request #{next_request_idx} sent: mode={args.rtc_mode} action_index={action_index_before_inference} "
                        f"delay_est={delay_estimate} execution_horizon={execution_horizon} leftover={leftover_len_at_send}"
                    )
                    next_request_idx += 1

            action_to_send = action_queue.get()
            if action_to_send is None:
                logging.warning("RTC queue underrun at step %s", step_idx)
                if not proc.is_alive():
                    logging.error("RTC worker exited unexpectedly with exitcode=%s", proc.exitcode)
            else:
                last_policy_action = np.asarray(action_to_send, dtype=np.float32).copy()
                robot.send_action_np(np.asarray(action_to_send[:7], dtype=float))

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(step_time - elapsed, 0.0))
    finally:
        try:
            in_q.put(None)
        except Exception:
            pass
        proc.join(timeout=5)
        robot.disconnect()


if __name__ == "__main__":
    main()
