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
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config as _policy_config
from openpi.shared.inference_kwargs import build_policy_infer_kwargs
from openpi.training import config as _config
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig


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

    while True:
        item = in_q.get()
        if item is None:
            del policy
            return

        idx, obs, rtc_context = item
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
    parser.add_argument("--rtc_s_min", type=int, default=25, help="minimum execution horizon before starting next inference")
    parser.add_argument("--rtc_initial_delay", type=int, default=0, help="initial conservative delay estimate")
    parser.add_argument("--rtc_delay_buffer_size", type=int, default=10, help="buffer size for conservative delay estimate")
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument("--max_steps", type=int, default=600000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    rtc_model_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=float(args.rtc_beta))

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

        logging.info(
            "Initial chunk ready: infer_ms=%.2f queue_len=%s action_horizon=%s action_dim=%s",
            infer_ms,
            action_queue.qsize(),
            action_horizon,
            action_dim,
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
                        real_delay = max(
                            action_queue.get_action_index() - inflight_request["action_index_before_inference"],
                            0,
                        )
                        action_queue.merge(
                            raw_actions=raw_actions,
                            processed_actions=processed_actions,
                            real_delay=real_delay,
                            action_index_before_inference=inflight_request["action_index_before_inference"],
                        )
                        delay_history.append(real_delay)
                        logging.info(
                            "RTC chunk #%s ready: infer_ms=%.2f delay_est=%s real_delay=%s execution_horizon=%s queue_len=%s",
                            idx,
                            infer_ms,
                            inflight_request["delay_estimate"],
                            real_delay,
                            inflight_request["execution_horizon"],
                            action_queue.qsize(),
                        )
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
                prev_actions = pad_prev_actions(
                    action_queue.get_raw_left_over(),
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
                rtc_context = {
                    "rtc_config": rtc_model_config,
                    "prev_actions": prev_actions,
                    "inference_delay": int(delay_estimate),
                    "execution_horizon": int(execution_horizon),
                }
                action_index_before_inference = action_queue.get_action_index()
                try:
                    in_q.put_nowait((next_request_idx, obs, rtc_context))
                except queue.Full:
                    logging.warning("Inference queue full; skipping RTC request at step %s", step_idx)
                else:
                    inflight_request = {
                        "idx": next_request_idx,
                        "action_index_before_inference": action_index_before_inference,
                        "delay_estimate": int(delay_estimate),
                        "execution_horizon": int(execution_horizon),
                    }
                    logging.info(
                        "RTC request #%s sent: action_index=%s delay_est=%s execution_horizon=%s leftover=%s",
                        next_request_idx,
                        action_index_before_inference,
                        delay_estimate,
                        execution_horizon,
                        0 if action_queue.get_raw_left_over() is None else len(action_queue.get_raw_left_over()),
                    )
                    next_request_idx += 1

            action_to_send = action_queue.get()
            if action_to_send is None:
                logging.warning("RTC queue underrun at step %s", step_idx)
            else:
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
