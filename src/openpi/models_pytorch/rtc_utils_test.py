"""Tests for RTC utils (PyTorch version)."""

import importlib.util
import math
import pathlib
import numpy as np

import torch

from openpi import transforms
from openpi.models_pytorch.rtc_utils import ExecutedPrefixReference
from openpi.models_pytorch.rtc_utils import RTCExecutedPrefixTransformSpec
from openpi.models_pytorch.rtc_utils import RTCActionQueue
from openpi.models_pytorch.rtc_utils import ActionChunkStats
from openpi.models_pytorch.rtc_utils import ActionChunkJointStats
from openpi.models_pytorch.rtc_utils import BoundaryJumpStats
from openpi.models_pytorch.rtc_utils import B1GuidanceDiagnostics
from openpi.models_pytorch.rtc_utils import RTCConfig
from openpi.models_pytorch.rtc_utils import apply_rtc_guidance
from openpi.models_pytorch.rtc_utils import build_chunk_handoff_dump_payload
from openpi.models_pytorch.rtc_utils import build_executed_prefix_reference
from openpi.models_pytorch.rtc_utils import compute_b1_prefix_smoothness_loss
from openpi.models_pytorch.rtc_utils import compute_b1_boundary_window_status
from openpi.models_pytorch.rtc_utils import compute_action_chunk_stats
from openpi.models_pytorch.rtc_utils import compute_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import compute_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import build_model_latency_trace
from openpi.models_pytorch.rtc_utils import compute_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import decode_actions_to_executed_prefix_torch
from openpi.models_pytorch.rtc_utils import ExecutionResampler
from openpi.models_pytorch.rtc_utils import aggregate_model_latency_traces
from openpi.models_pytorch.rtc_utils import format_action_chunk_joint_stats
from openpi.models_pytorch.rtc_utils import format_action_chunk_stats
from openpi.models_pytorch.rtc_utils import format_action_array_preview
from openpi.models_pytorch.rtc_utils import format_b1_boundary_window_status
from openpi.models_pytorch.rtc_utils import format_b1_guidance_diagnostics
from openpi.models_pytorch.rtc_utils import format_boundary_jump_stats
from openpi.models_pytorch.rtc_utils import format_checkpoint_action_norm_stats
from openpi.models_pytorch.rtc_utils import format_model_latency_summary
from openpi.models_pytorch.rtc_utils import format_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import get_b1_guided_window_start
from openpi.models_pytorch.rtc_utils import get_rtc_boundary_risks
from openpi.models_pytorch.rtc_utils import get_b1_inference_delay_estimate
from openpi.models_pytorch.rtc_utils import get_prefix_weights
from openpi.models_pytorch.rtc_utils import interpolate_execution_action
from openpi.models_pytorch.rtc_utils import is_complete_model_latency_trace
from openpi.models_pytorch.rtc_utils import ModelLatencyProfiler
from openpi.models_pytorch.rtc_utils import ModelLatencyCollector
from openpi.models_pytorch.rtc_utils import should_dump_chunk_handoff
from openpi.models_pytorch.rtc_utils import should_mark_rtc_delay_estimate_valid
from openpi.models_pytorch.rtc_utils import blend_action_chunks
from openpi.policies import agileX_policy
from openpi.shared.normalize import NormStats


class TestRTCConfig:
    def test_defaults(self):
        cfg = RTCConfig()
        assert cfg.enabled is False
        assert cfg.guidance_mode == "raw_prefix"
        assert cfg.prefix_attention_schedule == "linear"
        assert cfg.max_guidance_weight is None
        assert cfg.execution_horizon == 10
        assert cfg.sigma_d == 1.0
        assert cfg.prefix_steps == 5
        assert cfg.anchor_weight == 1.0
        assert cfg.prefix_weight == 1.0
        assert cfg.smooth_weight == 0.25
        assert cfg.guidance_start_fraction == 0.5
        assert cfg.arm_joint_dims == 6
        assert cfg.preserve_gripper is True
        assert cfg.b1_guided_delta_abs_max == 5.0

    def test_pickle(self):
        """RTCConfig must be picklable for multiprocessing Queue."""
        import pickle

        cfg = RTCConfig(enabled=True, execution_horizon=5, sigma_d=0.5)
        cfg2 = pickle.loads(pickle.dumps(cfg))
        assert cfg2.enabled is True
        assert cfg2.execution_horizon == 5
        assert cfg2.sigma_d == 0.5


class TestGetPrefixWeights:
    def test_linear_basic(self):
        w = get_prefix_weights(start=2, end=8, total=10, schedule="linear")
        assert w.shape == (10,)
        # Frozen prefix
        assert w[0].item() == 1.0
        assert w[1].item() == 1.0
        # Ramp region (6 steps, values should decay)
        assert w[2].item() > w[7].item()
        assert w[2].item() < 1.0
        assert w[7].item() > 0.0
        # Free suffix
        assert w[8].item() == 0.0
        assert w[9].item() == 0.0

    def test_linear_values_match_linspace(self):
        """Verify ramp values match linspace(1,0,steps+2)[1:-1]."""
        w = get_prefix_weights(start=0, end=5, total=10, schedule="linear")
        expected_ramp = torch.linspace(1.0, 0.0, 7)[1:-1]
        torch.testing.assert_close(w[:5], expected_ramp)
        assert (w[5:] == 0.0).all()

    def test_zeros_schedule(self):
        w = get_prefix_weights(start=3, end=7, total=10, schedule="zeros")
        assert (w[:3] == 1.0).all()
        assert (w[3:] == 0.0).all()

    def test_ones_schedule(self):
        w = get_prefix_weights(start=3, end=7, total=10, schedule="ones")
        assert (w[:7] == 1.0).all()
        assert (w[7:] == 0.0).all()

    def test_exp_schedule(self):
        w = get_prefix_weights(start=2, end=8, total=10, schedule="exp")
        assert w.shape == (10,)
        # Frozen prefix should still be ~1 (exp(1)-1)/(e-1) = 1.0
        assert abs(w[0].item() - 1.0) < 1e-5
        assert abs(w[1].item() - 1.0) < 1e-5
        # Ramp should decay faster than linear
        w_lin = get_prefix_weights(start=2, end=8, total=10, schedule="linear")
        # For middle values, exp should be smaller (faster decay)
        assert w[5].item() < w_lin[5].item()
        # Free suffix
        assert abs(w[8].item()) < 1e-5
        assert abs(w[9].item()) < 1e-5

    def test_exp_schedule_matches_lerobot_values(self):
        w = get_prefix_weights(start=5, end=14, total=25, schedule="exp")
        expected = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.7645, 0.5706, 0.4130, 0.2871, 0.1888, 0.1145, 0.0611, 0.0258, 0.0061]
        )
        torch.testing.assert_close(w[:14], expected, atol=1e-4, rtol=0.0)
        assert torch.all(w[14:] == 0.0)

    def test_edge_case_start_equals_end(self):
        w = get_prefix_weights(start=5, end=5, total=10, schedule="linear")
        assert (w[:5] == 1.0).all()
        assert (w[5:] == 0.0).all()

    def test_edge_case_start_greater_than_end(self):
        """start > end should be clamped to start = end."""
        w = get_prefix_weights(start=8, end=5, total=10, schedule="linear")
        assert (w[:5] == 1.0).all()
        assert (w[5:] == 0.0).all()

    def test_edge_case_end_greater_than_total(self):
        w = get_prefix_weights(start=2, end=15, total=10, schedule="linear")
        assert w.shape == (10,)
        assert (w[:2] == 1.0).all()
        # Ramp fills remaining 8 positions
        assert w[2].item() > 0.0

    def test_edge_case_start_zero(self):
        """Most common case: first inference, no delay."""
        w = get_prefix_weights(start=0, end=10, total=50, schedule="linear")
        assert w.shape == (50,)
        assert w[0].item() < 1.0  # No frozen prefix
        assert (w[10:] == 0.0).all()

    def test_edge_case_total_one(self):
        w = get_prefix_weights(start=0, end=1, total=1, schedule="linear")
        assert w.shape == (1,)

    def test_unknown_schedule_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown RTC schedule"):
            get_prefix_weights(start=0, end=5, total=10, schedule="unknown")


class TestApplyRTCGuidance:
    def test_matches_lerobot_reference_behavior(self):
        cfg = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
        x_t = torch.ones(1, 20, 1)
        prev_chunk = torch.full((1, 20, 1), 0.1)

        def mock_denoiser(x):
            return x * 0.5

        result = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=prev_chunk,
            inference_delay=5,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
        )

        expected = torch.tensor(
            [
                [
                    [1.4750],
                    [1.4750],
                    [1.4750],
                    [1.4750],
                    [1.4750],
                    [1.3125],
                    [1.1500],
                    [0.9875],
                    [0.8250],
                    [0.6625],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                    [0.5000],
                ]
            ]
        )

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=0.0)

    def test_pads_short_prev_chunk(self):
        cfg = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=5.0)

        def mock_denoiser(x):
            return x * 0.5

        result = apply_rtc_guidance(
            x_t=torch.randn(10, 6),
            prev_chunk_left_over=torch.randn(5, 6),
            inference_delay=5,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
        )

        assert result.shape == (10, 6)

    def test_b1_falls_back_when_reference_is_missing(self):
        cfg = RTCConfig(enabled=True, guidance_mode="executed_prefix_b1")
        x_t = torch.ones(1, 4, 3, dtype=torch.float32)

        def mock_denoiser(x):
            return x * 0.25

        baseline = mock_denoiser(x_t)
        result = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=2,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=None,
            executed_prefix_ref=None,
            observation_state=torch.zeros(1, 3, dtype=torch.float32),
            executed_prefix_transform_spec=None,
        )

        torch.testing.assert_close(result, baseline, atol=1e-6, rtol=0.0)

    def test_b1_guidance_window_uses_one_step_backoff(self):
        cfg = RTCConfig(
            enabled=True,
            guidance_mode="executed_prefix_b1",
            prefix_steps=2,
            anchor_weight=1.0,
            prefix_weight=1.0,
            smooth_weight=0.0,
            stay_weight=0.0,
            max_guidance_weight=1.0,
            guidance_start_fraction=0.0,
            arm_joint_dims=3,
        )
        x_t = torch.ones(1, 5, 3, dtype=torch.float32)
        observation_state = torch.zeros(1, 3, dtype=torch.float32)
        transform_spec = RTCExecutedPrefixTransformSpec(
            action_mean=np.zeros(3, dtype=np.float32),
            action_std=np.ones(3, dtype=np.float32),
            use_quantiles=False,
            delta_action_mask=None,
            output_joint_flip_mask=None,
        )

        def mock_denoiser(x):
            return torch.zeros_like(x)

        guided_v_t = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=2,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.zeros(3, dtype=np.float32),
            executed_prefix_ref=np.zeros((2, 3), dtype=np.float32),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )
        guided_x1 = x_t - 0.5 * guided_v_t

        torch.testing.assert_close(guided_x1[:, :1], torch.ones(1, 1, 3), atol=1e-5, rtol=0.0)
        assert torch.all(guided_x1[:, 1:3] < 1.0)
        torch.testing.assert_close(guided_x1[:, 3:], torch.ones(1, 2, 3), atol=1e-5, rtol=0.0)

    def test_b1_zero_weights_matches_baseline(self):
        cfg = RTCConfig(
            enabled=True,
            guidance_mode="executed_prefix_b1",
            prefix_steps=2,
            anchor_weight=0.0,
            prefix_weight=0.0,
            smooth_weight=0.0,
            stay_weight=0.0,
            max_guidance_weight=1.0,
            guidance_start_fraction=0.0,
            arm_joint_dims=3,
        )
        x_t = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        observation_state = torch.zeros(1, 3, dtype=torch.float32)
        transform_spec = RTCExecutedPrefixTransformSpec(
            action_mean=np.zeros(3, dtype=np.float32),
            action_std=np.ones(3, dtype=np.float32),
            use_quantiles=False,
            delta_action_mask=None,
            output_joint_flip_mask=None,
        )

        def mock_denoiser(x):
            return x * 0.5

        baseline = mock_denoiser(x_t)
        result = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=1,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.zeros(3, dtype=np.float32),
            executed_prefix_ref=np.zeros((2, 3), dtype=np.float32),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )

        torch.testing.assert_close(result, baseline, atol=1e-6, rtol=0.0)

    def test_b1_remains_finite_under_large_executed_action_scales(self):
        cfg = RTCConfig(
            enabled=True,
            guidance_mode="executed_prefix_b1",
            prefix_steps=2,
            anchor_weight=1.0,
            prefix_weight=1.0,
            smooth_weight=0.25,
            stay_weight=0.25,
            max_guidance_weight=1.0,
            guidance_start_fraction=0.0,
            arm_joint_dims=3,
        )
        x_t = torch.zeros(1, 4, 3, dtype=torch.float32)
        observation_state = torch.zeros(1, 3, dtype=torch.float32)
        transform_spec = RTCExecutedPrefixTransformSpec(
            action_mean=np.zeros(3, dtype=np.float32),
            action_std=np.ones(3, dtype=np.float32),
            action_q01=np.array([-1.0e30, -1.0e30, -1.0e30], dtype=np.float32),
            action_q99=np.array([1.0e30, 1.0e30, 1.0e30], dtype=np.float32),
            use_quantiles=True,
            delta_action_mask=(True, True, True),
            output_joint_flip_mask=None,
            arm_joint_dims=3,
        )

        def mock_denoiser(x):
            return torch.zeros_like(x)

        result = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=1,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            executed_prefix_ref=np.array(
                [
                    [5.0e29, -5.0e29, 2.5e29],
                    [4.0e29, -4.0e29, 2.0e29],
                ],
                dtype=np.float32,
            ),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )

        assert torch.isfinite(result).all()

    def test_b1_guided_delta_clip_is_configurable(self):
        base_kwargs = dict(
            enabled=True,
            guidance_mode="executed_prefix_b1",
            prefix_steps=2,
            anchor_weight=1.0,
            prefix_weight=1.0,
            smooth_weight=0.0,
            stay_weight=0.0,
            max_guidance_weight=20.0,
            guidance_start_fraction=0.0,
            arm_joint_dims=3,
        )
        tight_cfg = RTCConfig(**base_kwargs, b1_guided_delta_abs_max=0.1)
        loose_cfg = RTCConfig(**base_kwargs, b1_guided_delta_abs_max=20.0)
        x_t = torch.ones(1, 5, 3, dtype=torch.float32)
        observation_state = torch.zeros(1, 3, dtype=torch.float32)
        transform_spec = RTCExecutedPrefixTransformSpec(
            action_mean=np.zeros(3, dtype=np.float32),
            action_std=np.ones(3, dtype=np.float32),
            use_quantiles=False,
            delta_action_mask=None,
            output_joint_flip_mask=None,
            arm_joint_dims=3,
        )

        def mock_denoiser(x):
            return torch.zeros_like(x)

        tight_v_t = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=2,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=tight_cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.zeros(3, dtype=np.float32),
            executed_prefix_ref=np.zeros((2, 3), dtype=np.float32),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )
        loose_v_t = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=2,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=loose_cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.zeros(3, dtype=np.float32),
            executed_prefix_ref=np.zeros((2, 3), dtype=np.float32),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )

        tight_x1 = x_t - 0.5 * tight_v_t
        loose_x1 = x_t - 0.5 * loose_v_t
        tight_error = torch.mean(torch.abs(tight_x1[:, 1:3]))
        loose_error = torch.mean(torch.abs(loose_x1[:, 1:3]))

        assert loose_error < tight_error

    def test_b1_nonfinite_reference_falls_back_to_baseline(self):
        cfg = RTCConfig(
            enabled=True,
            guidance_mode="executed_prefix_b1",
            prefix_steps=2,
            anchor_weight=1.0,
            prefix_weight=1.0,
            smooth_weight=0.25,
            stay_weight=0.25,
            max_guidance_weight=1.0,
            guidance_start_fraction=0.0,
            arm_joint_dims=3,
        )
        x_t = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        observation_state = torch.zeros(1, 3, dtype=torch.float32)
        transform_spec = RTCExecutedPrefixTransformSpec(
            action_mean=np.zeros(3, dtype=np.float32),
            action_std=np.ones(3, dtype=np.float32),
            use_quantiles=False,
            delta_action_mask=None,
            output_joint_flip_mask=None,
            arm_joint_dims=3,
        )

        def mock_denoiser(x):
            return x * 0.25

        baseline = mock_denoiser(x_t)
        result = apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=1,
            time=torch.tensor(0.5),
            original_denoise_step_partial=mock_denoiser,
            rtc_config=cfg,
            num_flow_matching_steps=10,
            executed_prefix_prev=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            executed_prefix_ref=np.array(
                [
                    [np.nan, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            observation_state=observation_state,
            executed_prefix_transform_spec=transform_spec,
        )

        torch.testing.assert_close(result, baseline, atol=1e-6, rtol=0.0)


class TestRTCActionQueue:
    def test_merge_replaces_queue_with_real_delay(self):
        queue = RTCActionQueue(enabled=True)
        raw_chunk = np.arange(12, dtype=np.float32).reshape(6, 2)
        processed_chunk = raw_chunk + 100.0
        queue.merge(raw_chunk, processed_chunk, real_delay=0)

        np.testing.assert_array_equal(queue.get_left_over(), raw_chunk)
        np.testing.assert_array_equal(queue.get_processed_left_over(), processed_chunk)

        first_sent = queue.get()
        assert first_sent is not None
        action_index_before_inference = queue.get_action_index()

        second_sent = queue.get()
        assert second_sent is not None
        real_delay = queue.get_action_index() - action_index_before_inference

        next_raw_chunk = np.arange(12, 24, dtype=np.float32).reshape(6, 2)
        next_processed_chunk = next_raw_chunk + 100.0
        queue.merge(
            next_raw_chunk,
            next_processed_chunk,
            real_delay=real_delay,
            action_index_before_inference=action_index_before_inference,
        )

        np.testing.assert_array_equal(queue.get_left_over(), next_raw_chunk[real_delay:])
        np.testing.assert_array_equal(queue.get_processed_left_over(), next_processed_chunk[real_delay:])

    def test_leftover_tracks_truncated_raw_tail_across_chunks(self):
        queue = RTCActionQueue(enabled=True)
        first_raw = np.arange(20, dtype=np.float32).reshape(10, 2)
        first_processed = first_raw + 10.0
        queue.merge(first_raw, first_processed, real_delay=0)

        queue.get()
        queue.get()
        np.testing.assert_array_equal(queue.get_left_over(), first_raw[2:])

        action_index_before_inference = queue.get_action_index()
        queue.get()
        second_raw = np.arange(20, 40, dtype=np.float32).reshape(10, 2)
        second_processed = second_raw + 10.0
        queue.merge(
            second_raw,
            second_processed,
            real_delay=queue.get_action_index() - action_index_before_inference,
            action_index_before_inference=action_index_before_inference,
        )

        queue.get()
        queue.get()
        np.testing.assert_array_equal(queue.get_left_over(), second_raw[3:])


class TestExecutionResampler:
    def test_identity_when_send_fps_matches_policy_fps(self):
        current = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 700.0], dtype=np.float32)
        nxt = np.array([110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 1700.0], dtype=np.float32)

        out = interpolate_execution_action(
            current_action=current,
            next_action=nxt,
            interpolation_fraction=0.0,
            arm_joint_dims=6,
        )

        np.testing.assert_allclose(out, current)

    def test_interpolates_arm_joints_but_not_gripper(self):
        current = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 600.0], dtype=np.float32)
        nxt = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 1600.0], dtype=np.float32)

        out = interpolate_execution_action(
            current_action=current,
            next_action=nxt,
            interpolation_fraction=0.5,
            arm_joint_dims=6,
        )

        np.testing.assert_allclose(out[:6], np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=np.float32))
        assert out[6] == current[6]

    def test_resampler_holds_current_action_without_next_policy_target(self):
        resampler = ExecutionResampler(policy_fps=10.0, send_fps=20.0, arm_joint_dims=6)
        current = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0], dtype=np.float32)

        resampler.start_policy_step(current)
        first = resampler.sample(next_policy_action=None)
        second = resampler.sample(next_policy_action=None)
        hold = resampler.hold_current()

        np.testing.assert_allclose(first, current)
        np.testing.assert_allclose(second, current)
        np.testing.assert_allclose(hold, current)


class TestRTCRuntimeStats:
    def test_reports_when_delay_consumes_all_effective_overlap(self):
        stats = compute_rtc_runtime_stats(
            prev_raw_left_over=np.zeros((0, 2), dtype=np.float32),
            prev_processed_left_over=np.zeros((0, 2), dtype=np.float32),
            new_raw_actions=np.arange(12, dtype=np.float32).reshape(6, 2),
            new_processed_actions=np.arange(12, dtype=np.float32).reshape(6, 2),
            real_delay=4,
            leftover_len_at_send=4,
            rtc_execution_horizon=3,
            action_index_before_inference=7,
        )

        assert stats.action_index_before_inference == 7
        assert stats.leftover_len_at_send == 4
        assert stats.real_delay == 4
        assert stats.rtc_execution_horizon == 3
        assert stats.effective_guided_steps == 0
        assert stats.overlap_len == 0
        assert stats.overlap_l2_before is None
        assert stats.overlap_l2_after is None

    def test_computes_overlap_and_l2_for_raw_and_processed_actions(self):
        stats = compute_rtc_runtime_stats(
            prev_raw_left_over=np.array([[1.0], [2.0]], dtype=np.float32),
            prev_processed_left_over=np.array([[100.0], [200.0]], dtype=np.float32),
            new_raw_actions=np.array([[9.0], [1.0], [2.0], [8.0]], dtype=np.float32),
            new_processed_actions=np.array([[999.0], [110.0], [230.0], [777.0]], dtype=np.float32),
            real_delay=1,
            leftover_len_at_send=4,
            rtc_execution_horizon=3,
            action_index_before_inference=2,
        )

        assert stats.effective_guided_steps == 2
        assert stats.overlap_len == 2
        assert stats.overlap_l2_before == 0.0
        assert math.isclose(stats.overlap_l2_after, math.sqrt(10.0**2 + 30.0**2), rel_tol=0.0, abs_tol=1e-6)

    def test_formats_runtime_stats_for_terminal_output(self):
        stats = compute_rtc_runtime_stats(
            prev_raw_left_over=np.array([[1.0], [2.0]], dtype=np.float32),
            prev_processed_left_over=np.array([[10.0], [20.0]], dtype=np.float32),
            new_raw_actions=np.array([[9.0], [1.0], [2.0], [8.0]], dtype=np.float32),
            new_processed_actions=np.array([[99.0], [10.0], [23.0], [77.0]], dtype=np.float32),
            real_delay=1,
            leftover_len_at_send=4,
            rtc_execution_horizon=3,
            action_index_before_inference=2,
        )

        message = format_rtc_runtime_stats(stats, chunk_idx=3)

        assert message.startswith("  RTC result #3:")
        assert "leftover_len_at_send=4" in message
        assert "real_delay=1" in message
        assert "rtc_execution_horizon=3" in message
        assert "effective_guided_steps=2" in message
        assert "overlap=2" in message
        assert "overlap_l2_before=0.000000" in message
        assert "overlap_l2_after=3.000000" in message

    def test_reports_boundary_risks_from_runtime_stats(self):
        stats = compute_rtc_runtime_stats(
            prev_raw_left_over=np.zeros((0, 2), dtype=np.float32),
            prev_processed_left_over=np.zeros((0, 2), dtype=np.float32),
            new_raw_actions=np.arange(12, dtype=np.float32).reshape(6, 2),
            new_processed_actions=np.arange(12, dtype=np.float32).reshape(6, 2),
            real_delay=4,
            leftover_len_at_send=4,
            rtc_execution_horizon=3,
            action_index_before_inference=7,
        )

        assert get_rtc_boundary_risks(stats) == [
            "delay exhausted leftover tail before the new chunk arrived",
            "delay exceeded rtc_execution_horizon",
            "no effective guided steps remain at the executed boundary",
        ]


class TestB1BoundaryWindow:
    def test_guided_window_start_uses_one_step_backoff(self):
        assert get_b1_guided_window_start(
            inference_delay_estimate=16,
            action_horizon=50,
            prefix_steps=5,
        ) == 15
        assert get_b1_guided_window_start(
            inference_delay_estimate=0,
            action_horizon=50,
            prefix_steps=5,
        ) == 0

    def test_build_executed_prefix_reference_uses_backed_off_window_start(self):
        processed_leftover = np.arange(30, dtype=np.float32).reshape(10, 3)
        reference = build_executed_prefix_reference(
            processed_leftover=processed_leftover,
            current_state=np.array([999.0, 999.0, 999.0], dtype=np.float32),
            inference_delay_estimate=4,
            prefix_steps=3,
            action_horizon=10,
            arm_joint_dims=3,
        )

        assert reference is not None
        assert reference.window_start == 3
        np.testing.assert_allclose(reference.executed_prefix_prev, processed_leftover[2], atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(reference.executed_prefix_ref, processed_leftover[3:6], atol=1e-6, rtol=0.0)

    def test_reports_when_real_boundary_hits_guided_window(self):
        diagnostics = B1GuidanceDiagnostics(
            guided_window_start=15,
            prefix_steps=5,
            delay_estimate=16,
            boundary_l2_before=None,
            boundary_l2_after=None,
            boundary_abs_max_before=None,
            boundary_abs_max_after=None,
            prefix_acc_abs_max_before=None,
            prefix_acc_abs_max_after=None,
            total_loss=None,
            correction_norm_before_clip=None,
            correction_norm_after_clip=None,
            correction_clip_scale=None,
            guidance_weight=None,
            conservative_scale=None,
            guided_delta_l2_after_clip=None,
            guided_delta_abs_max_after_clip=None,
            guided_delta_clip_fraction=None,
        )

        status = compute_b1_boundary_window_status(diagnostics, real_delay=16)

        assert status.boundary_hit is True
        assert status.miss_steps == 0
        message = format_b1_boundary_window_status(status)
        assert "hit=yes" in message
        assert "guided_window=[15, 20)" in message
        assert "real_boundary=16" in message

    def test_reports_when_real_boundary_misses_guided_window(self):
        diagnostics = B1GuidanceDiagnostics(
            guided_window_start=8,
            prefix_steps=5,
            delay_estimate=8,
            boundary_l2_before=None,
            boundary_l2_after=None,
            boundary_abs_max_before=None,
            boundary_abs_max_after=None,
            prefix_acc_abs_max_before=None,
            prefix_acc_abs_max_after=None,
            total_loss=None,
            correction_norm_before_clip=None,
            correction_norm_after_clip=None,
            correction_clip_scale=None,
            guidance_weight=None,
            conservative_scale=None,
            guided_delta_l2_after_clip=None,
            guided_delta_abs_max_after_clip=None,
            guided_delta_clip_fraction=None,
        )

        status = compute_b1_boundary_window_status(diagnostics, real_delay=16)

        assert status.boundary_hit is False
        assert status.miss_steps == 4
        message = format_b1_boundary_window_status(status)
        assert "hit=no" in message
        assert "miss_steps=4" in message

    def test_formats_extended_b1_guidance_diagnostics(self):
        diagnostics = B1GuidanceDiagnostics(
            guided_window_start=3,
            prefix_steps=5,
            delay_estimate=4,
            boundary_l2_before=12.0,
            boundary_l2_after=10.0,
            boundary_abs_max_before=7.0,
            boundary_abs_max_after=5.0,
            prefix_acc_abs_max_before=9.0,
            prefix_acc_abs_max_after=6.0,
            total_loss=1.5,
            correction_norm_before_clip=12.0,
            correction_norm_after_clip=7.5,
            correction_clip_scale=0.625,
            guidance_weight=4.0,
            conservative_scale=0.8,
            guided_delta_l2_after_clip=3.5,
            guided_delta_abs_max_after_clip=1.25,
            guided_delta_clip_fraction=0.2,
        )

        message = format_b1_guidance_diagnostics(diagnostics, real_delay=5)

        assert "total_loss=1.500000" in message
        assert "correction_norm_before_clip=12.000000" in message
        assert "correction_norm_after_clip=7.500000" in message
        assert "correction_clip_scale=0.625000" in message
        assert "guidance_weight=4.000000" in message
        assert "conservative_scale=0.800000" in message
        assert "guided_delta_l2_after_clip=3.500000" in message
        assert "guided_delta_abs_max_after_clip=1.250000" in message
        assert "guided_delta_clip_fraction=0.200000" in message


class TestActionChunkStats:
    def test_computes_chunk_delta_and_accel_statistics(self):
        stats = compute_action_chunk_stats(
            actions=np.array(
                [
                    [1.0, 1.0, 9.0],
                    [2.0, 1.0, 8.0],
                    [4.0, 2.0, 7.0],
                ],
                dtype=np.float32,
            ),
            state=np.array([0.0, 1.0, 99.0], dtype=np.float32),
            joint_dims=2,
        )

        assert isinstance(stats, ActionChunkStats)
        assert stats.joint_dims == 2
        assert math.isclose(stats.state_to_first_l2, 1.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.state_to_first_abs_max, 1.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_delta_l2_mean, (1.0 + math.sqrt(5.0)) / 2.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_delta_l2_max, math.sqrt(5.0), rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_delta_abs_max, 2.0, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_acc_l2_mean, math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_acc_l2_max, math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.step_acc_abs_max, 1.0, rel_tol=0.0, abs_tol=1e-6)

    def test_formats_action_chunk_stats_for_terminal_output(self):
        stats = ActionChunkStats(
            joint_dims=6,
            state_to_first_l2=12.5,
            state_to_first_abs_max=8.0,
            step_delta_l2_mean=4.0,
            step_delta_l2_max=7.0,
            step_delta_abs_max=5.0,
            step_acc_l2_mean=2.0,
            step_acc_l2_max=3.0,
            step_acc_abs_max=1.5,
        )

        message = format_action_chunk_stats(stats, label="action_chunk")

        assert message.startswith("  action_chunk:")
        assert "joint_dims=6" in message
        assert "state_to_first_l2=12.500000" in message
        assert "step_delta_l2_mean=4.000000" in message
        assert "step_delta_l2_max=7.000000" in message
        assert "step_acc_abs_max=1.500000" in message

    def test_computes_jointwise_chunk_stats(self):
        stats = compute_action_chunk_joint_stats(
            actions=np.array(
                [
                    [1.0, 1.0, 9.0],
                    [2.0, -1.0, 8.0],
                    [4.0, 2.0, 7.0],
                ],
                dtype=np.float32,
            ),
            state=np.array([0.0, 2.0, 99.0], dtype=np.float32),
            joint_dims=2,
        )

        assert isinstance(stats, ActionChunkJointStats)
        np.testing.assert_allclose(stats.state_to_first, np.array([1.0, -1.0], dtype=np.float32))
        np.testing.assert_allclose(stats.step_delta_abs_max, np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(stats.step_acc_abs_max, np.array([1.0, 5.0], dtype=np.float32))

    def test_formats_jointwise_chunk_stats_for_terminal_output(self):
        stats = ActionChunkJointStats(
            joint_dims=3,
            state_to_first=np.array([1.0, -2.5, 0.0], dtype=np.float32),
            step_delta_abs_max=np.array([3.0, 4.0, 5.0], dtype=np.float32),
            step_acc_abs_max=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        )

        message = format_action_chunk_joint_stats(stats, label="action_chunk_joint")

        assert message.startswith("  action_chunk_joint:")
        assert "state_to_first=[1.000000, -2.500000, 0.000000]" in message
        assert "step_delta_abs_max=[3.000000, 4.000000, 5.000000]" in message
        assert "step_acc_abs_max=[0.500000, 1.500000, 2.500000]" in message


class TestActionArrayPreview:
    def test_formats_small_action_array_without_truncation(self):
        preview = format_action_array_preview(
            np.array(
                [
                    [1.0, 2.0],
                    [3.5, 4.5],
                ],
                dtype=np.float32,
            ),
            label="smoothed_actions",
            max_rows=3,
            precision=2,
        )

        assert preview.startswith("  smoothed_actions:")
        assert "shape=(2, 2)" in preview
        assert "rows=[[1.00, 2.00], [3.50, 4.50]]" in preview
        assert "head=" not in preview
        assert "tail=" not in preview

    def test_formats_large_action_array_with_head_and_tail_preview(self):
        preview = format_action_array_preview(
            np.arange(24, dtype=np.float32).reshape(6, 4),
            label="queue_actions",
            max_rows=2,
            precision=1,
        )

        assert preview.startswith("  queue_actions:")
        assert "shape=(6, 4)" in preview
        assert "head=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]" in preview
        assert "tail=[[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]]" in preview


class TestBoundaryJumpStats:
    def test_computes_queue_head_jump_against_old_head(self):
        stats = compute_boundary_jump_stats(
            previous_action=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            next_action=np.array([7.0, 25.0, 27.0], dtype=np.float32),
            joint_dims=2,
        )

        assert isinstance(stats, BoundaryJumpStats)
        assert stats.joint_dims == 2
        np.testing.assert_allclose(stats.delta, np.array([-3.0, 5.0], dtype=np.float32))
        assert math.isclose(stats.l2, math.sqrt(34.0), rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(stats.abs_max, 5.0, rel_tol=0.0, abs_tol=1e-6)

    def test_formats_queue_head_jump_for_terminal_output(self):
        stats = BoundaryJumpStats(
            joint_dims=3,
            delta=np.array([1.0, -2.0, 0.5], dtype=np.float32),
            l2=2.5,
            abs_max=2.0,
        )

        message = format_boundary_jump_stats(stats, label="queue_boundary")

        assert message.startswith("  queue_boundary:")
        assert "joint_dims=3" in message
        assert "delta=[1.000000, -2.000000, 0.500000]" in message
        assert "l2=2.500000" in message
        assert "abs_max=2.000000" in message


class TestBlendActionChunks:
    def test_empty_old_actions_returns_new_chunk_unchanged(self):
        blended = blend_action_chunks(
            old_actions=[],
            new_actions=np.array(
                [
                    [100.0, 200.0],
                    [110.0, 210.0],
                ],
                dtype=np.float32,
            ),
            curve="linear",
            max_transition_steps=3,
        )

        np.testing.assert_allclose(
            blended,
            np.array(
                [
                    [100.0, 200.0],
                    [110.0, 210.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
            rtol=0.0,
        )

    def test_linear_transition_can_be_limited_to_short_window(self):
        blended = blend_action_chunks(
            old_actions=np.array(
                [
                    [0.0],
                    [10.0],
                    [20.0],
                    [30.0],
                ],
                dtype=np.float32,
            ),
            new_actions=np.array(
                [
                    [100.0],
                    [110.0],
                    [120.0],
                    [130.0],
                ],
                dtype=np.float32,
            ),
            curve="linear",
            max_transition_steps=2,
        )

        np.testing.assert_allclose(
            blended,
            np.array(
                [
                    [100.0 / 3.0],
                    [230.0 / 3.0],
                    [120.0],
                    [130.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-5,
            rtol=0.0,
        )

    def test_zero_transition_window_keeps_full_overlap_behavior(self):
        blended = blend_action_chunks(
            old_actions=np.array(
                [
                    [0.0],
                    [10.0],
                    [20.0],
                ],
                dtype=np.float32,
            ),
            new_actions=np.array(
                [
                    [100.0],
                    [110.0],
                    [120.0],
                ],
                dtype=np.float32,
            ),
            curve="linear",
            max_transition_steps=0,
        )

        np.testing.assert_allclose(
            blended,
            np.array(
                [
                    [25.0],
                    [60.0],
                    [95.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
            rtol=0.0,
        )

    def test_formats_checkpoint_action_norm_stats(self):
        norm_stats = {
            "actions": NormStats(
                mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                std=np.array([0.5, 1.5, 2.5], dtype=np.float32),
                q01=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
                q99=np.array([4.0, 5.0, 6.0], dtype=np.float32),
            )
        }

        message = format_checkpoint_action_norm_stats(norm_stats, joint_dims=2)

        assert message is not None
        assert message.startswith("Checkpoint action norm stats:")
        assert "mean=[1.000000, 2.000000]" in message
        assert "std=[0.500000, 1.500000]" in message
        assert "q01=[-1.000000, -2.000000]" in message
        assert "q99=[4.000000, 5.000000]" in message


class TestExecutedPrefixB1Helpers:
    def test_torch_decode_matches_numpy_output_pipeline_for_quantile_delta_and_agilex(self):
        raw_actions = torch.tensor(
            [
                [
                    [0.0, 0.5, -0.5, 0.25, -0.25, 0.75, 0.1],
                    [0.5, 0.0, 0.5, -0.25, 0.25, -0.5, -0.2],
                ]
            ],
            dtype=torch.float32,
        )
        state = torch.tensor([[10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 0.7]], dtype=torch.float32)
        norm_stats = NormStats(
            mean=np.zeros(7, dtype=np.float32),
            std=np.ones(7, dtype=np.float32),
            q01=np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -1.0], dtype=np.float32),
            q99=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0], dtype=np.float32),
        )
        spec = RTCExecutedPrefixTransformSpec(
            action_mean=norm_stats.mean,
            action_std=norm_stats.std,
            action_q01=norm_stats.q01,
            action_q99=norm_stats.q99,
            use_quantiles=True,
            delta_action_mask=(True, True, True, True, True, True),
            output_joint_flip_mask=tuple(agileX_policy._joint_flip_mask().tolist()),
            arm_joint_dims=6,
        )

        torch_decoded = decode_actions_to_executed_prefix_torch(
            raw_actions,
            observation_state=state,
            transform_spec=spec,
        )

        pipeline = transforms.Unnormalize({"actions": norm_stats}, use_quantiles=True)({
            "actions": raw_actions[0].cpu().numpy(),
            "state": state[0].cpu().numpy(),
        })
        pipeline = transforms.AbsoluteActions((True, True, True, True, True, True))(pipeline)
        pipeline = agileX_policy.AgileXOutputs(adapt_to_pi=True)(pipeline)

        np.testing.assert_allclose(
            torch_decoded[0].cpu().numpy(),
            np.asarray(pipeline["actions"][:, :6], dtype=np.float32),
            atol=1e-5,
            rtol=0.0,
        )

    def test_build_executed_prefix_reference_uses_backed_off_delay_and_repeats_last_available(self):
        reference = build_executed_prefix_reference(
            processed_leftover=np.array(
                [
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 1.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 1.0],
                    [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 1.0],
                ],
                dtype=np.float32,
            ),
            current_state=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0], dtype=np.float32),
            inference_delay_estimate=2,
            prefix_steps=3,
            action_horizon=50,
            arm_joint_dims=6,
        )

        assert isinstance(reference, ExecutedPrefixReference)
        assert reference.window_start == 1
        np.testing.assert_allclose(reference.executed_prefix_prev, np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]))
        np.testing.assert_allclose(
            reference.executed_prefix_ref,
            np.array(
                [
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
                    [40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_build_executed_prefix_reference_uses_state_when_boundary_has_no_previous_leftover(self):
        reference = build_executed_prefix_reference(
            processed_leftover=np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0]], dtype=np.float32),
            current_state=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0], dtype=np.float32),
            inference_delay_estimate=0,
            prefix_steps=2,
            action_horizon=50,
            arm_joint_dims=6,
        )

        np.testing.assert_allclose(reference.executed_prefix_prev, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        np.testing.assert_allclose(
            reference.executed_prefix_ref,
            np.array(
                [
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_build_executed_prefix_reference_returns_none_when_queue_is_empty(self):
        reference = build_executed_prefix_reference(
            processed_leftover=None,
            current_state=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0], dtype=np.float32),
            inference_delay_estimate=3,
            prefix_steps=5,
            action_horizon=50,
            arm_joint_dims=6,
        )

        assert reference is None

    def test_b1_smoothness_penalizes_boundary_curvature_with_previous_action(self):
        y_prev = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        z = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        loss = compute_b1_prefix_smoothness_loss(y_prev, z)

        assert math.isclose(loss.item(), 8.0, rel_tol=0.0, abs_tol=1e-6)

    def test_delay_estimate_is_only_marked_valid_when_request_had_leftover_tail(self):
        assert should_mark_rtc_delay_estimate_valid(0) is False
        assert should_mark_rtc_delay_estimate_valid(1) is True
        assert should_mark_rtc_delay_estimate_valid(30) is True

    def test_b1_uses_conservative_warmup_delay_estimate_until_first_real_b1_delay(self):
        assert get_b1_inference_delay_estimate(base_delay_estimate=8, b1_delay_estimate_valid=False) == 16
        assert get_b1_inference_delay_estimate(base_delay_estimate=8, b1_delay_estimate_valid=True) == 8
        assert get_b1_inference_delay_estimate(base_delay_estimate=0, b1_delay_estimate_valid=False) == 0


class TestChunkHandoffDumpHelpers:
    def test_should_dump_chunk_handoff_requires_enable(self):
        assert should_dump_chunk_handoff(
            step_idx=10,
            enabled=False,
            step_min=None,
            step_max=None,
            stride=1,
        ) is False

    def test_should_dump_chunk_handoff_respects_range_and_stride(self):
        assert should_dump_chunk_handoff(10, enabled=True, step_min=5, step_max=20, stride=2) is True
        assert should_dump_chunk_handoff(11, enabled=True, step_min=5, step_max=20, stride=2) is False
        assert should_dump_chunk_handoff(4, enabled=True, step_min=5, step_max=20, stride=2) is False
        assert should_dump_chunk_handoff(21, enabled=True, step_min=5, step_max=20, stride=2) is False

    def test_build_chunk_handoff_dump_payload_normalizes_shapes(self):
        payload = build_chunk_handoff_dump_payload(
            step_idx=12,
            mode="rtc",
            real_delay=3,
            guided_window_start=2,
            prior_start_index=0,
            new_start_index=20,
            handoff_index=3,
            raw_prior=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            raw_new=np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            executed_prior=np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32),
            executed_new=np.array([13.0, 14.0, 15.0], dtype=np.float32),
        )

        assert payload["step_idx"] == 12
        assert payload["mode"] == "rtc"
        assert payload["real_delay"] == 3
        assert payload["guided_window_start"] == 2
        assert payload["prior_start_index"] == 0
        assert payload["new_start_index"] == 20
        assert payload["handoff_index"] == 3
        assert payload["raw_prior"].shape == (1, 3)
        assert payload["raw_new"].shape == (1, 3)
        assert payload["executed_prior"].shape == (2, 3)
        assert payload["executed_new"].shape == (1, 3)

    def test_chunk_handoff_plot_indices_use_absolute_chunk_timeline(self):
        script_path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "plot_chunk_handoff.py"
        spec = importlib.util.spec_from_file_location("plot_chunk_handoff", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        prior_x, new_x, handoff_x = module.build_plot_indices(
            prior_len=50,
            new_len=50,
            prior_start_index=0,
            new_start_index=20,
            handoff_index=3,
        )

        assert np.array_equal(prior_x[[0, -1]], np.array([0, 49], dtype=np.int32))
        assert np.array_equal(new_x[[0, -1]], np.array([20, 69], dtype=np.int32))
        assert handoff_x == 23

    def test_chunk_handoff_plot_indices_fall_back_to_legacy_behavior(self):
        script_path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "plot_chunk_handoff.py"
        spec = importlib.util.spec_from_file_location("plot_chunk_handoff", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        prior_x, new_x, handoff_x = module.build_plot_indices(
            prior_len=50,
            new_len=50,
            prior_start_index=0,
            new_start_index=None,
            handoff_index=3,
        )

        assert np.array_equal(prior_x[[0, -1]], np.array([0, 49], dtype=np.int32))
        assert np.array_equal(new_x[[0, -1]], np.array([3, 52], dtype=np.int32))
        assert handoff_x == 3


class TestModelLatencyProfilingHelpers:
    def test_profiler_builds_trace_with_derived_denoising_fields(self):
        profiler = ModelLatencyProfiler(enabled=True)
        profiler.add_duration("image_encoders_ms", 18.0)
        profiler.add_duration("llm_prefill_ms", 44.0)
        profiler.add_sample("denoising_step_trace_ms", 10.0)
        profiler.add_sample("denoising_step_trace_ms", 14.0)

        trace = build_model_latency_trace(
            profiler=profiler,
            mode="no_rtc",
            metadata={"num_steps": 2},
        )

        assert trace.mode == "no_rtc"
        assert trace.component_ms["image_encoders_ms"] == 18.0
        assert trace.component_ms["llm_prefill_ms"] == 44.0
        assert trace.component_ms["denoising_total_ms"] == 24.0
        assert trace.component_ms["denoising_step_mean_ms"] == 12.0
        assert trace.sample_ms["denoising_step_trace_ms"] == [10.0, 14.0]
        assert trace.metadata["num_steps"] == 2

    def test_aggregate_latency_traces_computes_mean_and_std(self):
        first = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 18.0, "llm_prefill_ms": 44.0},
                sample_ms={"denoising_step_trace_ms": [10.0, 14.0]},
            ),
            mode="rtc",
            metadata={"num_steps": 2},
        )
        second = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 22.0, "llm_prefill_ms": 40.0},
                sample_ms={"denoising_step_trace_ms": [12.0, 16.0]},
            ),
            mode="rtc",
            metadata={"num_steps": 2},
        )

        summary = aggregate_model_latency_traces([first, second])

        assert summary.mode == "rtc"
        assert summary.runs == 2
        assert summary.mean_ms["image_encoders_ms"] == 20.0
        assert summary.mean_ms["llm_prefill_ms"] == 42.0
        assert summary.mean_ms["denoising_total_ms"] == 26.0
        assert summary.mean_ms["denoising_step_mean_ms"] == 13.0
        assert summary.std_ms["image_encoders_ms"] == 2.0
        assert summary.counts["image_encoders_ms"] == 2

    def test_aggregate_latency_traces_tracks_field_counts_for_partial_rtc_fields(self):
        first = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "rtc_guidance_total_ms": 30.0},
            ),
            mode="rtc",
        )
        second = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 12.0},
            ),
            mode="rtc",
        )

        summary = aggregate_model_latency_traces([first, second])

        assert summary.runs == 2
        assert summary.counts["image_encoders_ms"] == 2
        assert summary.counts["rtc_guidance_total_ms"] == 1

    def test_format_latency_summary_matches_paper_style_components(self):
        trace = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={
                    "image_encoders_ms": 18.0,
                    "llm_prefill_ms": 44.0,
                    "rtc_guidance_total_ms": 21.0,
                },
                sample_ms={"denoising_step_trace_ms": [7.0, 7.0]},
            ),
            mode="rtc",
            metadata={"num_steps": 2},
        )

        rendered = format_model_latency_summary(trace)

        assert "Mode: rtc" in rendered
        assert "image_encoders_ms=18.000" in rendered
        assert "llm_prefill_ms=44.000" in rendered
        assert "denoising_total_ms=14.000" in rendered
        assert "rtc_guidance_total_ms=21.000" in rendered

    def test_complete_latency_trace_requires_rtc_guidance_field_for_rtc_mode(self):
        rtc_without_guidance = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "llm_prefill_ms": 20.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="rtc",
        )
        rtc_with_guidance = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "llm_prefill_ms": 20.0, "rtc_guidance_total_ms": 15.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="rtc",
        )
        no_rtc = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "llm_prefill_ms": 20.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="no_rtc",
        )

        assert is_complete_model_latency_trace(rtc_without_guidance) is False
        assert is_complete_model_latency_trace(rtc_with_guidance) is True
        assert is_complete_model_latency_trace(no_rtc) is True

    def test_latency_collector_skips_incomplete_rtc_trace_without_consuming_warmup(self):
        collector = ModelLatencyCollector(warmup_remaining=1, target_runs=2)
        incomplete = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="rtc",
        )
        complete = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "rtc_guidance_total_ms": 5.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="rtc",
        )

        assert collector.ingest(incomplete) == "incomplete"
        assert collector.warmup_remaining == 1
        assert collector.ingest(complete) == "warmup"
        assert collector.warmup_remaining == 0

    def test_latency_collector_records_first_complete_trace_and_becomes_ready(self):
        collector = ModelLatencyCollector(warmup_remaining=0, target_runs=2)
        first = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 10.0, "rtc_guidance_total_ms": 5.0},
                sample_ms={"denoising_step_trace_ms": [1.0, 2.0]},
            ),
            mode="rtc",
        )
        second = build_model_latency_trace(
            profiler=ModelLatencyProfiler(
                enabled=True,
                component_ms={"image_encoders_ms": 12.0, "rtc_guidance_total_ms": 6.0},
                sample_ms={"denoising_step_trace_ms": [1.5, 2.5]},
            ),
            mode="rtc",
        )

        assert collector.ingest(first) == "recorded_first"
        assert collector.single_trace == first
        assert collector.is_ready() is False

        assert collector.ingest(second) == "recorded"
        assert collector.single_trace == first
        assert collector.is_ready() is True
