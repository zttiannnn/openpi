"""Tests for RTC utils (PyTorch version)."""

import math
import numpy as np

import torch

from openpi.models_pytorch.rtc_utils import RTCActionQueue
from openpi.models_pytorch.rtc_utils import RTCConfig
from openpi.models_pytorch.rtc_utils import apply_rtc_guidance
from openpi.models_pytorch.rtc_utils import compute_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import format_rtc_runtime_stats
from openpi.models_pytorch.rtc_utils import get_rtc_boundary_risks
from openpi.models_pytorch.rtc_utils import get_prefix_weights


class TestRTCConfig:
    def test_defaults(self):
        cfg = RTCConfig()
        assert cfg.enabled is False
        assert cfg.prefix_attention_schedule == "linear"
        assert cfg.max_guidance_weight is None
        assert cfg.execution_horizon == 10
        assert cfg.sigma_d == 1.0

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
