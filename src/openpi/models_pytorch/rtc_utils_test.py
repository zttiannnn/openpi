"""Tests for RTC utils (PyTorch version)."""

import math

import torch

from openpi.models_pytorch.rtc_utils import RTCConfig, get_prefix_weights


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
