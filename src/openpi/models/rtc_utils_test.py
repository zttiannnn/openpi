import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rtc_utils import RTCAttentionSchedule, get_prefix_weights


class TestRTCUtils:
    def test_get_prefix_weights_linear(self):
        # Total 10, start 2, end 8
        # Ones: [0, 1] (2 elements)
        # Ramp: [2...7] (6 elements) -> linspace(1, 0, 6+2)[1:-1]
        # Zeros: [8, 9] (2 elements)
        weights = get_prefix_weights(2, 8, 10, RTCAttentionSchedule.LINEAR)
        
        assert weights.shape == (10,)
        
        # Check ones
        np.testing.assert_allclose(weights[:2], 1.0)
        
        # Check zeros
        np.testing.assert_allclose(weights[8:], 0.0)
        
        # Check ramp
        # 6 steps excluding endpoints 0 and 1 -> 1/7 decrement per step?
        # LeRobot implementation logic: 
        # linspace(1, 0, steps+2)[1:-1]
        # steps=6. linspace(1, 0, 8) -> [1, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7, 0]
        # [1:-1] -> [6/7, 5/7, 4/7, 3/7, 2/7, 1/7]
        expected_ramp = np.linspace(1, 0, 8)[1:-1]
        np.testing.assert_allclose(weights[2:8], expected_ramp, atol=1e-5)

    def test_get_prefix_weights_zeros(self):
        weights = get_prefix_weights(3, 8, 10, RTCAttentionSchedule.ZEROS)
        np.testing.assert_allclose(weights[:3], 1.0)
        np.testing.assert_allclose(weights[3:], 0.0)

    def test_get_prefix_weights_ones(self):
        weights = get_prefix_weights(3, 8, 10, RTCAttentionSchedule.ONES)
        np.testing.assert_allclose(weights[:8], 1.0)
        np.testing.assert_allclose(weights[8:], 0.0)

    def test_jit_compatibility(self):
        # Ensure the function can be jitted with static args for enum
        # But dynamic args for start/end? 
        # Our implementation uses lax.cond so it should be jittable
        
        @jax.jit
        def _wrapper(start, end):
            return get_prefix_weights(start, end, 10, RTCAttentionSchedule.LINEAR)
            
        weights = _wrapper(2, 8)
        assert weights.shape == (10,)
        np.testing.assert_allclose(weights[:2], 1.0)

    def test_edge_cases(self):
        # Start > End
        weights = get_prefix_weights(5, 3, 10, RTCAttentionSchedule.LINEAR)
        # Should behave as if start=3 (min(start, end)) -> leads to ramp from 3 to 3? 
        # Python impl: start = min(start, end) -> start=3. 
        # Ramp length: total-end-start = 10-3-3 = 4? No.
        # skip_end = 10-3=7. 
        # steps = 10 - 7 - 3 = 0.
        # So weights should be ones[:3] + zeros[7] ? 
        np.testing.assert_allclose(weights[:3], 1.0)
        np.testing.assert_allclose(weights[3:], 0.0)

        # End > Total
        weights = get_prefix_weights(2, 12, 10, RTCAttentionSchedule.LINEAR)
        # skip_end = 0.
        # steps = 10 - 0 - 2 = 8.
        np.testing.assert_allclose(weights[:2], 1.0)
        # Ramp covers the rest
