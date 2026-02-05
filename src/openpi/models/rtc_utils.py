from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

import jax
import jax.numpy as jnp


class RTCAttentionSchedule(str, Enum):
    """Schedule for RTC prefix attention weights."""
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Configuration for Real-Time Chunking (RTC)."""
    # Whether RTC is enabled
    enabled: bool = False

    # Attention schedule for prefix weights
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR

    # Max guidance weight (if None, use num_steps)
    max_guidance_weight: Optional[float] = None

    # Horizon used to build prefix weights
    execution_horizon: int = 10

    # Prior variance scale parameter (default key for Alex Soare's analysis)
    # Default 1.0 matches original RTC behavior
    sigma_d: float = 1.0

    # Whether to use full trajectory alignment (gradient-free)
    # If True, guidance just nudges trajectory towards previous chunk
    full_trajectory_alignment: bool = True


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR,
) -> jax.Array:
    """
    Calculate prefix weights for RTC guidance.
    
    Args:
        start: Start index of the overlap (inference delay).
        end: End index (execution horizon).
        total: Total length of the sequence (action horizon).
        schedule: Weight schedule type.
        
    Returns:
        Array of shape (total,) containing weights in [0, 1].
    """
    # JAX-compatible implementation using static shapes where possible
    
    start = jnp.minimum(start, end)
    
    def _get_linear_weights():
        # Calculate linear weights: linspace(1, 0)
        # Length of the ramp: total - skip_end - start
        # but we need to match the python implementation logic
        
        # Python: linspace(1, 0, steps+2)[1:-1]
        skip_steps_at_end = jnp.maximum(total - end, 0)
        steps = total - skip_steps_at_end - start
        
        def _compute_lin():
            # Create a ramp from 1 to 0 over 'steps'
            # We want 'steps' elements. 
            # jnp.linspace(1, 0, steps) gives exactly 'steps' elements from 1 to 0.
            # But the logic in LeRobot uses linspace(1, 0, steps+2)[1:-1]
            # which is equivalent to excluding endpoints if they match exactly? 
            # LeRobot: linspace(1, 0, steps+2) -> [1, w1, w2, ..., 0] -> [w1, ... wn]
            # So it generates 'steps' internal points.
            
            # Let's use direct calculation:
            # i ranges from 0 to steps-1
            # weight = 1 - (i + 1) / (steps + 1)
            indices = jnp.arange(steps)
            ramp = 1.0 - (indices + 1.0) / (steps + 1.0)
            return ramp

        lin_weights = jax.lax.cond(
            (end <= start) | (steps <= 0),
            lambda: jnp.zeros((0,)),
            _compute_lin
        )
        
        # Now assemble the full weight vector
        # [ones(start), lin_weights, zeros(skip_end)]
        # Use dynamic update slice or concatenation for JAX
        
        # Initialize with zeros
        weights = jnp.zeros(total)
        
        # Add ones at the beginning
        # We can implement this by a mask
        indices = jnp.arange(total)
        
        # 1. Leading ones: indices < start
        mask_ones = indices < start
        weights = jnp.where(mask_ones, 1.0, weights)
        
        # 2. Ramp: start <= indices < (total - skip_end)
        # Note: total - skip_end is essentially min(total, end) which is 'end' if end < total
        ramp_end = total - skip_steps_at_end
        mask_ramp = (indices >= start) & (indices < ramp_end)
        
        # Calculate ramp values for all indices (to avoid dynamic sizing issues)
        # ramp_val = 1 - (indices - start + 1) / (ramp_end - start + 1)
        denom = ramp_end - start + 1.0
        ramp_vals = 1.0 - (indices - start + 1.0) / (denom + 1e-6) # avoid div 0
        
        weights = jnp.where(mask_ramp, ramp_vals, weights)
        
        # Trailing zeros are already 0
        return weights

    if schedule == RTCAttentionSchedule.ZEROS:
        weights = jnp.zeros(total)
        weights = jnp.where(jnp.arange(total) < start, 1.0, weights)
        return weights
        
    elif schedule == RTCAttentionSchedule.ONES:
        weights = jnp.ones(total)
        weights = jnp.where(jnp.arange(total) >= end, 0.0, weights)
        return weights
        
    elif schedule == RTCAttentionSchedule.LINEAR:
        return _get_linear_weights()
        
    elif schedule == RTCAttentionSchedule.EXP:
        weights = _get_linear_weights()
        # Apply exp transformation: (exp(w) - 1) / (e - 1)
        # Only apply to the ramp part (0 < w < 1)
        # But applying to 0 gives 0, applying to 1 gives 1.
        # So we can apply to all.
        weights = (jnp.exp(weights) - 1.0) / (math.e - 1.0)
        return weights
        
    else:
        # Fallback
        return jnp.zeros(total)
