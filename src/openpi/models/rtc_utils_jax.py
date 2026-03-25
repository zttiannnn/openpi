import math

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class PaperRTCConfig:
    """Minimal JAX RTC config for the paper-style raw action-space algorithm."""

    enabled: bool = True
    beta: float = 5.0


def _ensure_batched_actions(actions: jax.Array) -> tuple[jax.Array, bool]:
    squeezed = actions.ndim == 2
    if squeezed:
        actions = actions[None, ...]
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with rank 2 or 3, got shape {actions.shape}")
    return actions, squeezed


def _pad_prev_actions(
    prev_actions: jax.Array,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    dtype: jnp.dtype,
) -> jax.Array:
    prev_actions, _ = _ensure_batched_actions(jnp.asarray(prev_actions, dtype=dtype))
    if prev_actions.shape[0] == 1 and batch_size != 1:
        prev_actions = jnp.broadcast_to(prev_actions, (batch_size, *prev_actions.shape[1:]))
    elif prev_actions.shape[0] != batch_size:
        raise ValueError(f"Cannot broadcast prev_actions batch {prev_actions.shape[0]} to {batch_size}")

    padded = jnp.zeros((batch_size, action_horizon, action_dim), dtype=dtype)
    copy_horizon = min(prev_actions.shape[1], action_horizon)
    copy_dim = min(prev_actions.shape[2], action_dim)
    return padded.at[:, :copy_horizon, :copy_dim].set(prev_actions[:, :copy_horizon, :copy_dim])


def get_paper_prefix_weights(
    *,
    inference_delay: int | jax.Array,
    execution_horizon: int | jax.Array,
    total: int,
) -> jax.Array:
    """Build the soft mask from RTC Eq. 5."""

    total = max(int(total), 0)
    if total == 0:
        return jnp.zeros((0,), dtype=jnp.float32)

    start = jnp.clip(jnp.asarray(inference_delay, dtype=jnp.int32), 0, total)
    end = jnp.clip(total - jnp.asarray(execution_horizon, dtype=jnp.int32), 0, total)
    start = jnp.minimum(start, end)

    indices = jnp.arange(total, dtype=jnp.float32)
    denom = jnp.maximum((end - start + 1).astype(jnp.float32), 1.0)
    c_i = (end.astype(jnp.float32) - indices) / denom
    decay = c_i * jnp.expm1(c_i) / (math.e - 1.0)

    return jnp.where(
        indices < start.astype(jnp.float32),
        1.0,
        jnp.where(indices < end.astype(jnp.float32), decay, 0.0),
    )


def compute_paper_guidance_weight(time: int | float | jax.Array, *, beta: float) -> jax.Array:
    """Compute the clipped RTC guidance weight from RTC Eq. 2/4."""

    time = jnp.asarray(time, dtype=jnp.float32)
    tau = 1.0 - time
    numerator = time**2 + tau**2
    denominator = jnp.maximum(time * tau, 1e-6)
    unclipped = numerator / denominator
    clipped = jnp.minimum(jnp.asarray(beta, dtype=jnp.float32), unclipped)
    return jnp.where(time <= 0.0, 0.0, clipped)


def apply_rtc_guidance(
    *,
    x_t: jax.Array,
    prev_chunk_left_over: jax.Array,
    inference_delay: int | jax.Array,
    execution_horizon: int | jax.Array,
    time: int | float | jax.Array,
    denoise_step_partial,
    rtc_config: PaperRTCConfig,
) -> jax.Array:
    """Apply paper-style raw RTC guidance to one JAX denoising step."""

    x_t, squeezed = _ensure_batched_actions(jnp.asarray(x_t))
    dtype = x_t.dtype
    batch_size, action_horizon, action_dim = x_t.shape

    def unguided(_: None) -> jax.Array:
        return denoise_step_partial(x_t)

    def guided(_: None) -> jax.Array:
        prev_actions = _pad_prev_actions(
            prev_chunk_left_over,
            batch_size=batch_size,
            action_horizon=action_horizon,
            action_dim=action_dim,
            dtype=dtype,
        )
        weights = get_paper_prefix_weights(
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            total=action_horizon,
        ).astype(dtype)[None, :, None]
        step_time = jnp.asarray(time, dtype=dtype)

        def estimate_and_velocity(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            v_t = denoise_step_partial(x)
            x1_t = x - step_time * v_t
            return x1_t, v_t

        (x1_t, v_t), pullback = jax.vjp(estimate_and_velocity, x_t)
        err = (prev_actions - x1_t) * weights
        correction, = pullback((err, jnp.zeros_like(v_t)))

        guidance_weight = compute_paper_guidance_weight(step_time, beta=rtc_config.beta).astype(dtype)
        return v_t - guidance_weight * correction

    enabled = jnp.asarray(rtc_config.enabled, dtype=jnp.bool_)
    guided_v_t = jax.lax.cond(enabled, guided, unguided, operand=None)
    return guided_v_t[0] if squeezed else guided_v_t
