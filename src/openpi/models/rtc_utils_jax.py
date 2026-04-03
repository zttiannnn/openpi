import math

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class RTCExecutedPrefixTransformSpec:
    """JAX-native spec for decoding model outputs into executed arm actions."""

    action_mean: tuple[float, ...] | jax.Array | None
    action_std: tuple[float, ...] | jax.Array | None
    action_q01: tuple[float, ...] | jax.Array | None = None
    action_q99: tuple[float, ...] | jax.Array | None = None
    use_quantiles: bool = struct.field(pytree_node=False, default=False)
    delta_action_mask: tuple[bool, ...] | None = struct.field(pytree_node=False, default=None)
    output_joint_flip_mask: tuple[float, ...] | None = struct.field(pytree_node=False, default=None)
    arm_joint_dims: int = struct.field(pytree_node=False, default=6)


@struct.dataclass
class PaperRTCConfig:
    """Minimal JAX RTC config for paper-style raw or executed-overlap guidance."""

    enabled: bool = True
    beta: float = 5.0
    mode: str = struct.field(pytree_node=False, default="raw_paper")
    guidance_mode: str | None = struct.field(pytree_node=False, default=None)


def resolve_rtc_mode(rtc_config: PaperRTCConfig) -> str:
    mode = rtc_config.guidance_mode if rtc_config.guidance_mode is not None else rtc_config.mode
    if mode in {"raw_paper", "raw_prefix_paper", "raw_prefix"}:
        return "raw_paper"
    if mode == "executed_overlap_paper":
        return mode
    raise ValueError(f"Unknown RTC mode: {mode}")


def _ensure_batched_actions(actions: jax.Array) -> tuple[jax.Array, bool]:
    squeezed = actions.ndim == 2
    if squeezed:
        actions = actions[None, ...]
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with rank 2 or 3, got shape {actions.shape}")
    return actions, squeezed


def _as_float32_array(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float32)


def _ensure_batched_state(state: jax.Array, *, batch_size: int) -> jax.Array:
    if state.ndim == 1:
        state = state[None, ...]
    if state.ndim != 2:
        raise ValueError(f"Expected observation_state with rank 1 or 2, got shape {state.shape}")
    if state.shape[0] == batch_size:
        return state
    if state.shape[0] == 1 and batch_size != 1:
        return jnp.broadcast_to(state, (batch_size, state.shape[1]))
    raise ValueError(f"Cannot broadcast observation_state batch {state.shape[0]} to {batch_size}")


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


def _prepare_processed_leftover(
    processed_leftover: jax.Array,
    *,
    batch_size: int,
    dtype: jnp.dtype,
) -> jax.Array:
    leftover, _ = _ensure_batched_actions(jnp.asarray(processed_leftover, dtype=dtype))
    if leftover.shape[0] == 1 and batch_size != 1:
        return jnp.broadcast_to(leftover, (batch_size, *leftover.shape[1:]))
    if leftover.shape[0] != batch_size:
        raise ValueError(f"Cannot broadcast processed_leftover batch {leftover.shape[0]} to {batch_size}")
    return leftover


def decode_actions_to_executed_prefix_jax(
    raw_actions: jax.Array,
    *,
    observation_state: jax.Array,
    transform_spec: RTCExecutedPrefixTransformSpec,
) -> jax.Array:
    """Decode normalized RTC actions into executed arm-joint space."""

    actions, _ = _ensure_batched_actions(_as_float32_array(raw_actions))
    state = _ensure_batched_state(_as_float32_array(observation_state), batch_size=actions.shape[0])

    decoded = actions
    action_dim = decoded.shape[-1]

    if transform_spec.use_quantiles:
        if transform_spec.action_q01 is None or transform_spec.action_q99 is None:
            raise ValueError("Quantile decoding requires action_q01 and action_q99")
        q01 = _as_float32_array(transform_spec.action_q01).reshape(-1)
        q99 = _as_float32_array(transform_spec.action_q99).reshape(-1)
        stat_dim = q01.shape[0]
        if stat_dim < action_dim:
            first = (decoded[..., :stat_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            decoded = jnp.concatenate([first, decoded[..., stat_dim:]], axis=-1)
        else:
            q01 = q01[:action_dim]
            q99 = q99[:action_dim]
            decoded = (decoded + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        if transform_spec.action_mean is None or transform_spec.action_std is None:
            raise ValueError("Z-score decoding requires action_mean and action_std")
        mean = _as_float32_array(transform_spec.action_mean).reshape(-1)
        std = _as_float32_array(transform_spec.action_std).reshape(-1)
        if mean.shape[0] > action_dim:
            mean = mean[:action_dim]
        elif mean.shape[0] < action_dim:
            mean = jnp.concatenate([mean, jnp.zeros((action_dim - mean.shape[0],), dtype=jnp.float32)], axis=0)
        if std.shape[0] > action_dim:
            std = std[:action_dim]
        elif std.shape[0] < action_dim:
            std = jnp.concatenate([std, jnp.ones((action_dim - std.shape[0],), dtype=jnp.float32)], axis=0)
        decoded = decoded * (std + 1e-6) + mean

    if transform_spec.delta_action_mask is not None:
        mask = jnp.asarray(transform_spec.delta_action_mask, dtype=jnp.bool_).reshape(-1)
        dims = min(mask.shape[0], decoded.shape[-1], state.shape[-1])
        if dims > 0:
            delta_base = jnp.where(mask[:dims], state[:, :dims], jnp.zeros_like(state[:, :dims]))
            decoded = decoded.at[:, :, :dims].add(delta_base[:, None, :])

    if transform_spec.output_joint_flip_mask is not None:
        flip_mask = _as_float32_array(transform_spec.output_joint_flip_mask).reshape(-1)
        dims = min(flip_mask.shape[0], decoded.shape[-1])
        if dims > 0:
            decoded = decoded.at[:, :, :dims].multiply(flip_mask[:dims])

    joint_dims = max(0, min(int(transform_spec.arm_joint_dims), decoded.shape[-1]))
    return decoded[:, :, :joint_dims]


def get_executed_prefix_loss_scale_jax(
    transform_spec: RTCExecutedPrefixTransformSpec,
) -> jax.Array:
    """Return per-joint scales for stable executed-prefix losses."""

    arm_joint_dims = max(int(transform_spec.arm_joint_dims), 0)
    if arm_joint_dims == 0:
        return jnp.ones((0,), dtype=jnp.float32)

    scale = None
    if transform_spec.use_quantiles:
        if transform_spec.action_q01 is not None and transform_spec.action_q99 is not None:
            q01 = _as_float32_array(transform_spec.action_q01).reshape(-1)
            q99 = _as_float32_array(transform_spec.action_q99).reshape(-1)
            scale = jnp.abs(q99 - q01) / 2.0
    else:
        if transform_spec.action_std is not None:
            scale = jnp.abs(_as_float32_array(transform_spec.action_std).reshape(-1))

    if scale is None:
        return jnp.ones((arm_joint_dims,), dtype=jnp.float32)

    if scale.shape[0] < arm_joint_dims:
        scale = jnp.concatenate(
            [scale, jnp.ones((arm_joint_dims - scale.shape[0],), dtype=jnp.float32)],
            axis=0,
        )
    else:
        scale = scale[:arm_joint_dims]

    return jnp.maximum(scale, 1.0)


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


def get_prefix_weights(
    *,
    start: int | jax.Array,
    end: int | jax.Array,
    total: int,
    schedule: str = "linear",
) -> jax.Array:
    """Compute prefix weights matching the PyTorch RTC schedules."""

    total = max(int(total), 0)
    if total == 0:
        return jnp.zeros((0,), dtype=jnp.float32)

    start = jnp.clip(jnp.asarray(start, dtype=jnp.int32), 0, total)
    end = jnp.clip(jnp.asarray(end, dtype=jnp.int32), 0, total)
    start = jnp.minimum(start, end)

    indices = jnp.arange(total, dtype=jnp.float32)
    start_f = start.astype(jnp.float32)
    end_f = end.astype(jnp.float32)

    if schedule == "zeros":
        return jnp.where(indices < start_f, 1.0, 0.0)
    if schedule == "ones":
        return jnp.where(indices < end_f, 1.0, 0.0)

    if schedule not in {"linear", "exp"}:
        raise ValueError(f"Unknown RTC schedule: {schedule!r}")

    denom = jnp.maximum((end - start).astype(jnp.float32) + 1.0, 1.0)
    ramp = (end_f - indices) / denom
    if schedule == "exp":
        ramp = ramp * jnp.expm1(ramp) / (math.e - 1.0)

    return jnp.where(indices < start_f, 1.0, jnp.where(indices < end_f, ramp, 0.0))


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
    prev_chunk_left_over: jax.Array | None,
    inference_delay: int | jax.Array,
    execution_horizon: int | jax.Array,
    time: int | float | jax.Array,
    denoise_step_partial,
    rtc_config: PaperRTCConfig,
    processed_leftover: jax.Array | None = None,
    observation_state: jax.Array | None = None,
    executed_transform_spec: RTCExecutedPrefixTransformSpec | None = None,
) -> jax.Array:
    """Apply one RTC guidance step in raw or executed-overlap mode."""

    x_t, squeezed = _ensure_batched_actions(jnp.asarray(x_t))
    dtype = x_t.dtype
    batch_size, action_horizon, action_dim = x_t.shape

    def unguided(_: None) -> jax.Array:
        return jnp.asarray(denoise_step_partial(x_t), dtype=dtype)

    def raw_guided(_: None) -> jax.Array:
        if prev_chunk_left_over is None:
            return unguided(None)
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
        return jnp.asarray(v_t - guidance_weight * correction, dtype=dtype)

    has_executed_inputs = (
        processed_leftover is not None and observation_state is not None and executed_transform_spec is not None
    )

    def executed_overlap_guided(_: None) -> jax.Array:
        if not has_executed_inputs:
            return unguided(None)

        processed_target = _prepare_processed_leftover(
            processed_leftover,
            batch_size=batch_size,
            dtype=dtype,
        )
        state = _ensure_batched_state(
            _as_float32_array(observation_state),
            batch_size=batch_size,
        )
        step_time = jnp.asarray(time, dtype=dtype)

        def loss_and_velocity(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            v_t = jnp.asarray(denoise_step_partial(x), dtype=dtype)
            x1_t = x - step_time * v_t
            decoded = decode_actions_to_executed_prefix_jax(
                x1_t,
                observation_state=state,
                transform_spec=executed_transform_spec,
            ).astype(dtype)

            available_overlap = min(decoded.shape[1], processed_target.shape[1])
            arm_joint_dims = min(decoded.shape[2], processed_target.shape[2])
            if available_overlap <= 0 or arm_joint_dims <= 0:
                return jnp.zeros((), dtype=dtype), v_t

            weights = get_prefix_weights(
                start=inference_delay,
                end=available_overlap,
                total=available_overlap,
                schedule="exp",
            ).astype(dtype)[None, :, None]
            loss_scale = get_executed_prefix_loss_scale_jax(executed_transform_spec).astype(dtype)[:arm_joint_dims]
            decoded_overlap = decoded[:, :available_overlap, :arm_joint_dims] / loss_scale[None, None, :]
            target_overlap = processed_target[:, :available_overlap, :arm_joint_dims] / loss_scale[None, None, :]
            weighted_error = jnp.square(decoded_overlap - target_overlap) * weights
            denom = jnp.maximum(jnp.sum(weights), 1e-6) * arm_joint_dims * max(batch_size, 1)
            return jnp.sum(weighted_error) / denom, v_t

        (_loss, v_t), correction = jax.value_and_grad(loss_and_velocity, has_aux=True)(x_t)
        guidance_weight = compute_paper_guidance_weight(step_time, beta=rtc_config.beta).astype(dtype)
        return jnp.asarray(v_t - guidance_weight * correction, dtype=dtype)

    mode = resolve_rtc_mode(rtc_config)

    def guided(_: None) -> jax.Array:
        if mode == "executed_overlap_paper":
            return executed_overlap_guided(None)
        return raw_guided(None)

    enabled = jnp.asarray(rtc_config.enabled, dtype=jnp.bool_)
    guided_v_t = jax.lax.cond(enabled, guided, unguided, operand=None)
    return guided_v_t[0] if squeezed else guided_v_t
