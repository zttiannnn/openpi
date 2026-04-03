import numpy as np
import jax
import jax.numpy as jnp
import pytest

from openpi.models import rtc_utils_jax


def test_get_paper_prefix_weights_matches_expected_exp_decay():
    weights = rtc_utils_jax.get_paper_prefix_weights(inference_delay=5, execution_horizon=11, total=25)
    expected = jnp.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.7645, 0.5706, 0.4130, 0.2871, 0.1888, 0.1145, 0.0611, 0.0258, 0.0061],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(weights[:14]), np.asarray(expected), atol=1e-4, rtol=0.0)
    np.testing.assert_allclose(np.asarray(weights[14:]), 0.0, atol=1e-6, rtol=0.0)


def test_compute_paper_guidance_weight_matches_boundary_behavior():
    assert np.isclose(float(rtc_utils_jax.compute_paper_guidance_weight(1.0, beta=5.0)), 5.0)
    assert np.isclose(float(rtc_utils_jax.compute_paper_guidance_weight(0.0, beta=5.0)), 0.0)


def test_apply_rtc_guidance_only_changes_overlap_region():
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=5.0)
    x_t = jnp.ones((1, 10, 1), dtype=jnp.float32)
    prev_chunk = jnp.full((1, 10, 1), 0.1, dtype=jnp.float32)

    def denoise_step_partial(x):
        return x * 0.5

    baseline = denoise_step_partial(x_t)
    guided = rtc_utils_jax.apply_rtc_guidance(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=2,
        execution_horizon=6,
        time=jnp.asarray(0.5, dtype=jnp.float32),
        denoise_step_partial=denoise_step_partial,
        rtc_config=rtc_config,
    )

    np.testing.assert_allclose(np.asarray(guided[:, 4:]), np.asarray(baseline[:, 4:]), atol=1e-6, rtol=0.0)
    assert np.max(np.abs(np.asarray(guided[:, :4] - baseline[:, :4]))) > 1e-4


def test_decode_actions_to_executed_prefix_z_score_unnormalizes_and_slices_arm_dims():
    spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=jnp.float32),
        action_std=jnp.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=jnp.float32),
        arm_joint_dims=6,
    )
    raw_actions = jnp.array(
        [
            [[0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 9.0, 10.0], [1.0, 0.0, 0.5, -1.0, -2.0, 2.0, 11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )

    decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        raw_actions,
        observation_state=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        transform_spec=spec,
    )

    expected = jnp.array(
        [[[10.0, 23.0, 26.0, 42.5, 62.0, 46.0], [12.0, 20.0, 32.0, 35.0, 38.0, 74.0]]],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(expected), atol=1e-6, rtol=0.0)


def test_decode_actions_to_executed_prefix_quantile_unnormalizes():
    spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(4, dtype=jnp.float32),
        action_std=jnp.ones(4, dtype=jnp.float32),
        action_q01=jnp.array([-2.0, -1.0, 0.0, 1.0], dtype=jnp.float32),
        action_q99=jnp.array([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
        use_quantiles=True,
        arm_joint_dims=4,
    )
    raw_actions = jnp.array([[[-1.0, 0.0, 1.0, 0.5]]], dtype=jnp.float32)

    decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        raw_actions,
        observation_state=jnp.array([0.0], dtype=jnp.float32),
        transform_spec=spec,
    )

    expected = jnp.array([[[-2.0, 1.0, 4.0, 4.0]]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(expected), atol=1e-6, rtol=0.0)


def test_decode_actions_to_executed_prefix_adds_delta_state_before_slice():
    spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(8, dtype=jnp.float32),
        action_std=jnp.ones(8, dtype=jnp.float32),
        delta_action_mask=(True, False, True, False, False, False, False, False),
        arm_joint_dims=6,
    )
    raw_actions = jnp.zeros((2, 1, 8), dtype=jnp.float32)
    observation_state = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        raw_actions,
        observation_state=observation_state,
        transform_spec=spec,
    )

    expected = jnp.array([[[1.0, 0.0, 3.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 3.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(expected), atol=1e-6, rtol=0.0)


def test_decode_actions_to_executed_prefix_applies_joint_flip_mask_and_truncates():
    spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(8, dtype=jnp.float32),
        action_std=jnp.ones(8, dtype=jnp.float32),
        output_joint_flip_mask=(1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 7.0, 8.0),
        arm_joint_dims=6,
    )
    raw_actions = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=jnp.float32)

    decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        raw_actions,
        observation_state=jnp.array([0.0], dtype=jnp.float32),
        transform_spec=spec,
    )

    expected = jnp.array([[[1.0, -2.0, 1.5, -2.0, 10.0, -12.0]]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(expected), atol=2e-5, rtol=0.0)


def test_decode_actions_to_executed_prefix_composed_order_path():
    spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32),
        action_std=jnp.array([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
        delta_action_mask=(True, False, True, False),
        output_joint_flip_mask=(-1.0, 1.0, -0.5, 2.0),
        arm_joint_dims=3,
    )
    raw_actions = jnp.array([[[1.0, -1.0, 0.5, 2.0]]], dtype=jnp.float32)
    observation_state = jnp.array([3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)

    decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        raw_actions,
        observation_state=observation_state,
        transform_spec=spec,
    )

    expected = jnp.array([[[-15.0, 17.0, -18.5]]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(decoded), np.asarray(expected), atol=2e-5, rtol=0.0)


def test_get_executed_prefix_loss_scale_jax_matches_expected_clamp_behavior():
    zscore_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(4, dtype=jnp.float32),
        action_std=jnp.array([0.5, 1.0, 2.0, 3.0], dtype=jnp.float32),
        arm_joint_dims=4,
    )
    quantile_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(4, dtype=jnp.float32),
        action_std=jnp.ones(4, dtype=jnp.float32),
        action_q01=jnp.array([-4.0, -1.0, 1.0, 2.0], dtype=jnp.float32),
        action_q99=jnp.array([0.0, 3.0, 7.0, 4.0], dtype=jnp.float32),
        use_quantiles=True,
        arm_joint_dims=4,
    )

    zscore_scale = rtc_utils_jax.get_executed_prefix_loss_scale_jax(zscore_spec)
    quantile_scale = rtc_utils_jax.get_executed_prefix_loss_scale_jax(quantile_spec)

    np.testing.assert_allclose(np.asarray(zscore_scale), np.asarray([1.0, 1.0, 2.0, 3.0]), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(np.asarray(quantile_scale), np.asarray([2.0, 2.0, 3.0, 1.0]), atol=1e-6, rtol=0.0)


def test_apply_rtc_guidance_executed_overlap_reduces_decoded_overlap_error_when_execution_horizon_exceeds_overlap():
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=0.5, mode="executed_overlap_paper")
    transform_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(2, dtype=jnp.float32),
        action_std=jnp.ones(2, dtype=jnp.float32),
        arm_joint_dims=2,
    )
    x_t = jnp.array([[[2.0, -2.0], [2.0, -2.0], [2.0, -2.0]]], dtype=jnp.float32)
    processed_target = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    observation_state = jnp.zeros((1, 2), dtype=jnp.float32)
    time = jnp.asarray(0.5, dtype=jnp.float32)

    def denoise_step_partial(x):
        return 3.0 * x

    baseline_v_t = denoise_step_partial(x_t)
    baseline_decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        x_t - time * baseline_v_t,
        observation_state=observation_state,
        transform_spec=transform_spec,
    )

    guided_v_t = rtc_utils_jax.apply_rtc_guidance(
        x_t=x_t,
        prev_chunk_left_over=jnp.zeros_like(x_t),
        inference_delay=1,
        execution_horizon=6,
        time=time,
        denoise_step_partial=denoise_step_partial,
        rtc_config=rtc_config,
        processed_leftover=processed_target,
        observation_state=observation_state,
        executed_transform_spec=transform_spec,
    )
    guided_decoded = rtc_utils_jax.decode_actions_to_executed_prefix_jax(
        x_t - time * guided_v_t,
        observation_state=observation_state,
        transform_spec=transform_spec,
    )

    baseline_err = jnp.mean(jnp.square(baseline_decoded[:, :2, :] - processed_target[:, :2, :]))
    guided_err = jnp.mean(jnp.square(guided_decoded[:, :2, :] - processed_target[:, :2, :]))

    assert float(baseline_err) > 0.0
    assert float(guided_err) < float(baseline_err)


def test_apply_rtc_guidance_executed_overlap_respects_processed_leftover_len_for_padded_targets():
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=0.5, mode="executed_overlap_paper")
    transform_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(2, dtype=jnp.float32),
        action_std=jnp.ones(2, dtype=jnp.float32),
        arm_joint_dims=2,
    )
    x_t = jnp.array([[[1.5, -1.5], [1.0, -1.0], [0.5, -0.5], [0.25, -0.25]]], dtype=jnp.float32)
    short_target = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    padded_target = jnp.array(
        [[[0.0, 0.0], [0.0, 0.0], [25.0, -25.0], [25.0, -25.0]]],
        dtype=jnp.float32,
    )
    observation_state = jnp.zeros((1, 2), dtype=jnp.float32)
    time = jnp.asarray(0.5, dtype=jnp.float32)

    def denoise_step_partial(x):
        return 0.75 * x

    guided_short = rtc_utils_jax.apply_rtc_guidance(
        x_t=x_t,
        prev_chunk_left_over=jnp.zeros_like(x_t),
        inference_delay=0,
        execution_horizon=1,
        time=time,
        denoise_step_partial=denoise_step_partial,
        rtc_config=rtc_config,
        processed_leftover=short_target,
        processed_leftover_len=2,
        observation_state=observation_state,
        executed_transform_spec=transform_spec,
    )
    guided_padded = rtc_utils_jax.apply_rtc_guidance(
        x_t=x_t,
        prev_chunk_left_over=jnp.zeros_like(x_t),
        inference_delay=0,
        execution_horizon=1,
        time=time,
        denoise_step_partial=denoise_step_partial,
        rtc_config=rtc_config,
        processed_leftover=padded_target,
        processed_leftover_len=2,
        observation_state=observation_state,
        executed_transform_spec=transform_spec,
    )

    np.testing.assert_allclose(np.asarray(guided_padded), np.asarray(guided_short), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "missing_field",
    ("processed_leftover", "observation_state", "executed_transform_spec"),
)
def test_apply_rtc_guidance_executed_overlap_missing_inputs_fallback_under_jit(missing_field):
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=0.5, mode="executed_overlap_paper")
    transform_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros(2, dtype=jnp.float32),
        action_std=jnp.ones(2, dtype=jnp.float32),
        arm_joint_dims=2,
    )
    x_t = jnp.array([[[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]]], dtype=jnp.float32)
    processed_leftover = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    observation_state = jnp.zeros((1, 2), dtype=jnp.float32)

    def denoise_step_partial(x):
        return 0.25 * x + 0.1

    expected = denoise_step_partial(x_t)

    if missing_field == "processed_leftover":
        processed_leftover = None
    elif missing_field == "observation_state":
        observation_state = None
    else:
        transform_spec = None

    @jax.jit
    def apply_guidance():
        return rtc_utils_jax.apply_rtc_guidance(
            x_t=x_t,
            prev_chunk_left_over=jnp.zeros_like(x_t),
            inference_delay=0,
            execution_horizon=1,
            time=jnp.asarray(0.5, dtype=jnp.float32),
            denoise_step_partial=denoise_step_partial,
            rtc_config=rtc_config,
            processed_leftover=processed_leftover,
            observation_state=observation_state,
            executed_transform_spec=transform_spec,
        )

    actual = apply_guidance()

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=0.0)
