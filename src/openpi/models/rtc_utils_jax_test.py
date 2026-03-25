import numpy as np
import jax.numpy as jnp

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
