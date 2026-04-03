from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import rtc_utils_jax
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.shared.normalize import NormStats


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_model_rtc():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, _act = config.fake_obs(batch_size), config.fake_act(batch_size)
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, beta=5.0)
    prev_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=jnp.float32)

    actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_config,
        prev_actions=prev_actions,
        inference_delay=2,
        execution_horizon=25,
    )
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_model_rtc_executed_overlap_mode():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, _act = config.fake_obs(batch_size), config.fake_act(batch_size)
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, mode="executed_overlap_paper", beta=5.0)
    noise = jnp.linspace(
        -1.0,
        1.0,
        batch_size * model.action_horizon * model.action_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, model.action_horizon, model.action_dim)
    prev_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=jnp.float32)
    processed_leftover = jnp.full((batch_size, model.action_horizon, 6), 3.0, dtype=jnp.float32)
    executed_transform_spec = rtc_utils_jax.RTCExecutedPrefixTransformSpec(
        action_mean=jnp.zeros((model.action_dim,), dtype=jnp.float32),
        action_std=jnp.ones((model.action_dim,), dtype=jnp.float32),
        arm_joint_dims=6,
    )

    baseline = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=False, mode="executed_overlap_paper", beta=5.0),
        prev_actions=prev_actions,
        inference_delay=2,
        execution_horizon=25,
        noise=noise,
    )
    raw_actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=True, mode="raw_paper", beta=5.0),
        prev_actions=prev_actions,
        inference_delay=2,
        execution_horizon=25,
        noise=noise,
    )
    executed_actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_config,
        inference_delay=2,
        execution_horizon=25,
        processed_leftover=processed_leftover,
        observation_state=obs.state,
        executed_transform_spec=executed_transform_spec,
        noise=noise,
    )
    assert executed_actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert not jnp.allclose(executed_actions, baseline)
    assert not jnp.allclose(executed_actions, raw_actions)


def test_pi0_model_rtc_executed_overlap_missing_inputs_matches_unguided_baseline():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, _act = config.fake_obs(batch_size), config.fake_act(batch_size)
    noise = jnp.linspace(
        -0.5,
        0.5,
        batch_size * model.action_horizon * model.action_dim,
        dtype=jnp.float32,
    ).reshape(batch_size, model.action_horizon, model.action_dim)
    rtc_config = rtc_utils_jax.PaperRTCConfig(enabled=True, mode="executed_overlap_paper", beta=5.0)
    prev_actions = jnp.full((batch_size, model.action_horizon, model.action_dim), 2.0, dtype=jnp.float32)

    baseline = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=False, mode="executed_overlap_paper", beta=5.0),
        prev_actions=prev_actions,
        inference_delay=2,
        execution_horizon=25,
        noise=noise,
    )
    actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=5,
        rtc_config=rtc_config,
        prev_actions=prev_actions,
        inference_delay=2,
        execution_horizon=25,
        noise=noise,
    )

    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert jnp.allclose(actions, baseline)


def test_pi0_model_module_jit_accepts_static_executed_transform_spec():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    rtc_transform_spec = _policy_config._build_rtc_executed_prefix_transform_spec_jax(
        norm_stats={
            "actions": NormStats(
                mean=jnp.zeros((model.action_dim,), dtype=jnp.float32),
                std=jnp.ones((model.action_dim,), dtype=jnp.float32),
                q01=jnp.full((model.action_dim,), -1.0, dtype=jnp.float32),
                q99=jnp.full((model.action_dim,), 1.0, dtype=jnp.float32),
            )
        },
        use_quantiles=False,
        output_transforms=[],
    )
    assert rtc_transform_spec is not None
    setattr(model, "rtc_executed_prefix_transform_spec", rtc_transform_spec)

    batch_size = 2
    obs, _act = config.fake_obs(batch_size), config.fake_act(batch_size)
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=5)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_paper_rtc_config_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown RTC mode"):
        rtc_utils_jax.resolve_rtc_mode(rtc_utils_jax.PaperRTCConfig(enabled=True, mode="typo_mode"))


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
