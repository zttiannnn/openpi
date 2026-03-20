from openpi.shared.inference_kwargs import build_policy_infer_kwargs


def test_build_policy_infer_kwargs_includes_num_steps():
    infer_kwargs = build_policy_infer_kwargs(num_steps=8, rtc_context=None)

    assert infer_kwargs == {"num_steps": 8}


def test_build_policy_infer_kwargs_merges_rtc_context():
    rtc_context = {
        "rtc_config": object(),
        "inference_delay": 3,
    }

    infer_kwargs = build_policy_infer_kwargs(num_steps=6, rtc_context=rtc_context)

    assert infer_kwargs["num_steps"] == 6
    assert infer_kwargs["rtc_config"] is rtc_context["rtc_config"]
    assert infer_kwargs["inference_delay"] == 3


def test_build_policy_infer_kwargs_includes_profile_flag():
    infer_kwargs = build_policy_infer_kwargs(
        num_steps=10,
        rtc_context=None,
        profile_model=True,
    )

    assert infer_kwargs == {
        "num_steps": 10,
        "profile_model": True,
    }


def test_build_policy_infer_kwargs_merges_rtc_and_profile():
    rtc_context = {
        "rtc_config": object(),
        "inference_delay": 5,
    }

    infer_kwargs = build_policy_infer_kwargs(
        num_steps=12,
        rtc_context=rtc_context,
        profile_model=True,
    )

    assert infer_kwargs["num_steps"] == 12
    assert infer_kwargs["profile_model"] is True
    assert infer_kwargs["rtc_config"] is rtc_context["rtc_config"]
    assert infer_kwargs["inference_delay"] == 5
