from collections.abc import Mapping
from typing import Any


def build_policy_infer_kwargs(
    *,
    num_steps: int,
    rtc_context: Mapping[str, Any] | None = None,
    profile_model: bool = False,
) -> dict[str, Any]:
    """Build kwargs forwarded from runtime scripts to policy.infer()."""

    infer_kwargs: dict[str, Any] = {"num_steps": int(num_steps)}
    if profile_model:
        infer_kwargs["profile_model"] = True
    if rtc_context:
        infer_kwargs.update(dict(rtc_context))
    return infer_kwargs
