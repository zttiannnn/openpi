from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np


def build_delay_histogram_summary(
    delay_steps: list[int] | tuple[int, ...],
    *,
    fps: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(delay_steps, dtype=np.int32)
    fps = float(fps)
    step_duration_ms = 1000.0 / fps

    summary: dict[str, Any] = {
        "fps": fps,
        "step_duration_ms": step_duration_ms,
        "num_samples": int(values.size),
        "samples": values.tolist(),
        "histogram": [],
        "summary": {
            "min_delay_steps": None,
            "max_delay_steps": None,
            "mean_delay_steps": None,
            "p50_delay_steps": None,
            "p90_delay_steps": None,
            "p95_delay_steps": None,
        },
    }
    if metadata:
        summary.update(metadata)

    if values.size == 0:
        return summary

    unique_values, counts = np.unique(values, return_counts=True)
    histogram = []
    for delay, count in zip(unique_values.tolist(), counts.tolist(), strict=True):
        histogram.append(
            {
                "delay_steps": int(delay),
                "delay_ms": float(delay) * step_duration_ms,
                "count": int(count),
                "fraction": float(count) / float(values.size),
            }
        )

    summary["histogram"] = histogram
    summary["summary"] = {
        "min_delay_steps": int(np.min(values)),
        "max_delay_steps": int(np.max(values)),
        "mean_delay_steps": float(np.mean(values)),
        "p50_delay_steps": float(np.percentile(values, 50)),
        "p90_delay_steps": float(np.percentile(values, 90)),
        "p95_delay_steps": float(np.percentile(values, 95)),
    }
    return summary


def write_delay_histogram_json(
    path: str | pathlib.Path,
    *,
    summary: dict[str, Any],
    base_dir: str | pathlib.Path | None = None,
) -> None:
    output_path = pathlib.Path(path)
    if not output_path.is_absolute() and base_dir is not None:
        output_path = pathlib.Path(base_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
