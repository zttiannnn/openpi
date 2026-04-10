import json

import pytest

from openpi.shared import delay_histogram as _delay_histogram


def test_build_delay_histogram_summary_counts_steps_and_ms():
    summary = _delay_histogram.build_delay_histogram_summary(
        [2, 3, 3, 4],
        fps=20,
        metadata={"mode": "async"},
    )

    assert summary["num_samples"] == 4
    assert summary["fps"] == pytest.approx(20.0)
    assert summary["step_duration_ms"] == pytest.approx(50.0)
    assert summary["mode"] == "async"

    assert summary["summary"]["min_delay_steps"] == 2
    assert summary["summary"]["max_delay_steps"] == 4
    assert summary["summary"]["mean_delay_steps"] == pytest.approx(3.0)
    assert summary["summary"]["p50_delay_steps"] == pytest.approx(3.0)

    assert summary["histogram"] == [
        {
            "delay_steps": 2,
            "delay_ms": 100.0,
            "count": 1,
            "fraction": 0.25,
        },
        {
            "delay_steps": 3,
            "delay_ms": 150.0,
            "count": 2,
            "fraction": 0.5,
        },
        {
            "delay_steps": 4,
            "delay_ms": 200.0,
            "count": 1,
            "fraction": 0.25,
        },
    ]


def test_write_delay_histogram_json_creates_parent_directory(tmp_path):
    output_path = tmp_path / "delay_stats" / "async_delay_histogram.json"
    summary = _delay_histogram.build_delay_histogram_summary([0, 1, 1], fps=30)

    _delay_histogram.write_delay_histogram_json(output_path, summary=summary)

    payload = json.loads(output_path.read_text())
    assert payload["num_samples"] == 3
    assert payload["histogram"] == [
        {
            "delay_steps": 0,
            "delay_ms": 0.0,
            "count": 1,
            "fraction": pytest.approx(1 / 3),
        },
        {
            "delay_steps": 1,
            "delay_ms": pytest.approx(1000.0 / 30.0),
            "count": 2,
            "fraction": pytest.approx(2 / 3),
        },
    ]


def test_write_delay_histogram_json_resolves_relative_path_against_base_dir(tmp_path):
    summary = _delay_histogram.build_delay_histogram_summary([2, 2, 3], fps=20)

    _delay_histogram.write_delay_histogram_json(
        "scripts/debug/Async/delay_histogram_async_fps20.json",
        summary=summary,
        base_dir=tmp_path,
    )

    output_path = tmp_path / "scripts/debug/Async/delay_histogram_async_fps20.json"
    payload = json.loads(output_path.read_text())
    assert payload["num_samples"] == 3
