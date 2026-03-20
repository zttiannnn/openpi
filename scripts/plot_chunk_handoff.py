#!/usr/bin/env python3
"""Plot prior/new chunk handoff snapshots dumped from inference_smooth_0912.py."""

from __future__ import annotations

import argparse
import os
import pathlib

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY", ""):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _normalize_chunk(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 1:
        return data[None, :]
    if data.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D chunk array, got shape {data.shape}")
    return data


def _load_inputs(input_path: pathlib.Path) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.npz") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_plot_indices(
    *,
    prior_len: int,
    new_len: int,
    prior_start_index: int = 0,
    new_start_index: int | None,
    handoff_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    prior_x = np.arange(int(prior_len), dtype=np.int32) + int(prior_start_index)
    if new_start_index is None:
        effective_new_start = int(handoff_index)
        handoff_x = int(handoff_index)
    else:
        effective_new_start = int(new_start_index)
        handoff_x = effective_new_start + int(handoff_index)
    new_x = np.arange(int(new_len), dtype=np.int32) + effective_new_start
    return prior_x, new_x, handoff_x


def _plot_space(
    *,
    npz_path: pathlib.Path,
    output_dir: pathlib.Path,
    space: str,
    prior: np.ndarray,
    new: np.ndarray,
    prior_start_index: int,
    new_start_index: int | None,
    handoff_index: int,
    metadata: dict[str, int | str],
    joint_dims: int,
) -> pathlib.Path:
    prior = _normalize_chunk(prior)
    new = _normalize_chunk(new)
    dims = min(joint_dims, prior.shape[1], new.shape[1])
    if dims <= 0:
        raise ValueError(f"No joint dimensions available for plotting in {npz_path}")

    ncols = 2
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False, sharex=False)
    axes = axes.ravel()

    prior_x, new_x, handoff_x = build_plot_indices(
        prior_len=prior.shape[0],
        new_len=new.shape[0],
        prior_start_index=prior_start_index,
        new_start_index=new_start_index,
        handoff_index=handoff_index,
    )

    for dim in range(dims):
        ax = axes[dim]
        ax.plot(prior_x, prior[:, dim], color="tab:blue", linewidth=1.5, label="Prior")
        ax.plot(new_x, new[:, dim], color="tab:red", linewidth=1.5, label="New")
        ax.axvline(int(handoff_x), color="0.7", linestyle="--", linewidth=1.2)
        ax.set_title(f"{space.capitalize()} joint {dim}")
        ax.set_xlabel("Step #")
        ax.set_ylabel("Action value")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right")

    for idx in range(dims, len(axes)):
        fig.delaxes(axes[idx])

    step_idx = metadata.get("step_idx", "unknown")
    mode = metadata.get("mode", "unknown")
    real_delay = metadata.get("real_delay", "n/a")
    guided_window_start = metadata.get("guided_window_start", "n/a")
    new_start = metadata.get("new_start_index", "legacy")
    handoff_abs = metadata.get("handoff_abs_index", "legacy")
    fig.suptitle(
        f"{space.capitalize()} chunk handoff | step={step_idx} mode={mode} "
        f"real_delay={real_delay} guided_window_start={guided_window_start} "
        f"new_start={new_start} handoff_abs={handoff_abs}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{npz_path.stem}_{space}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prior/new chunk handoff snapshots from .npz dumps")
    parser.add_argument("--input", required=True, help="Path to one .npz snapshot or a directory of snapshots")
    parser.add_argument("--output_dir", required=True, help="Directory for generated figures")
    parser.add_argument("--space", choices=["raw", "executed", "both"], default="both", help="Which handoff space to plot")
    parser.add_argument("--joint_dims", type=int, default=6, help="Number of arm-joint dimensions to plot")
    args = parser.parse_args()

    input_paths = _load_inputs(pathlib.Path(args.input))
    output_dir = pathlib.Path(args.output_dir)
    requested_spaces = ["raw", "executed"] if args.space == "both" else [args.space]

    for npz_path in input_paths:
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = {
                "step_idx": int(data["step_idx"]),
                "mode": str(data["mode"]),
                "real_delay": int(data["real_delay"]),
                "guided_window_start": int(data["guided_window_start"]),
                "prior_start_index": int(data["prior_start_index"]) if "prior_start_index" in data else 0,
                "new_start_index": int(data["new_start_index"]) if "new_start_index" in data else None,
                "handoff_index": int(data["handoff_index"]),
            }
            handoff_index = int(metadata["handoff_index"])
            prior_start_index = int(metadata["prior_start_index"])
            new_start_index = metadata["new_start_index"]
            if new_start_index is None:
                handoff_abs_index = handoff_index
            else:
                handoff_abs_index = int(new_start_index) + handoff_index
            metadata["handoff_abs_index"] = handoff_abs_index
            for space in requested_spaces:
                prior_key = f"{space}_prior"
                new_key = f"{space}_new"
                if prior_key not in data or new_key not in data:
                    raise KeyError(f"Missing keys {prior_key}/{new_key} in {npz_path}")
                out_path = _plot_space(
                    npz_path=npz_path,
                    output_dir=output_dir,
                    space=space,
                    prior=data[prior_key],
                    new=data[new_key],
                    prior_start_index=prior_start_index,
                    new_start_index=new_start_index,
                    handoff_index=handoff_index,
                    metadata=metadata,
                    joint_dims=args.joint_dims,
                )
                print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
