"""Compare SGBM and RAFT-Stereo depth outputs for the same KITTI frames.

This is a consistency/coverage comparison, not an accuracy evaluation: KITTI
odometry sequence 00 does not provide dense depth ground truth for every pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--sgbm-dir", type=Path, required=True)
    parser.add_argument("--raft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument(
        "--preview-indices",
        type=int,
        nargs="*",
        default=[0, 25, 50, 75, 100, 125, 150, 175, 199],
    )
    return parser.parse_args()


def finite_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def save_comparison(
    output_path: Path,
    rgb_path: Path,
    sgbm: np.ndarray,
    raft: np.ndarray,
    frame_id: str,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    sgbm_masked = np.ma.masked_where(sgbm <= 0, sgbm)
    raft_masked = np.ma.masked_where(raft <= 0, raft)
    overlap = (sgbm > 0) & (raft > 0)
    difference = np.ma.masked_where(~overlap, np.abs(raft - sgbm))

    figure, axes = plt.subplots(2, 2, figsize=(13, 6.8), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"KITTI 00 frame {frame_id} - left RGB")
    axes[0, 1].imshow(sgbm_masked, cmap="turbo_r", vmin=0.5, vmax=40.0)
    axes[0, 1].set_title("SGBM depth (m, 0.5-40 for display)")
    axes[1, 0].imshow(raft_masked, cmap="turbo_r", vmin=0.5, vmax=40.0)
    axes[1, 0].set_title("RAFT-Stereo depth (m, 0.5-40 for display)")
    axes[1, 1].imshow(difference, cmap="magma", vmin=0.0, vmax=10.0)
    axes[1, 1].set_title("Absolute difference on mutually valid pixels (m)")
    for axis in axes.flat:
        axis.axis("off")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir()

    preview_indices = set(args.preview_indices)
    rows: list[dict[str, float | int | str]] = []
    expected_shape = (376, 1241)

    for index in range(args.num_frames):
        frame_id = f"{index:06d}"
        sgbm_path = args.sgbm_dir / f"{frame_id}_depth.npy"
        raft_path = args.raft_dir / f"{frame_id}_depth.npy"
        rgb_path = args.sequence_dir / "image_2" / f"{frame_id}.png"
        sgbm = np.load(sgbm_path, allow_pickle=False)
        raft = np.load(raft_path, allow_pickle=False)
        for label, depth in (("SGBM", sgbm), ("RAFT", raft)):
            if depth.shape != expected_shape or depth.dtype != np.float32:
                raise RuntimeError(
                    f"{label} {frame_id}: expected float32 {expected_shape}, "
                    f"got {depth.dtype} {depth.shape}"
                )
            if not np.isfinite(depth).all() or (depth < 0).any():
                raise RuntimeError(f"{label} {frame_id}: invalid numeric values")

        sgbm_valid = (sgbm >= 0.5) & (sgbm <= 80.0)
        raft_valid = (raft >= 0.5) & (raft <= 80.0)
        overlap = sgbm_valid & raft_valid
        union = sgbm_valid | raft_valid
        difference = np.abs(raft[overlap] - sgbm[overlap])
        rows.append(
            {
                "index": index,
                "frame_id": frame_id,
                "sgbm_valid_ratio": float(sgbm_valid.mean()),
                "raft_valid_ratio": float(raft_valid.mean()),
                "valid_ratio_gain": float(raft_valid.mean() - sgbm_valid.mean()),
                "valid_iou": float(overlap.sum() / union.sum()) if union.any() else 0.0,
                "sgbm_depth_median_m": finite_percentile(sgbm[sgbm_valid], 50),
                "raft_depth_median_m": finite_percentile(raft[raft_valid], 50),
                "overlap_abs_diff_median_m": finite_percentile(difference, 50),
                "overlap_abs_diff_mean_m": float(difference.mean()) if difference.size else float("nan"),
                "overlap_abs_diff_p95_m": finite_percentile(difference, 95),
            }
        )
        if index in preview_indices:
            save_comparison(
                preview_dir / f"{frame_id}_sgbm_vs_raft.png",
                rgb_path,
                sgbm,
                raft,
                frame_id,
            )

    csv_path = args.output_dir / "sgbm_vs_raft_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [
        "sgbm_valid_ratio",
        "raft_valid_ratio",
        "valid_ratio_gain",
        "valid_iou",
        "sgbm_depth_median_m",
        "raft_depth_median_m",
        "overlap_abs_diff_median_m",
        "overlap_abs_diff_mean_m",
        "overlap_abs_diff_p95_m",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }

    summary = {
        "scope": "KITTI odometry sequence 00, frames 000000-000199",
        "interpretation": (
            "Coverage and mutual-consistency comparison only; neither method is treated "
            "as dense ground truth."
        ),
        "depth_range_m": [0.5, 80.0],
        "frames": args.num_frames,
        "aggregate": aggregate,
        "preview_indices": sorted(preview_indices),
    }
    with (args.output_dir / "sgbm_vs_raft_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    x = np.arange(args.num_frames)
    sgbm_ratio = np.asarray([row["sgbm_valid_ratio"] for row in rows], dtype=float)
    raft_ratio = np.asarray([row["raft_valid_ratio"] for row in rows], dtype=float)
    median_difference = np.asarray(
        [row["overlap_abs_diff_median_m"] for row in rows], dtype=float
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(x, sgbm_ratio, label="SGBM", linewidth=1.4)
    axes[0].plot(x, raft_ratio, label="RAFT-Stereo", linewidth=1.4)
    axes[0].set_ylabel("Valid depth ratio")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, median_difference, color="tab:red", linewidth=1.4)
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Median |RAFT-SGBM| (m)\non overlap")
    axes[1].grid(alpha=0.25)
    figure.suptitle("KITTI 00 first 200 frames: SGBM vs RAFT-Stereo")
    figure.savefig(args.output_dir / "aggregate_comparison.png", dpi=150)
    plt.close(figure)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
