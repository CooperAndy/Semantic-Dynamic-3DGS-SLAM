from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    numeric = {
        "mapping_frames",
        "full_rgb_psnr",
        "full_rgb_ms_ssim",
        "full_rgb_lpips",
        "common_depth_l1_all_valid_m",
        "silhouette_coverage_common_depth",
        "hole_ratio_common_depth",
        "mapping_time_total_estimate_s",
        "mapping_time_s_per_mapped_frame",
        "final_gaussians",
        "peak_cuda_gib",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    rows = load_rows(args.csv)
    heldout = sorted(
        (row for row in rows if row["evaluation_role"] == "heldout"),
        key=lambda row: row["mapping_frames"],
    )
    in_sample = next(row for row in rows if row["evaluation_role"] == "in_sample")

    x = [row["mapping_frames"] for row in heldout]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    panels = [
        ("full_rgb_psnr", "Full-RGB PSNR (dB)", True),
        ("full_rgb_ms_ssim", "Full-RGB MS-SSIM", True),
        ("full_rgb_lpips", "Full-RGB LPIPS (lower is better)", False),
        ("common_depth_l1_all_valid_m", "Common-support depth L1 (m, lower is better)", False),
    ]
    for axis, (key, title, _) in zip(axes.flat, panels):
        axis.plot(x, [row[key] for row in heldout], "o-", color="#1769aa", linewidth=2)
        for row in heldout:
            axis.annotate(f"{row[key]:.3f}", (row["mapping_frames"], row[key]),
                          xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Mapping frames (held-out evaluation fixed at 10 views)")
        axis.set_xticks(x)
        style_axis(axis)
    fig.suptitle("S2: Held-out quality versus mapping-frame budget (T=0.60)", fontsize=14)
    fig.savefig(args.output_dir / "s2_heldout_quality_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    cost_panels = [
        ("mapping_time_total_estimate_s", "Estimated mapping time (s)"),
        ("final_gaussians", "Final Gaussians"),
        ("peak_cuda_gib", "Peak CUDA allocated (GiB)"),
    ]
    for axis, (key, title) in zip(axes, cost_panels):
        axis.plot(x, [row[key] for row in heldout], "o-", color="#d35400", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Mapping frames")
        axis.set_xticks(x)
        style_axis(axis)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    fig.suptitle("S2: Mapping cost versus mapping-frame budget", fontsize=14)
    fig.savefig(args.output_dir / "s2_mapping_cost_curve.png", dpi=180)
    plt.close(fig)

    m190 = next(row for row in heldout if int(row["mapping_frames"]) == 190)
    labels = ["190-frame held-out", "200-frame in-sample"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    compare_panels = [
        ("full_rgb_psnr", "Full-RGB PSNR (dB)", "#2874a6"),
        ("full_rgb_lpips", "Full-RGB LPIPS (lower is better)", "#8e44ad"),
        ("common_depth_l1_all_valid_m", "Common-support depth L1 (m)", "#117864"),
    ]
    for axis, (key, title, color) in zip(axes, compare_panels):
        values = [m190[key], in_sample[key]]
        bars = axis.bar(labels, values, color=[color, "#aab7b8"])
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        style_axis(axis)
    fig.suptitle("Same 10 views, different roles: generalization is not the same as reconstruction", fontsize=13)
    fig.savefig(args.output_dir / "s2_heldout_vs_insample.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
