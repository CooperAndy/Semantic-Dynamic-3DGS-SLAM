"""Render the S1 confidence-gate retention evidence as standalone figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [float(row["threshold"]) for row in rows] != [0.5, 0.6, 0.7, 0.8, 0.9]:
        raise ValueError("expected sorted S1 thresholds 0.50, 0.60, 0.70, 0.80, 0.90")

    threshold = [float(row["threshold"]) for row in rows]
    eligible_total = [float(row["eligible_ratio_total"]) for row in rows]
    eligible_valid = [float(row["eligible_ratio_valid_depth"]) for row in rows]
    valid_depth = float(rows[0]["valid_depth_ratio_total"])
    positive = float(rows[0]["positive_confidence_ratio_total"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
        }
    )
    blue = "#285F8F"
    gold = "#B27818"
    grey = "#A7AFB8"
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(threshold, eligible_total, color=blue, marker="o", linewidth=2.2, label="Eligible / all pixels")
    ax.plot(threshold, eligible_valid, color=gold, marker="s", linestyle="--", linewidth=2.0, label="Eligible / valid depth")
    for x, y in zip(threshold, eligible_total, strict=True):
        ax.annotate(f"{y:.1%}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", color=blue)
    ax.set_title("A. Confidence threshold vs seed retention")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Retention ratio")
    ax.set_xticks(threshold)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    labels = ["All pixels", "Valid depth\n0.5-40 m", "C > 0", *[f"C >= {value:.2f}" for value in threshold]]
    values = [1.0, valid_depth, positive, *eligible_total]
    colors = [grey, grey, grey, "#16A34A", blue, blue, gold, blue]
    bars = ax.barh(labels, values, color=colors, edgecolor="#374151", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of all image pixels")
    ax.set_title("B. Nested filtering evidence")
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        ax.text(min(value + 0.018, 1.01), bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)

    fig.suptitle("S1 - KITTI seq00, frames 000000-000199 (T=0.50 later extension)", fontsize=14, color="#1F2937")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
