"""Plot aggregate Step 8 E5 mask coverage statistics from mask_stats.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([int(row["index"]) for row in rows])
    raw = np.asarray([float(row["raw_valid_ratio_0p5_80m"]) for row in rows])
    backend = np.asarray([float(row["e4_backend_valid_ratio_0p5_40m"]) for row in rows])
    e5 = np.asarray([float(row["e5_valid_ratio"]) for row in rows])
    far = np.asarray([float(row["far_removed_ratio"]) for row in rows])
    sky = np.asarray([float(row["sky_removed_ratio"]) for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(x, raw, label="Raw RAFT valid (0.5-80 m)", linewidth=1.3)
    axes[0].plot(x, backend, label="E4 backend valid (0.5-40 m)", linewidth=1.3)
    axes[0].plot(x, e5, label="E5 sky/far mask valid", linewidth=1.5)
    axes[0].set_ylabel("Valid pixel ratio")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, far, label="Far removed (>40 m)", linewidth=1.3)
    axes[1].plot(x, sky, label="Conservative sky removed", linewidth=1.3)
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Removed pixel ratio")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("KITTI 00 first 200 frames: E5 sky/far mask coverage")
    figure.savefig(args.output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
