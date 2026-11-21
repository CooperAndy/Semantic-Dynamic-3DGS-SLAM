"""Analyze E3/E4/E6a under the same pre-registered held-out protocol."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = {
    "E3-SGBM": "e3",
    "E4-RAFT": "e4",
    "E6a-ConfHard": "e6a",
}
METRICS = {
    "psnr": ("PSNR (dB)", "higher"),
    "l1": ("Depth error (m)", "lower"),
    "ssim": ("MS-SSIM", "higher"),
    "lpips": ("LPIPS", "lower"),
}


def parse_runtime(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    pats = {
        "tracking_seconds_per_frame": r"Average Tracking/Frame Time: ([0-9.eE+-]+) s",
        "mapping_milliseconds_per_iteration": r"Average Mapping/Iteration Time: ([0-9.eE+-]+) ms",
        "mapping_seconds_per_frame": r"Average Mapping/Frame Time: ([0-9.eE+-]+) s",
        "total_runtime_seconds": r"11R Total Runtime: ([0-9.eE+-]+) s",
        "peak_cuda_memory_gib": r"11R Peak CUDA Memory Allocated: ([0-9.eE+-]+) GiB",
    }
    out = {}
    for key, pat in pats.items():
        hits = re.findall(pat, text)
        out[key] = float(hits[-1]) if hits else None
    return out


def load_run(exp_dir, log_path):
    exp = Path(exp_dir)
    arrays = {name: np.atleast_1d(np.loadtxt(exp / "eval" / f"{name}.txt")).astype(float)
              for name in ["psnr", "rmse", "l1", "ssim", "lpips"]}
    if {len(x) for x in arrays.values()} != {200}:
        raise ValueError(f"Expected 200 metrics in {exp}: {[len(x) for x in arrays.values()]}")
    with np.load(exp / "params.npz", allow_pickle=True) as p:
        held_out = [int(x) for x in p["eval_only_frames"].tolist()]
        keyframes = [int(x) for x in p["keyframe_time_indices"].tolist()]
        means = p["means3D"]
        gaussian_count = int(means.shape[0])
    if held_out != list(range(19, 200, 20)):
        raise ValueError(f"Unexpected held-out list in {exp}: {held_out}")
    if set(held_out).intersection(keyframes):
        raise ValueError(f"Held-out leakage in {exp}")
    return {
        "arrays": arrays,
        "held_out_frames": held_out,
        "keyframes": keyframes,
        "gaussian_count": gaussian_count,
        "keyframe_count": len(keyframes),
        "params_bytes": (exp / "params.npz").stat().st_size,
        "runtime": parse_runtime(log_path),
    }


def means(run, mask):
    return {name: float(values[mask].mean()) for name, values in run["arrays"].items()}


def main():
    p = argparse.ArgumentParser()
    for key in METHODS.values():
        p.add_argument(f"--{key}-dir", required=True)
        p.add_argument(f"--{key}-log", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for label, key in METHODS.items():
        runs[label] = load_run(getattr(args, f"{key}_dir"), getattr(args, f"{key}_log"))

    held = np.zeros(200, dtype=bool); held[list(range(19, 200, 20))] = True
    train = ~held
    summary_rows, frame_rows = [], []
    for label, run in runs.items():
        for split, mask in [("training", train), ("held_out", held), ("all", np.ones(200, bool))]:
            m = means(run, mask)
            summary_rows.append({
                "method": label, "split": split, "frames": int(mask.sum()),
                "psnr_db": m["psnr"], "depth_reported_rmse_m": m["rmse"],
                "depth_l1_m": m["l1"], "ms_ssim": m["ssim"], "lpips": m["lpips"],
                "gaussian_count": run["gaussian_count"], "keyframe_count": run["keyframe_count"],
                "params_bytes": run["params_bytes"],
                "mapping_seconds_per_frame": run["runtime"]["mapping_seconds_per_frame"],
                "peak_cuda_memory_gib": run["runtime"]["peak_cuda_memory_gib"],
                "total_runtime_seconds": run["runtime"]["total_runtime_seconds"],
            })
        for frame in range(200):
            frame_rows.append({
                "method": label, "frame_id": frame,
                "split": "held_out" if held[frame] else "training",
                "psnr_db": run["arrays"]["psnr"][frame],
                "depth_l1_m": run["arrays"]["l1"][frame],
                "ms_ssim": run["arrays"]["ssim"][frame],
                "lpips": run["arrays"]["lpips"][frame],
            })
    with open(out / "fair_holdout_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys()); w.writeheader(); w.writerows(summary_rows)
    with open(out / "fair_holdout_per_frame.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=frame_rows[0].keys()); w.writeheader(); w.writerows(frame_rows)

    comparisons = []
    e6 = runs["E6a-ConfHard"]
    for baseline in ["E3-SGBM", "E4-RAFT"]:
        b = runs[baseline]
        item = {"comparison": f"E6a-ConfHard minus {baseline}", "held_out_frames": 10, "metrics": {}}
        for metric, (_, direction) in METRICS.items():
            d = e6["arrays"][metric][held] - b["arrays"][metric][held]
            wins = d > 0 if direction == "higher" else d < 0
            item["metrics"][metric] = {
                "mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
                "e6a_better_frames": int(wins.sum()), "e6a_worse_frames": int((~wins).sum()),
                "direction": direction,
            }
        comparisons.append(item)

    result = {
        "status": "COMPLETE",
        "protocol": "KITTI seq00 frames 0-199; frames 19,39,...,199 excluded from insertion, mapping loss, and keyframes for every method",
        "fixed_pose": "same external ORB-SLAM3 P2/Twc trajectory",
        "held_out_frames": list(range(19, 200, 20)),
        "runs": {label: {
            "training": means(run, train), "held_out": means(run, held), "all": means(run, np.ones(200, bool)),
            "gaussian_count": run["gaussian_count"], "keyframe_count": run["keyframe_count"],
            "params_bytes": run["params_bytes"], "runtime": run["runtime"],
            "heldout_in_keyframes": sorted(set(run["held_out_frames"]).intersection(run["keyframes"])),
        } for label, run in runs.items()},
        "paired_heldout_comparisons": comparisons,
        "evaluator_note": "The inherited rmse.txt averages sqrt(squared per-pixel error), making it numerically identical to L1; depth_l1_m is the honest label.",
    }
    (out / "fair_holdout_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.1))
    x = np.arange(3); labels = list(METHODS.keys())
    for ax, (metric, (title, direction)) in zip(axes, METRICS.items()):
        vals = [runs[label]["arrays"][metric][held].mean() for label in labels]
        ax.bar(x, vals, color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_title(f"Held-out {title} {'↑' if direction == 'higher' else '↓'}")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Fair held-out comparison: identical poses, frames, and mapping settings")
    fig.tight_layout()
    fig.savefig(out / "fair_holdout_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
