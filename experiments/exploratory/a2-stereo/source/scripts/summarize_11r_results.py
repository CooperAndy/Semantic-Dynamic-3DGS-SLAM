"""Create split-aware 11R metrics from a completed SplaTAM run."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    eval_dir, out = Path(args.eval_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["psnr", "rmse", "l1", "ssim", "lpips"]
    values = {name: np.atleast_1d(np.loadtxt(eval_dir / f"{name}.txt")).astype(float) for name in names}
    lengths = {name: len(v) for name, v in values.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"metric length mismatch: {lengths}")

    raw = np.load(args.params, allow_pickle=True)
    num_frames = raw["cam_trans"].shape[-1]
    if lengths["psnr"] != num_frames:
        raise ValueError(f"eval_every=1 required: metrics={lengths['psnr']} frames={num_frames}")
    held_out = sorted(int(x) for x in raw["eval_only_frames"].tolist())
    held_set = set(held_out)
    split_masks = {
        "all": np.ones(num_frames, dtype=bool),
        "training": np.array([i not in held_set for i in range(num_frames)]),
        "held_out": np.array([i in held_set for i in range(num_frames)]),
    }

    frame_rows = []
    for i in range(num_frames):
        frame_rows.append({
            "frame_id": i,
            "split": "held_out" if i in held_set else "training",
            "psnr_db": values["psnr"][i],
            "depth_reported_rmse_m": values["rmse"][i],
            "depth_l1_m": values["l1"][i],
            "ms_ssim": values["ssim"][i],
            "lpips": values["lpips"][i],
        })
    with open(out / "heldout_per_frame_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
        w.writeheader(); w.writerows(frame_rows)

    split_rows = []
    for split, mask in split_masks.items():
        split_rows.append({
            "split": split,
            "frames": int(mask.sum()),
            "psnr_db": float(values["psnr"][mask].mean()),
            "depth_reported_rmse_m": float(values["rmse"][mask].mean()),
            "depth_l1_m": float(values["l1"][mask].mean()),
            "ms_ssim": float(values["ssim"][mask].mean()),
            "lpips": float(values["lpips"][mask].mean()),
        })
    with open(out / "heldout_split_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=split_rows[0].keys())
        w.writeheader(); w.writerows(split_rows)

    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    patterns = {
        "tracking_frame_seconds": r"Average Tracking/Frame Time: ([0-9.eE+-]+) s",
        "mapping_iteration_ms": r"Average Mapping/Iteration Time: ([0-9.eE+-]+) ms",
        "mapping_frame_seconds": r"Average Mapping/Frame Time: ([0-9.eE+-]+) s",
        "total_runtime_seconds": r"11R Total Runtime: ([0-9.eE+-]+) s",
        "peak_cuda_memory_gib": r"11R Peak CUDA Memory Allocated: ([0-9.eE+-]+) GiB",
    }
    runtime = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text)
        runtime[key] = float(matches[-1]) if matches else None

    by_split = {row["split"]: row for row in split_rows}
    summary = {
        "status": "COMPLETE",
        "scope": f"KITTI seq00 frames 0-{num_frames-1}",
        "protocol": "pre-registered evaluation-only frames are excluded from Gaussian insertion, mapping loss, and keyframes",
        "held_out_frames": held_out,
        "metrics": by_split,
        "generalization_gap_heldout_minus_training": {
            k: by_split["held_out"][k] - by_split["training"][k]
            for k in ["psnr_db", "depth_reported_rmse_m", "depth_l1_m", "ms_ssim", "lpips"]
        },
        "map": {
            "gaussian_count": int(raw["means3D"].shape[0]),
            "keyframe_count": int(raw["keyframe_time_indices"].shape[0]),
            "keyframe_time_indices": [int(x) for x in raw["keyframe_time_indices"].tolist()],
            "heldout_in_keyframes": sorted(set(held_out).intersection(int(x) for x in raw["keyframe_time_indices"].tolist())),
            "params_bytes": Path(args.params).stat().st_size,
        },
        "runtime": runtime,
        "evaluator_note": "The inherited SplaTAM 'rmse' implementation averages sqrt(squared per-pixel error), so it is numerically identical to depth L1 and is reported with its original filename plus this caveat.",
        "pose_note": "SplaTAM ATE=0 compares the fixed ORB poses with the same pose input; use the independently measured ORB ATE/RPE instead.",
    }
    with open(out / "heldout_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
