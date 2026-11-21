"""Summarize S1 held-out SplaTAM runs into one ablation CSV.

This script reads the frozen ten held-out indices (19, 39, ..., 199). It does
not average the 200-frame eval arrays because those include mapping frames.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
HOLDOUT = np.arange(19, 200, 20, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--splatam-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def extract(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing log field matching: {pattern}")
    return float(match.group(1))


def mean_at(eval_dir: Path, name: str) -> float:
    values = np.loadtxt(eval_dir / f"{name}.txt", dtype=np.float64)
    if values.shape != (200,):
        raise ValueError(f"expected 200 {name} values in {eval_dir}, got {values.shape}")
    return float(values[HOLDOUT].mean())


def main() -> None:
    args = parse_args()
    with args.retention_csv.open("r", encoding="utf-8", newline="") as handle:
        retention = {round(float(row["threshold"]), 2): row for row in csv.DictReader(handle)}
    if set(retention) != set(THRESHOLDS):
        raise ValueError(f"retention CSV must contain exactly {THRESHOLDS}")

    rows: list[dict[str, str | int | float]] = []
    for threshold in THRESHOLDS:
        tag = f"t{int(round(threshold * 100)):03d}"
        if threshold == 0.80:
            run_name = "seq00_e6a_confhard_holdout_200_0"
            log_path = Path("/root/autodl-tmp/holdout_e6a_confhard_200.log")
            status = "HISTORICAL_GPU_BASELINE_VERIFIED"
        else:
            run_name = f"seq00_s1_confhard_{tag}_holdout_200_0"
            log_path = args.evidence_root / "logs" / f"s1_{tag}_holdout_200.log"
            status = "GPU_MAPPING_COMPLETE"
        experiment = args.splatam_root / "experiments" / "KITTI" / run_name
        eval_dir = experiment / "eval"
        params_path = experiment / "params.npz"
        if not eval_dir.is_dir() or not params_path.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"incomplete run for threshold={threshold:.2f}: {experiment}")

        text = log_path.read_text(encoding="utf-8", errors="replace")
        with np.load(params_path, allow_pickle=False) as params:
            final_gaussians = int(params["means3D"].shape[0])
            stored_holdout = params["eval_only_frames"].astype(np.int64)
            if not np.array_equal(stored_holdout, HOLDOUT):
                raise ValueError(f"held-out indices changed for {run_name}: {stored_holdout.tolist()}")

        source = retention[threshold]
        rows.append(
            {
                "threshold": threshold,
                "status": status,
                "valid_depth_ratio_total": float(source["valid_depth_ratio_total"]),
                "eligible_ratio_total": float(source["eligible_ratio_total"]),
                "eligible_ratio_valid_depth": float(source["eligible_ratio_valid_depth"]),
                "initial_gaussians": int(source["frame0_eligible_pixels"]),
                "final_gaussians": final_gaussians,
                "heldout_psnr": mean_at(eval_dir, "psnr"),
                "heldout_ms_ssim": mean_at(eval_dir, "ssim"),
                "heldout_lpips": mean_at(eval_dir, "lpips"),
                "heldout_depth_l1_m": mean_at(eval_dir, "l1"),
                "mapping_time_s_per_mapped_frame": extract(
                    r"Average Mapping/Frame Time:\s*([0-9.]+)\s*s", text
                ),
                "peak_cuda_gib": extract(r"11R Peak CUDA Memory Allocated:\s*([0-9.]+)\s*GiB", text),
                "total_runtime_s": extract(r"11R Total Runtime:\s*([0-9.]+)\s*s", text),
                "heldout_indices": ";".join(str(value) for value in HOLDOUT.tolist()),
                "run_name": run_name,
                "log_path": str(log_path),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_csv}")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
