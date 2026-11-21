"""Extend the frozen four-threshold S1 tables with the verified T=0.50 run.

The 2026-09-17 inputs live in results/archive_4_threshold_20260917.
This script does not infer quality from the threshold-specific legacy metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FROZEN = RESULTS / "archive_4_threshold_20260917"
NEW = ROOT / "server_results" / "s1_t050_20260929"
BUNDLE = ROOT / "server_results" / "final_bundle_20260929"
RUN = NEW / "experiments" / "seq00_s1_confhard_t050_holdout_200_0"
HELDOUT = np.arange(19, 200, 20, dtype=np.int64)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_number(pattern: str, log: str) -> float:
    match = re.search(pattern, log)
    if not match:
        raise ValueError(f"missing mapping log field: {pattern}")
    return float(match.group(1))


def legacy_mean(name: str) -> float:
    values = np.loadtxt(RUN / "eval" / f"{name}.txt", dtype=np.float64)
    if values.shape != (200,):
        raise ValueError(f"expected 200 legacy {name} values, found {values.shape}")
    return float(values[HELDOUT].mean())


def main() -> None:
    base_ablation = read_csv(FROZEN / "confidence_threshold_ablation.csv")
    base_retention = read_csv(FROZEN / "confidence_threshold_retention.csv")
    base_per_frame = read_csv(FROZEN / "per_frame_threshold_retention.csv")
    if [float(r["threshold"]) for r in base_ablation] != [0.6, 0.7, 0.8, 0.9]:
        raise ValueError("frozen ablation thresholds changed")
    if [float(r["threshold"]) for r in base_retention] != [0.6, 0.7, 0.8, 0.9]:
        raise ValueError("frozen retention thresholds changed")
    if len(base_per_frame) != 800:
        raise ValueError("expected 800 frozen per-frame rows")

    new_retention = read_csv(NEW / "input_audit" / "confidence_threshold_retention.csv")
    new_per_frame = read_csv(NEW / "input_audit" / "per_frame_threshold_retention.csv")
    if len(new_retention) != 1 or float(new_retention[0]["threshold"]) != 0.5:
        raise ValueError("new retention must contain exactly T=0.50")
    if len(new_per_frame) != 200 or {float(r["threshold"]) for r in new_per_frame} != {0.5}:
        raise ValueError("expected 200 new T=0.50 per-frame rows")
    if [r["frame_id"] for r in new_per_frame] != [f"{i:06d}" for i in range(200)]:
        raise ValueError("T=0.50 frame IDs changed")
    if new_retention[0]["valid_depth_pixels"] != base_retention[0]["valid_depth_pixels"]:
        raise ValueError("valid-depth population changed")

    summary = json.loads((NEW / "common_support" / "common_support_summary.json").read_text())
    if summary["threshold"] != 0.5 or summary["heldout_frames"] != HELDOUT.tolist():
        raise ValueError("T=0.50 common-support protocol changed")
    frame_rows = read_csv(NEW / "common_support" / "common_support_per_frame.csv")
    if [int(r["frame"]) for r in frame_rows] != HELDOUT.tolist():
        raise ValueError("T=0.50 common-support per-frame list changed")
    if "EXIT_CODE 0" not in (NEW / "mapping_status.txt").read_text():
        raise ValueError("mapping did not finish successfully")
    log = (NEW / "logs" / "mapping.log").read_text(errors="replace")
    with np.load(RUN / "params.npz", allow_pickle=False) as params:
        if params["eval_only_frames"].tolist() != HELDOUT.tolist():
            raise ValueError("saved map held-out frames changed")
        final_gaussians = int(params["means3D"].shape[0])

    retention = [new_retention[0], *base_retention]
    write_csv(RESULTS / "confidence_threshold_retention.csv", retention, list(base_retention[0]))
    per_frame = [*new_per_frame, *base_per_frame]
    per_frame.sort(key=lambda row: (float(row["threshold"]), int(row["index"])))
    write_csv(RESULTS / "per_frame_threshold_retention.csv", per_frame, list(base_per_frame[0]))

    metrics = summary["metrics"]
    source = new_retention[0]
    t050: dict[str, object] = {
        "threshold": "0.50",
        "status": "S1_T050_EXTENSION_COMMON_SUPPORT_COMPLETE_20260929",
        "valid_depth_ratio_total": source["valid_depth_ratio_total"],
        "eligible_ratio_total": source["eligible_ratio_total"],
        "eligible_ratio_valid_depth": source["eligible_ratio_valid_depth"],
        "initial_gaussians": source["frame0_eligible_pixels"],
        "final_gaussians": final_gaussians,
        **{key: metrics[key] for key in (
            "full_rgb_psnr", "full_rgb_ms_ssim", "full_rgb_lpips",
            "common_depth_l1_all_valid_m", "common_depth_l1_supported_m",
            "silhouette_coverage_all", "silhouette_coverage_common_depth",
            "hole_ratio_common_depth",
        )},
        "mapping_time_s_per_mapped_frame": log_number(
            r"Average Mapping/Frame Time:\s*([0-9.]+)\s*s", log
        ),
        "peak_cuda_gib": log_number(r"11R Peak CUDA Memory Allocated:\s*([0-9.]+)\s*GiB", log),
        "total_runtime_s": log_number(r"11R Total Runtime:\s*([0-9.]+)\s*s", log),
        "legacy_threshold_specific_psnr": legacy_mean("psnr"),
        "legacy_threshold_specific_ms_ssim": legacy_mean("ssim"),
        "legacy_threshold_specific_lpips": legacy_mean("lpips"),
        "legacy_threshold_specific_depth_l1_m": legacy_mean("l1"),
    }
    ablation = [t050, *base_ablation]
    write_csv(RESULTS / "confidence_threshold_ablation.csv", ablation, list(base_ablation[0]))

    old_complete = read_csv(ROOT / "server_results" / "final_bundle" / "confidence_threshold_ablation_complete.csv")
    if len(old_complete) != 4:
        raise ValueError("expected four frozen legacy mapping rows")
    t050_complete: dict[str, object] = {
        "threshold": 0.5,
        "status": "S1_T050_EXTENSION_GPU_MAPPING_COMPLETE_20260929",
        "valid_depth_ratio_total": source["valid_depth_ratio_total"],
        "eligible_ratio_total": source["eligible_ratio_total"],
        "eligible_ratio_valid_depth": source["eligible_ratio_valid_depth"],
        "initial_gaussians": source["frame0_eligible_pixels"],
        "final_gaussians": final_gaussians,
        "heldout_psnr": t050["legacy_threshold_specific_psnr"],
        "heldout_ms_ssim": t050["legacy_threshold_specific_ms_ssim"],
        "heldout_lpips": t050["legacy_threshold_specific_lpips"],
        "heldout_depth_l1_m": t050["legacy_threshold_specific_depth_l1_m"],
        "mapping_time_s_per_mapped_frame": t050["mapping_time_s_per_mapped_frame"],
        "peak_cuda_gib": t050["peak_cuda_gib"],
        "total_runtime_s": t050["total_runtime_s"],
        "heldout_indices": ";".join(map(str, HELDOUT.tolist())),
        "run_name": RUN.name,
        "log_path": "/root/autodl-tmp/s1_t050_chy_20260929/logs/mapping.log",
    }
    write_csv(
        BUNDLE / "confidence_threshold_ablation_complete.csv",
        [t050_complete, *old_complete], list(old_complete[0]),
    )
    old_legacy_frames = read_csv(ROOT / "server_results" / "final_bundle" / "heldout_per_frame_metrics.csv")
    if len(old_legacy_frames) != 40:
        raise ValueError("expected 40 frozen legacy held-out frame rows")
    legacy_values = {
        name: np.loadtxt(RUN / "eval" / f"{name}.txt", dtype=np.float64)
        for name in ("psnr", "ssim", "lpips", "l1")
    }
    new_legacy_frames = [
        {"threshold": 0.5, "frame": int(frame),
         **{name: float(values[frame]) for name, values in legacy_values.items()}}
        for frame in HELDOUT
    ]
    write_csv(
        BUNDLE / "heldout_per_frame_metrics.csv",
        [*new_legacy_frames, *old_legacy_frames], list(old_legacy_frames[0]),
    )
    write_csv(BUNDLE / "confidence_threshold_retention.csv", retention, list(base_retention[0]))
    write_csv(BUNDLE / "confidence_threshold_ablation_five_common_support.csv", ablation, list(base_ablation[0]))
    write_csv(BUNDLE / "per_frame_threshold_retention.csv", per_frame, list(base_per_frame[0]))

    original = json.loads((FROZEN / "s1_final_evidence.json").read_text())
    evidence = {
        "status": "S1_FIVE_THRESHOLDS_COMMON_SUPPORT_COMPLETE",
        "updated_at": "2026-09-29",
        "scope": original["scope"],
        "tested_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
        "original_preregistered_thresholds": original["tested_thresholds"],
        "posthoc_extension_threshold": 0.5,
        "recommended_tested_threshold": 0.6,
        "selection_rule": original["selection_rule"],
        "common_support_protocol": original["common_support_protocol"],
        "legacy_protocol_warning": original["legacy_protocol_warning"],
        "frozen_four_threshold_evidence": "results/archive_4_threshold_20260917/s1_final_evidence.json",
        "t050_archive": {
            "path": "server_results/s1_t050_bundle_20260929.tgz",
            "sha256": sha256(ROOT / "server_results" / "s1_t050_bundle_20260929.tgz"),
            "mapping_params_sha256": sha256(RUN / "params.npz"),
            "common_support_summary_sha256": sha256(NEW / "common_support" / "common_support_summary.json"),
            "mapping_log_sha256": sha256(NEW / "logs" / "mapping.log"),
        },
    }
    (RESULTS / "s1_final_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (BUNDLE / "s1_final_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ablation)} ablation, {len(retention)} retention, {len(per_frame)} per-frame rows")
    print(f"T=0.50 legacy held-out PSNR {t050['legacy_threshold_specific_psnr']:.4f} (diagnostic only)")


if __name__ == "__main__":
    main()
