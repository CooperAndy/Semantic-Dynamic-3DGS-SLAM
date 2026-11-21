from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


HELDOUT = tuple(range(19, 200, 20))
MODES = ("m025", "m050", "m100", "m190", "noheldout200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splatam-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def extract_number(text: str, pattern: str) -> float:
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"missing log pattern: {pattern}")
    return float(matches[-1])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result_root = args.evidence_root / "results" / "s2_frame_efficiency_20260919"
    log_root = args.evidence_root / "logs" / "s2_frame_efficiency_20260919"
    run_names = {
        "m025": "seq00_s2_confhard_t060_m025_holdout10_0",
        "m050": "seq00_s2_confhard_t060_m050_holdout10_0",
        "m100": "seq00_s2_confhard_t060_m100_holdout10_0",
        "m190": "seq00_s1_confhard_t060_holdout_200_0",
        "noheldout200": "seq00_s2_confhard_t060_noheldout_200_0",
    }
    rows = []
    validations = {}
    for mode in MODES:
        params_path = args.splatam_root / "experiments" / "KITTI" / run_names[mode] / "params.npz"
        log_path = (
            args.evidence_root / "logs" / "s1_t060_holdout_200.log"
            if mode == "m190"
            else log_root / f"{mode}.log"
        )
        common_path = result_root / "common_support" / mode / "common_support_summary.json"
        if not params_path.is_file() or not log_path.is_file() or not common_path.is_file():
            raise FileNotFoundError((params_path, log_path, common_path))

        with np.load(params_path, allow_pickle=True) as archive:
            final_gaussians = int(archive["means3D"].shape[0])
            initial_gaussians = int(np.count_nonzero(archive["timestep"] == 0))
            keyframes = tuple(int(value) for value in archive["keyframe_time_indices"].tolist())
            eval_only = tuple(int(value) for value in archive["eval_only_frames"].tolist())
            if "mapping_frame_indices" in archive.files:
                mapping_frames = tuple(int(value) for value in archive["mapping_frame_indices"].tolist())
            elif mode == "m190":
                mapping_frames = tuple(manifest["mapping_budgets"]["190"])
            else:
                raise ValueError(f"{mode} lacks mapping_frame_indices")

        expected_mapping = (
            tuple(manifest["noheldout_mapping_frames"])
            if mode == "noheldout200"
            else tuple(manifest["mapping_budgets"][str(int(mode[1:]))])
        )
        if mapping_frames != expected_mapping:
            raise ValueError(f"{mode} mapping frames changed")
        if mode == "noheldout200":
            if eval_only or not set(HELDOUT).issubset(mapping_frames):
                raise ValueError("no-held-out run did not include all matched evaluation views")
            role = "in_sample"
        else:
            if eval_only != HELDOUT:
                raise ValueError(f"{mode} held-out frames changed: {eval_only}")
            if set(HELDOUT).intersection(mapping_frames) or set(HELDOUT).intersection(keyframes):
                raise ValueError(f"{mode} held-out leakage")
            role = "heldout"

        common = json.loads(common_path.read_text(encoding="utf-8"))
        if common["evaluation_role"] != role or tuple(common["evaluation_frames"]) != HELDOUT:
            raise ValueError(f"{mode} common-support evaluation role/frames changed")
        metrics = common["metrics"]
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row = {
            "mode": mode,
            "evaluation_role": role,
            "mapping_frames": len(mapping_frames),
            "mapping_fraction_of_190": len(mapping_frames) / 190.0,
            "heldout_frames": 0 if mode == "noheldout200" else len(HELDOUT),
            "initial_gaussians": initial_gaussians,
            "final_gaussians": final_gaussians,
            "full_rgb_psnr": metrics["full_rgb_psnr"],
            "full_rgb_ms_ssim": metrics["full_rgb_ms_ssim"],
            "full_rgb_lpips": metrics["full_rgb_lpips"],
            "common_depth_l1_all_valid_m": metrics["common_depth_l1_all_valid_m"],
            "silhouette_coverage_common_depth": metrics["silhouette_coverage_common_depth"],
            "hole_ratio_common_depth": metrics["hole_ratio_common_depth"],
            "mapping_time_s_per_mapped_frame": extract_number(log_text, r"Average Mapping/Frame Time:\s*([0-9.]+)\s*s"),
            "peak_cuda_gib": extract_number(log_text, r"(?:S2|11R) Peak CUDA Memory Allocated:\s*([0-9.]+)\s*GiB"),
            "total_runtime_s": extract_number(log_text, r"(?:S2|11R) Total Runtime:\s*([0-9.]+)\s*s"),
            "keyframe_count": len(keyframes),
            "run_name": run_names[mode],
        }
        rows.append(row)
        validations[mode] = {
            "mapping_frames_match_manifest": True,
            "evaluation_role": role,
            "evaluation_frames": list(HELDOUT),
            "heldout_mapping_overlap": sorted(set(HELDOUT).intersection(mapping_frames)),
            "heldout_keyframe_overlap": sorted(set(HELDOUT).intersection(keyframes)),
        }

    m190 = next(row for row in rows if row["mode"] == "m190")
    for row in rows:
        row["psnr_delta_vs_m190"] = row["full_rgb_psnr"] - m190["full_rgb_psnr"]
        row["lpips_delta_vs_m190"] = row["full_rgb_lpips"] - m190["full_rgb_lpips"]
        row["depth_l1_delta_vs_m190_m"] = row["common_depth_l1_all_valid_m"] - m190["common_depth_l1_all_valid_m"]
        row["mapping_time_total_estimate_s"] = row["mapping_time_s_per_mapped_frame"] * row["mapping_frames"]

    output_csv = args.output_dir / "s2_mapping_frame_efficiency.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    noheldout = next(row for row in rows if row["mode"] == "noheldout200")
    summary = {
        "status": "FORMAL_S2_COMMON_SUPPORT_COMPLETE",
        "protocol_manifest": str(args.manifest),
        "budget_curve_modes": ["m025", "m050", "m100", "m190"],
        "matched_view_noheldout_comparison": {
            "heldout_mode": "m190",
            "in_sample_mode": "noheldout200",
            "psnr_difference_in_sample_minus_heldout": noheldout["full_rgb_psnr"] - m190["full_rgb_psnr"],
            "ms_ssim_difference": noheldout["full_rgb_ms_ssim"] - m190["full_rgb_ms_ssim"],
            "lpips_difference": noheldout["full_rgb_lpips"] - m190["full_rgb_lpips"],
            "depth_l1_difference_m": noheldout["common_depth_l1_all_valid_m"] - m190["common_depth_l1_all_valid_m"],
            "warning": "The no-held-out score is in-sample reconstruction and is not a generalization metric.",
        },
        "validations": validations,
    }
    (args.output_dir / "s2_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
