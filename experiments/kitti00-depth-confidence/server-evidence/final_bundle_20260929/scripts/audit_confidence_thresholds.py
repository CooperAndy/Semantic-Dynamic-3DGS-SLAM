"""Audit confidence-gate retention and materialize hard-depth variants.

The script streams one KITTI frame at a time so it can run on a small CPU-only
instance.  It does not evaluate mapping quality: it only derives reproducible
gate statistics and depth inputs for a later, GPU-backed SplaTAM ablation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80, 0.90]
    )
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=40.0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument(
        "--write-hard-depth",
        action="store_true",
        help="Write one depth folder per threshold; existing files are never overwritten.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_id_from_confidence(path: Path) -> str:
    return path.name.removesuffix("_confidence.npy")


def frame_id_from_depth(path: Path) -> str:
    return path.name.removesuffix("_depth.npy")


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be unique values in (0, 1]")
    if not 0.0 <= args.min_depth < args.max_depth:
        raise ValueError("expected 0 <= min-depth < max-depth")

    confidence_paths = sorted(args.confidence_dir.glob("*_confidence.npy"))
    depth_paths = sorted(args.depth_dir.glob("*_depth.npy"))
    if len(confidence_paths) != args.num_frames or len(depth_paths) != args.num_frames:
        raise ValueError(
            f"expected {args.num_frames} frames, found confidence={len(confidence_paths)} "
            f"depth={len(depth_paths)}"
        )
    confidence_ids = [frame_id_from_confidence(path) for path in confidence_paths]
    depth_ids = [frame_id_from_depth(path) for path in depth_paths]
    if confidence_ids != depth_ids:
        raise ValueError("confidence/depth frame IDs do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "confidence_threshold_retention.csv"
    per_frame_path = args.output_dir / "per_frame_threshold_retention.csv"
    manifest_path = args.output_dir / "input_manifest_sha256.csv"
    audit_path = args.output_dir / "audit_summary.json"
    for path in (summary_path, per_frame_path, manifest_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    hard_depth_dirs: dict[float, Path] = {}
    if args.write_hard_depth:
        for threshold in thresholds:
            directory = args.output_dir / f"depth_confhard_t{threshold:.2f}"
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"refusing non-empty hard-depth directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            hard_depth_dirs[threshold] = directory

    total_pixels = 0
    valid_pixels = 0
    positive_confidence_pixels = 0
    eligible_counts = {threshold: 0 for threshold in thresholds}
    per_frame_rows: list[dict[str, int | float | str]] = []
    manifest_rows: list[dict[str, str | int]] = []
    expected_shape: tuple[int, ...] | None = None

    for index, (frame_id, confidence_path, depth_path) in enumerate(
        zip(confidence_ids, confidence_paths, depth_paths, strict=True)
    ):
        confidence = np.load(confidence_path, allow_pickle=False)
        depth = np.load(depth_path, allow_pickle=False)
        if confidence.dtype != np.float32 or depth.dtype != np.float32:
            raise ValueError(f"frame {frame_id}: expected float32 arrays")
        if confidence.shape != depth.shape or confidence.ndim != 2:
            raise ValueError(f"frame {frame_id}: confidence/depth shape mismatch")
        if expected_shape is None:
            expected_shape = confidence.shape
        if confidence.shape != expected_shape:
            raise ValueError(f"frame {frame_id}: unexpected shape {confidence.shape}")
        if not np.isfinite(confidence).all() or not np.isfinite(depth).all():
            raise ValueError(f"frame {frame_id}: non-finite input")
        if float(confidence.min()) < 0.0 or float(confidence.max()) > 1.0:
            raise ValueError(f"frame {frame_id}: confidence outside [0, 1]")

        valid = (depth >= args.min_depth) & (depth <= args.max_depth)
        frame_total = int(depth.size)
        frame_valid = int(valid.sum())
        frame_positive = int(((confidence > 0.0) & valid).sum())
        total_pixels += frame_total
        valid_pixels += frame_valid
        positive_confidence_pixels += frame_positive

        for threshold in thresholds:
            eligible = (confidence >= threshold) & valid
            frame_eligible = int(eligible.sum())
            eligible_counts[threshold] += frame_eligible
            per_frame_rows.append(
                {
                    "index": index,
                    "frame_id": frame_id,
                    "threshold": threshold,
                    "total_pixels": frame_total,
                    "valid_depth_pixels": frame_valid,
                    "eligible_seed_pixels": frame_eligible,
                    "valid_depth_ratio_total": frame_valid / frame_total,
                    "eligible_ratio_total": frame_eligible / frame_total,
                    "eligible_ratio_valid_depth": frame_eligible / frame_valid
                    if frame_valid
                    else 0.0,
                }
            )
            if args.write_hard_depth:
                hard_depth = depth.copy()
                hard_depth[~eligible] = 0.0
                output_path = hard_depth_dirs[threshold] / f"{frame_id}_depth.npy"
                np.save(output_path, hard_depth.astype(np.float32, copy=False), allow_pickle=False)

        manifest_rows.extend(
            [
                {
                    "frame_id": frame_id,
                    "kind": "confidence",
                    "path": str(confidence_path),
                    "size_bytes": confidence_path.stat().st_size,
                    "sha256": sha256_file(confidence_path),
                },
                {
                    "frame_id": frame_id,
                    "kind": "depth",
                    "path": str(depth_path),
                    "size_bytes": depth_path.stat().st_size,
                    "sha256": sha256_file(depth_path),
                },
            ]
        )
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == args.num_frames:
            print(f"[{index + 1:03d}/{args.num_frames}] {frame_id}", flush=True)

    summary_rows: list[dict[str, int | float]] = []
    for threshold in thresholds:
        eligible = eligible_counts[threshold]
        summary_rows.append(
            {
                "threshold": threshold,
                "frames": args.num_frames,
                "total_pixels": total_pixels,
                "valid_depth_pixels": valid_pixels,
                "positive_confidence_pixels": positive_confidence_pixels,
                "eligible_seed_pixels": eligible,
                "valid_depth_ratio_total": valid_pixels / total_pixels,
                "positive_confidence_ratio_total": positive_confidence_pixels / total_pixels,
                "eligible_ratio_total": eligible / total_pixels,
                "eligible_ratio_valid_depth": eligible / valid_pixels,
                "removed_ratio_from_valid_depth": 1.0 - eligible / valid_pixels,
                "frame0_eligible_pixels": next(
                    int(row["eligible_seed_pixels"])
                    for row in per_frame_rows
                    if row["index"] == 0 and row["threshold"] == threshold
                ),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with per_frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame_rows[0]))
        writer.writeheader()
        writer.writerows(per_frame_rows)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "status": "CPU_GATE_AUDIT_COMPLETE_GPU_MAPPING_NOT_EVALUATED",
        "scope": f"KITTI seq00 frames 000000-{args.num_frames - 1:06d}",
        "shape": list(expected_shape or ()),
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "thresholds": thresholds,
        "formula": "eligible = valid_depth[0.5m,40m] AND confidence >= threshold",
        "summary_csv": summary_path.name,
        "per_frame_csv": per_frame_path.name,
        "manifest_csv": manifest_path.name,
        "hard_depth_written": args.write_hard_depth,
        "mapping_metrics_assessed": False,
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"audit": audit, "rows": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
