"""Recompute KITTI08 confidence using only members of a frozen Train subset.

This imports the frozen original stereo/temporal primitives, but never reads
the old 200-frame confidence maps, Dev/Test RGB, or Dev/Test depth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from build_depth_confidence_kitti import (
    confidence_from_stereo, load_depth, load_poses, parse_calibration,
    temporal_confidence,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "smoke"], required=True)
    args = parser.parse_args()
    root = args.protocol_root.resolve()
    source = Path("/root/autodl-tmp/seq21_chy_20260926")
    original = source / "sequences/21_1200_1399"
    split_path = root / "split.csv"
    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    if sha256(split_path) != manifest["split_sha256"]:
        raise ValueError("split.csv hash differs from frozen manifest")
    with split_path.open(encoding="utf-8", newline="") as handle:
        split = {int(row["local_index"]): row["split"] for row in csv.DictReader(handle)}
    index_path = root / ("mapping_index_train160.csv" if args.mode == "full" else "mapping_index_smoke10.csv")
    with index_path.open(encoding="utf-8", newline="") as handle:
        ids = [int(row["source_local_index"]) for row in csv.DictReader(handle)]
    expected = 160 if args.mode == "full" else 10
    if len(ids) != expected or len(set(ids)) != expected or any(split[i] != "Train" for i in ids):
        raise ValueError("selected indices are not unique Train members")
    if ids != sorted(ids):
        raise ValueError("Train IDs must be chronological")
    output = root / f"confidence_{args.mode}"
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")

    shape = (376, 1241)
    calibration = parse_calibration(original / "calib.txt")
    poses = load_poses(source / "tracking/poses_orb_p2_c2w_rebased.txt", 200)
    scale = calibration["fx"] * calibration["baseline_m"]
    depths: dict[int, np.ndarray] = {}
    for i in ids:
        depth_path = source / "raft_left" / f"{i:06d}_depth.npy"
        depths[i] = load_depth(depth_path, shape)

    output.mkdir(parents=True, exist_ok=False)
    for folder in ("confidence", "seed_mask", "loss_mask", "depth_confhard"):
        (output / folder).mkdir()
    audit = []
    for position, i in enumerate(ids):
        name = f"{i:06d}"
        left_path = original / "image_2" / f"{name}.png"
        right_path = original / "image_3" / f"{name}.png"
        left_bgr, right_bgr = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        if left_bgr is None or right_bgr is None or left_bgr.shape[:2] != shape or right_bgr.shape[:2] != shape:
            raise ValueError(f"bad stereo pair {name}")
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        depth = depths[i]
        right_depth = load_depth(source / "raft_right" / f"{name}_depth.npy", shape)
        static_mask = np.load(source / "static_valid_masks" / f"{name}_valid.npy", allow_pickle=False)
        if static_mask.dtype != np.bool_ or static_mask.shape != shape:
            raise ValueError(f"bad static mask {name}")
        valid = (depth >= 0.5) & (depth <= 40.0) & static_mask
        stereo, _, _, _ = confidence_from_stereo(left, right, depth, right_depth,
                                                 scale, valid, 1.0, 0.10)
        neighbours = [ids[j] for j in (position - 1, position + 1) if 0 <= j < len(ids)]
        if any(split[j] != "Train" for j in neighbours):
            raise AssertionError(f"non-Train temporal neighbour for {name}")
        temporal = []
        for j in neighbours:
            conf, _, _ = temporal_confidence(depth, depths[j], poses[i], poses[j],
                                             calibration, valid, 0.10)
            temporal.append(conf)
        temporal_factor = np.maximum.reduce(temporal) if temporal else np.ones(shape, dtype=np.float32)
        confidence = (stereo * temporal_factor).astype(np.float32)
        confidence[~valid] = 0.0
        seed = confidence >= 0.60
        loss = confidence >= 0.20
        hard_depth = depth.copy()
        hard_depth[~seed] = 0.0
        np.save(output / "confidence" / f"{name}_confidence.npy", confidence, allow_pickle=False)
        np.save(output / "seed_mask" / f"{name}_seed_valid.npy", seed, allow_pickle=False)
        np.save(output / "loss_mask" / f"{name}_loss_valid.npy", loss, allow_pickle=False)
        np.save(output / "depth_confhard" / f"{name}_depth.npy", hard_depth, allow_pickle=False)
        audit.append({
            "mapping_index": position, "source_local_index": i, "original_frame_id": 1200 + i,
            "split": split[i], "temporal_neighbor_local_ids": ";".join(str(j) for j in neighbours),
            "temporal_neighbor_original_frames": ";".join(str(1200 + j) for j in neighbours),
            "static_roi_fraction": float(static_mask.mean()), "valid_depth_fraction": float(valid.mean()),
            "seed_fraction": float(seed.mean()), "loss_fraction": float(loss.mean()),
            "mean_confidence": float(confidence.mean()),
        })
        if position == 0 or (position + 1) % 10 == 0 or position + 1 == len(ids):
            print(f"[{position + 1}/{len(ids)}] original={1200+i} neighbours={neighbours} seed={seed.mean():.3f}", flush=True)

    with (output / "neighbour_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "mode": args.mode, "count": len(ids), "split_sha256": manifest["split_sha256"],
        "neighbour_policy": "nearest preceding/following selected Train IDs only",
        "no_dev_test_neighbour": all(split[j] == "Train" for row in audit for j in
                                     [int(x) for x in row["temporal_neighbor_local_ids"].split(";") if x]),
        "mean_seed_fraction": float(np.mean([row["seed_fraction"] for row in audit])),
        "neighbour_audit_sha256": sha256(output / "neighbour_audit.csv"),
        "source_helper_sha256": sha256(Path(__file__).with_name("build_depth_confidence_kitti.py")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["no_dev_test_neighbour"]:
        raise AssertionError("temporal neighbour audit failed")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
