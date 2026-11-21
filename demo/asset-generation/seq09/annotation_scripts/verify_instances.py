#!/usr/bin/env python3
"""Strict verifier for Stage 2 T2.1 instance segmentation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import cv2
import numpy as np


CANONICAL_CATEGORIES = {"car", "truck", "bus", "person", "cyclist", "motorcycle"}
DENSE_SEMANTIC_IDS = {
    "person": {11},
    "car": {13},
    "truck": {14},
    "bus": {15},
    "cyclist": {12, 18},
    "motorcycle": {12, 17},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=199)
    parser.add_argument("--max-track-jump", type=int, default=1000000)
    parser.add_argument("--dense-semantic-dir", type=Path)
    parser.add_argument("--max-same-category-mask-iou", type=float, default=0.30)
    parser.add_argument("--max-same-category-containment", type=float, default=0.70)
    parser.add_argument("--max-cross-category-mask-iou", type=float, default=0.80)
    parser.add_argument("--max-cross-category-containment", type=float, default=0.95)
    parser.add_argument("--min-dense-agreement", type=float, default=0.10)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum(dtype=np.int64))
    if intersection == 0:
        return 0.0, 0.0
    area_a = int(a.sum(dtype=np.int64))
    area_b = int(b.sum(dtype=np.int64))
    union = area_a + area_b - intersection
    return (
        intersection / max(union, 1),
        intersection / max(min(area_a, area_b), 1),
    )


def main() -> None:
    args = parse_args()
    expected_frames = args.end - args.start + 1
    json_files = sorted(args.instance_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    mask_files = sorted(args.instance_dir.glob("*_masks.npz"))
    if len(json_files) != expected_frames or len(mask_files) != expected_frames:
        fail(
            f"Expected {expected_frames} JSON/NPZ files, got {len(json_files)}/{len(mask_files)}"
        )

    total_instances = 0
    categories: dict[str, int] = {category: 0 for category in sorted(CANONICAL_CATEGORIES)}
    observed_tracks: set[int] = set()
    track_frames: dict[int, list[int]] = {}
    track_categories: dict[int, str] = {}
    empty_frames = 0
    dense_agreements: list[float] = []
    max_same_category_mask_iou = 0.0
    max_same_category_containment = 0.0
    max_cross_category_mask_iou = 0.0
    max_cross_category_containment = 0.0

    for frame_index in range(args.start, args.end + 1):
        frame_id = f"{frame_index:06d}"
        image_path = args.image_dir / f"{frame_id}.png"
        metadata_path = args.instance_dir / f"{frame_id}.json"
        mask_path = args.instance_dir / f"{frame_id}_masks.npz"
        if not image_path.is_file() or not metadata_path.is_file() or not mask_path.is_file():
            fail(f"Missing aligned artifact for frame {frame_id}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            fail(f"Could not decode {image_path}")
        height, width = image.shape[:2]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        masks = np.load(mask_path, allow_pickle=False)["masks"]
        instances = metadata.get("instances", [])
        dense_labels = None
        if args.dense_semantic_dir is not None:
            dense_path = args.dense_semantic_dir / "labels" / f"{frame_id}_semantic.npy"
            if not dense_path.is_file():
                fail(f"Missing dense semantic labels for {frame_id}")
            dense_labels = np.load(dense_path, allow_pickle=False)
            if dense_labels.dtype != np.uint8 or dense_labels.shape != (height, width):
                fail(f"Bad dense semantic labels for {frame_id}")

        if metadata.get("frame_id") != frame_id:
            fail(f"Bad frame_id in {metadata_path}")
        if metadata.get("image_shape") != [height, width]:
            fail(f"Bad image shape metadata for {frame_id}")
        if masks.dtype != np.bool_ or masks.shape != (len(instances), height, width):
            fail(f"Bad masks dtype/shape for {frame_id}: {masks.dtype} {masks.shape}")
        if len(instances) == 0:
            empty_frames += 1

        frame_tracks: set[int] = set()
        for index, instance in enumerate(instances):
            if instance.get("instance_id") != index or instance.get("mask_index") != index:
                fail(f"Non-contiguous instance/mask index in {frame_id}")
            category = instance.get("category")
            if category not in CANONICAL_CATEGORIES:
                fail(f"Non-canonical category {category!r} in {frame_id}")
            confidence = float(instance.get("confidence", float("nan")))
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                fail(f"Bad confidence in {frame_id}")
            track_id = int(instance.get("track_id", -1))
            if track_id < 0 or track_id >= args.max_track_jump:
                fail(f"Bad track_id {track_id} in {frame_id}")
            if track_id in frame_tracks:
                fail(f"Duplicate track_id {track_id} in frame {frame_id}")
            frame_tracks.add(track_id)
            observed_tracks.add(track_id)
            track_frames.setdefault(track_id, []).append(frame_index)
            previous_category = track_categories.setdefault(track_id, category)
            if previous_category != category:
                fail(f"Track {track_id} changes category in frame {frame_id}")

            mask = masks[index]
            area = int(mask.sum())
            if area <= 0 or area != int(instance.get("mask_area", -1)):
                fail(f"Bad mask area in frame {frame_id}, instance {index}")
            ys, xs = np.nonzero(mask)
            mask_bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
            bbox = np.asarray(instance.get("bbox", []), dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                fail(f"Bad bbox in frame {frame_id}, instance {index}")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                fail(f"Degenerate bbox in frame {frame_id}, instance {index}")
            if bbox[0] < -2 or bbox[1] < -2 or bbox[2] > width + 2 or bbox[3] > height + 2:
                fail(f"Out-of-bounds bbox in frame {frame_id}, instance {index}")
            # A detector bbox may legitimately include occluded/background pixels. The
            # invariant is that the predicted mask must remain inside that bbox.
            tolerance = 4.0
            if (
                bbox[0] > mask_bbox[0] + tolerance
                or bbox[1] > mask_bbox[1] + tolerance
                or bbox[2] < mask_bbox[2] - tolerance
                or bbox[3] < mask_bbox[3] - tolerance
            ):
                fail(f"Mask extends outside bbox in frame {frame_id}, instance {index}")

            expected_dense_ids = DENSE_SEMANTIC_IDS[category]
            stored_dense_ids = {int(value) for value in instance.get("dense_semantic_ids", [])}
            if stored_dense_ids != expected_dense_ids:
                fail(
                    f"Bad dense semantic ID mapping in {frame_id}, instance {index}: "
                    f"{sorted(stored_dense_ids)} != {sorted(expected_dense_ids)}"
                )
            if dense_labels is not None:
                actual_agreement = float(
                    np.isin(dense_labels[mask], list(expected_dense_ids)).mean()
                )
                stored_agreement = float(instance.get("dense_semantic_agreement", float("nan")))
                if not np.isfinite(stored_agreement) or abs(actual_agreement - stored_agreement) > 1e-9:
                    fail(f"Bad dense agreement in {frame_id}, instance {index}")
                if actual_agreement + 1e-12 < args.min_dense_agreement:
                    fail(f"Dense agreement below threshold in {frame_id}, instance {index}")
                dense_agreements.append(actual_agreement)

            total_instances += 1
            categories[category] += 1

        for left in range(len(instances)):
            for right in range(left + 1, len(instances)):
                same_category = instances[left]["category"] == instances[right]["category"]
                iou, containment = mask_overlap(masks[left], masks[right])
                if same_category:
                    max_same_category_mask_iou = max(max_same_category_mask_iou, iou)
                    max_same_category_containment = max(
                        max_same_category_containment, containment
                    )
                    iou_limit = args.max_same_category_mask_iou
                    containment_limit = args.max_same_category_containment
                else:
                    max_cross_category_mask_iou = max(max_cross_category_mask_iou, iou)
                    max_cross_category_containment = max(
                        max_cross_category_containment, containment
                    )
                    iou_limit = args.max_cross_category_mask_iou
                    containment_limit = args.max_cross_category_containment
                if (
                    iou + 1e-12 >= iou_limit
                    or containment + 1e-12 >= containment_limit
                ):
                    fail(
                        f"Duplicate {'same' if same_category else 'cross'}-category masks "
                        f"in {frame_id}: "
                        f"indices {left}/{right}, IoU={iou:.6f}, containment={containment:.6f}"
                    )

    summary_path = args.instance_dir / "generation_summary.json"
    if not summary_path.is_file():
        fail("Missing generation_summary.json")
    generation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if generation_summary.get("frames") != expected_frames:
        fail("Summary frame count does not match requested range")
    if generation_summary.get("total_instances") != total_instances:
        fail("Summary instance count does not match artifacts")

    model = generation_summary.get("model", {})
    model_weight = Path(model.get("weight", ""))
    model_sha256 = model.get("weight_sha256") or generation_summary.get("model_sha256")
    if not model_weight.is_file() or not isinstance(model_sha256, str):
        fail("Missing model weight provenance")
    if sha256_file(model_weight) != model_sha256:
        fail("Model weight SHA256 mismatch")

    if args.dense_semantic_dir is not None:
        dense_manifest = args.dense_semantic_dir / "SHA256SUMS.txt"
        expected_dense_manifest_sha = generation_summary.get("dense_semantic", {}).get(
            "manifest_sha256"
        )
        if not dense_manifest.is_file() or sha256_file(dense_manifest) != expected_dense_manifest_sha:
            fail("Dense semantic manifest SHA256 mismatch")

    track_lengths = [len(frames) for frames in track_frames.values()]
    singleton_tracks = sum(length == 1 for length in track_lengths)

    result = {
        "status": "PASS",
        "frames": expected_frames,
        "frame_range": [args.start, args.end],
        "total_instances": total_instances,
        "unique_tracks": len(observed_tracks),
        "empty_frames": empty_frames,
        "category_instances": categories,
        "model_sha256": model_sha256,
        "inference_ms": generation_summary.get("inference_ms"),
        "end_to_end_ms": generation_summary.get("end_to_end_ms"),
        "dense_semantic_agreement": {
            "mean": statistics.fmean(dense_agreements) if dense_agreements else None,
            "median": statistics.median(dense_agreements) if dense_agreements else None,
            "minimum": min(dense_agreements) if dense_agreements else None,
        },
        "duplicate_mask_check": {
            "max_same_category_iou": max_same_category_mask_iou,
            "max_same_category_containment": max_same_category_containment,
            "same_category_iou_limit": args.max_same_category_mask_iou,
            "same_category_containment_limit": args.max_same_category_containment,
            "max_cross_category_iou": max_cross_category_mask_iou,
            "max_cross_category_containment": max_cross_category_containment,
            "cross_category_iou_limit": args.max_cross_category_mask_iou,
            "cross_category_containment_limit": args.max_cross_category_containment,
        },
        "track_continuity": {
            "mean_observations_per_track": statistics.fmean(track_lengths)
            if track_lengths
            else 0.0,
            "median_observations_per_track": statistics.median(track_lengths)
            if track_lengths
            else 0.0,
            "singleton_tracks": singleton_tracks,
            "singleton_ratio": singleton_tracks / max(len(track_lengths), 1),
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
