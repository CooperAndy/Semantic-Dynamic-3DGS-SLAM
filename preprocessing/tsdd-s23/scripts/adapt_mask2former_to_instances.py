#!/usr/bin/env python3
"""Convert dense Cityscapes-19 semantics into tracked pseudo-instances.

This adapter is intentionally deterministic. It preserves the dense semantic
prediction as the source of truth and only adds connected-component grouping
and short-term tracking required by the existing T2.2/T2.3 interfaces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CITYSCAPES_IDS = {
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}
DIRECT_CATEGORIES = {
    CITYSCAPES_IDS["person"]: "person",
    CITYSCAPES_IDS["car"]: "car",
    CITYSCAPES_IDS["truck"]: "truck",
    CITYSCAPES_IDS["bus"]: "bus",
}
CANONICAL_CATEGORIES = {"car", "truck", "bus", "person", "cyclist", "motorcycle"}
COLORS = {
    "car": (30, 80, 230),
    "truck": (40, 160, 240),
    "bus": (50, 210, 210),
    "person": (210, 80, 50),
    "cyclist": (180, 60, 200),
    "motorcycle": (80, 200, 80),
}


@dataclass
class Detection:
    category: str
    confidence: float
    bbox: np.ndarray
    mask: np.ndarray
    source_categories: list[str]
    source_label_ids: list[int]
    track_id: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=199, help="Inclusive final frame")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--rider-pair-radius", type=int, default=24)
    parser.add_argument("--track-iou", type=float, default=0.08)
    parser.add_argument("--track-max-age", type=int, default=2)
    parser.add_argument("--flow-warp-dilation", type=int, default=3)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(np.asarray(values, dtype=np.float64), 95)),
    }


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot compute bbox for an empty mask")
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    if intersection == 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(intersection / max(union, 1))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def connected_masks(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(binary, dtype=np.uint8), connectivity=8
    )
    components: list[np.ndarray] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        components.append(np.ascontiguousarray(labels == index))
    components.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return components


def component_confidence(confidence: np.ndarray, mask: np.ndarray) -> float:
    values = confidence[mask]
    if values.size == 0:
        return 0.0
    return float(np.median(values.astype(np.float32, copy=False)))


def masks_are_near(a: np.ndarray, b: np.ndarray, radius: int) -> bool:
    if radius <= 0:
        return bool(np.logical_and(a, b).any())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    expanded = cv2.dilate(a.astype(np.uint8), kernel, iterations=1) > 0
    return bool(np.logical_and(expanded, b).any())


def pairing_score(rider: np.ndarray, vehicle: np.ndarray, radius: int) -> float:
    if not masks_are_near(rider, vehicle, radius):
        return float("-inf")
    rider_box = bbox_from_mask(rider)
    vehicle_box = bbox_from_mask(vehicle)
    rider_center = np.asarray(
        [(rider_box[0] + rider_box[2]) * 0.5, (rider_box[1] + rider_box[3]) * 0.5]
    )
    vehicle_center = np.asarray(
        [(vehicle_box[0] + vehicle_box[2]) * 0.5, (vehicle_box[1] + vehicle_box[3]) * 0.5]
    )
    scale = max(float(vehicle_box[2] - vehicle_box[0]), float(vehicle_box[3] - vehicle_box[1]), 1.0)
    return -float(np.linalg.norm(rider_center - vehicle_center) / scale)


def make_detection(
    category: str,
    mask: np.ndarray,
    confidence: np.ndarray,
    source_categories: list[str],
    source_label_ids: list[int],
) -> Detection:
    return Detection(
        category=category,
        confidence=component_confidence(confidence, mask),
        bbox=bbox_from_mask(mask),
        mask=np.ascontiguousarray(mask, dtype=np.bool_),
        source_categories=source_categories,
        source_label_ids=source_label_ids,
    )


def extract_pseudo_instances(
    labels: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float,
    min_area: int,
    rider_pair_radius: int,
) -> tuple[list[Detection], dict[str, int]]:
    if labels.shape != confidence.shape:
        raise ValueError("labels/confidence shape mismatch")
    valid_confidence = np.isfinite(confidence) & (confidence >= min_confidence)
    detections: list[Detection] = []

    for label_id, category in DIRECT_CATEGORIES.items():
        for mask in connected_masks((labels == label_id) & valid_confidence, min_area):
            detections.append(
                make_detection(category, mask, confidence, [category], [label_id])
            )

    riders = connected_masks(
        (labels == CITYSCAPES_IDS["rider"]) & valid_confidence, min_area
    )
    bicycles = connected_masks(
        (labels == CITYSCAPES_IDS["bicycle"]) & valid_confidence, min_area
    )
    motorcycles = connected_masks(
        (labels == CITYSCAPES_IDS["motorcycle"]) & valid_confidence, min_area
    )
    used_bicycles: set[int] = set()
    used_motorcycles: set[int] = set()

    pair_candidates: list[tuple[float, int, str, int]] = []
    for rider_index, rider in enumerate(riders):
        for index, motorcycle in enumerate(motorcycles):
            score = pairing_score(rider, motorcycle, rider_pair_radius)
            if np.isfinite(score):
                pair_candidates.append((score + 0.01, rider_index, "motorcycle", index))
        for index, bicycle in enumerate(bicycles):
            score = pairing_score(rider, bicycle, rider_pair_radius)
            if np.isfinite(score):
                pair_candidates.append((score, rider_index, "bicycle", index))

    rider_pairs: dict[int, tuple[str, int]] = {}
    for _, rider_index, vehicle_type, vehicle_index in sorted(pair_candidates, reverse=True):
        if rider_index in rider_pairs:
            continue
        if vehicle_type == "motorcycle" and vehicle_index in used_motorcycles:
            continue
        if vehicle_type == "bicycle" and vehicle_index in used_bicycles:
            continue
        rider_pairs[rider_index] = (vehicle_type, vehicle_index)
        if vehicle_type == "motorcycle":
            used_motorcycles.add(vehicle_index)
        else:
            used_bicycles.add(vehicle_index)

    for rider_index, rider in enumerate(riders):
        pair = rider_pairs.get(rider_index)
        if pair is None:
            detections.append(
                make_detection(
                    "cyclist",
                    rider,
                    confidence,
                    ["rider"],
                    [CITYSCAPES_IDS["rider"]],
                )
            )
            continue
        vehicle_type, vehicle_index = pair
        if vehicle_type == "motorcycle":
            mask = np.logical_or(rider, motorcycles[vehicle_index])
            detections.append(
                make_detection(
                    "motorcycle",
                    mask,
                    confidence,
                    ["rider", "motorcycle"],
                    [CITYSCAPES_IDS["rider"], CITYSCAPES_IDS["motorcycle"]],
                )
            )
        else:
            mask = np.logical_or(rider, bicycles[vehicle_index])
            detections.append(
                make_detection(
                    "cyclist",
                    mask,
                    confidence,
                    ["rider", "bicycle"],
                    [CITYSCAPES_IDS["rider"], CITYSCAPES_IDS["bicycle"]],
                )
            )

    for index, motorcycle in enumerate(motorcycles):
        if index not in used_motorcycles:
            detections.append(
                make_detection(
                    "motorcycle",
                    motorcycle,
                    confidence,
                    ["motorcycle"],
                    [CITYSCAPES_IDS["motorcycle"]],
                )
            )

    ignored = {
        "unpaired_bicycle_components": len(bicycles) - len(used_bicycles),
        "train_pixels": int(((labels == CITYSCAPES_IDS["train"]) & valid_confidence).sum()),
        "low_confidence_dynamic_pixels": int(
            np.isin(labels, list(CITYSCAPES_IDS.values())).astype(np.bool_)[~valid_confidence].sum()
        ),
    }
    return detections, ignored


def forward_warp_mask(mask: np.ndarray, flow: np.ndarray, dilation: int) -> np.ndarray:
    if flow.shape != (*mask.shape, 2):
        raise ValueError("flow/mask shape mismatch")
    ys, xs = np.nonzero(mask)
    warped = np.zeros_like(mask, dtype=np.uint8)
    if xs.size:
        destination_x = np.rint(xs + flow[ys, xs, 0]).astype(np.int64)
        destination_y = np.rint(ys + flow[ys, xs, 1]).astype(np.int64)
        inside = (
            (destination_x >= 0)
            & (destination_x < mask.shape[1])
            & (destination_y >= 0)
            & (destination_y < mask.shape[0])
        )
        warped[destination_y[inside], destination_x[inside]] = 1
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        warped = cv2.dilate(warped, kernel, iterations=1)
    return np.ascontiguousarray(warped > 0)


class FlowWarpTracker:
    def __init__(self, threshold: float, max_age: int, dilation: int) -> None:
        self.threshold = threshold
        self.max_age = max_age
        self.dilation = dilation
        self.next_track_id = 0
        self.active: dict[int, tuple[int, Detection]] = {}

    def update(
        self, frame_index: int, detections: list[Detection], previous_flow: np.ndarray | None
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, (last_frame, previous) in self.active.items():
                age = frame_index - last_frame
                if age > self.max_age or previous.category != detection.category:
                    continue
                if age == 1 and previous_flow is not None:
                    reference_mask = forward_warp_mask(previous.mask, previous_flow, self.dilation)
                    reference_bbox = bbox_from_mask(reference_mask) if reference_mask.any() else previous.bbox
                else:
                    reference_mask = previous.mask
                    reference_bbox = previous.bbox
                score = 0.75 * mask_iou(reference_mask, detection.mask) + 0.25 * bbox_iou(
                    reference_bbox, detection.bbox
                )
                if score >= self.threshold:
                    candidates.append((score, detection_index, track_id))

        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in assigned_detections or track_id in assigned_tracks:
                continue
            detections[detection_index].track_id = track_id
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)

        for detection in detections:
            if detection.track_id < 0:
                detection.track_id = self.next_track_id
                self.next_track_id += 1

        self.active = {
            track_id: item
            for track_id, item in self.active.items()
            if frame_index - item[0] <= self.max_age
        }
        for detection in detections:
            self.active[detection.track_id] = (frame_index, detection)


def load_previous_flow(flow_dir: Path | None, frame_index: int) -> np.ndarray | None:
    if flow_dir is None or frame_index <= 0:
        return None
    path = flow_dir / f"{frame_index - 1:06d}_{frame_index:06d}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return data["observed_flow"].astype(np.float32, copy=False)


def make_preview(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    overlay = image.copy()
    for detection in detections:
        color = np.asarray(COLORS[detection.category], dtype=np.float32)
        overlay[detection.mask] = (
            0.55 * overlay[detection.mask].astype(np.float32) + 0.45 * color
        ).astype(np.uint8)
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS[detection.category], 1)
        if (x2 - x1) * (y2 - y1) >= 800:
            cv2.putText(
                overlay,
                f"{detection.category[:4]}#{detection.track_id}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLORS[detection.category],
                1,
                cv2.LINE_AA,
            )
    return overlay


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be >= --start")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must lie in [0,1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    tracker = FlowWarpTracker(args.track_iou, args.track_max_age, args.flow_warp_dilation)
    rows: list[dict[str, Any]] = []
    frame_times: list[float] = []
    total_instances = 0
    ignored_totals = {
        "unpaired_bicycle_components": 0,
        "train_pixels": 0,
        "low_confidence_dynamic_pixels": 0,
    }
    category_counts = {category: 0 for category in sorted(CANONICAL_CATEGORIES)}
    flow_frames_used = 0

    for frame_index in range(args.start, args.end + 1):
        started = time.perf_counter()
        frame_id = f"{frame_index:06d}"
        image_path = args.image_dir / f"{frame_id}.png"
        label_path = args.semantic_dir / "labels" / f"{frame_id}_semantic.npy"
        confidence_path = args.semantic_dir / "confidence" / f"{frame_id}_confidence.npy"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode image: {image_path}")
        labels = np.load(label_path, allow_pickle=False)
        confidence = np.load(confidence_path, allow_pickle=False).astype(np.float32)
        if labels.dtype != np.uint8 or labels.shape != image.shape[:2]:
            raise RuntimeError(f"Bad labels at {frame_id}: {labels.dtype} {labels.shape}")
        if confidence.shape != image.shape[:2]:
            raise RuntimeError(f"Bad confidence shape at {frame_id}: {confidence.shape}")

        detections, ignored = extract_pseudo_instances(
            labels,
            confidence,
            args.min_confidence,
            args.min_area,
            args.rider_pair_radius,
        )
        previous_flow = load_previous_flow(args.flow_dir, frame_index)
        if previous_flow is not None:
            flow_frames_used += 1
        tracker.update(frame_index, detections, previous_flow)
        detections.sort(key=lambda item: (item.category, item.track_id))

        masks = (
            np.stack([item.mask for item in detections]).astype(np.bool_, copy=False)
            if detections
            else np.zeros((0, *image.shape[:2]), dtype=np.bool_)
        )
        np.savez_compressed(args.output_dir / f"{frame_id}_masks.npz", masks=masks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metadata = {
            "frame_id": frame_id,
            "image_path": str(image_path),
            "image_shape": list(image.shape[:2]),
            "instances": [
                {
                    "instance_id": index,
                    "track_id": item.track_id,
                    "category": item.category,
                    "confidence": item.confidence,
                    "bbox": [float(value) for value in item.bbox],
                    "mask_index": index,
                    "mask_area": int(item.mask.sum()),
                    "source_categories": item.source_categories,
                    "source_label_ids": item.source_label_ids,
                    "source_model": "Mask2Former Cityscapes semantic",
                    "grouping": "semantic_connected_component",
                }
                for index, item in enumerate(detections)
            ],
            "ignored": ignored,
            "timing_ms": {"end_to_end": elapsed_ms, "inference": 0.0, "adapter": elapsed_ms},
            "flow_warp_used": previous_flow is not None,
        }
        (args.output_dir / f"{frame_id}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if args.preview_every > 0 and (frame_index - args.start) % args.preview_every == 0:
            cv2.imwrite(str(preview_dir / f"{frame_id}.png"), make_preview(image, detections))

        total_instances += len(detections)
        frame_times.append(elapsed_ms)
        for item in detections:
            category_counts[item.category] += 1
        for key, value in ignored.items():
            ignored_totals[key] += value
        rows.append(
            {
                "frame_id": frame_id,
                "instances": len(detections),
                "unique_tracks_so_far": tracker.next_track_id,
                "flow_warp_used": previous_flow is not None,
                "adapter_ms": elapsed_ms,
                **{f"instances_{key}": sum(item.category == key for item in detections) for key in category_counts},
            }
        )
        print(
            f"[{frame_index - args.start + 1:04d}/{args.end - args.start + 1:04d}] "
            f"{frame_id}: instances={len(detections)} tracks={tracker.next_track_id} "
            f"flow={'yes' if previous_flow is not None else 'no'} {elapsed_ms:.1f}ms",
            flush=True,
        )

    with (args.output_dir / "per_frame_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    semantic_manifest = args.semantic_dir / "SHA256SUMS.txt"
    summary = {
        "status": "GENERATED_PSEUDO_INSTANCES",
        "scientific_status": "UNVALIDATED_REQUIRES_GT_AND_VISUAL_REVIEW",
        "frames": args.end - args.start + 1,
        "frame_range": [args.start, args.end],
        "source": {
            "type": "Mask2Former dense Cityscapes-19 semantics",
            "semantic_dir": str(args.semantic_dir),
            "semantic_manifest_sha256": sha256_file(semantic_manifest) if semantic_manifest.is_file() else None,
            "flow_dir": str(args.flow_dir) if args.flow_dir else None,
            "flow_frames_used": flow_frames_used,
        },
        "grouping": {
            "method": "confidence-gated 8-connected components",
            "min_confidence": args.min_confidence,
            "min_area": args.min_area,
            "rider_pair_radius": args.rider_pair_radius,
            "unpaired_bicycle_policy": "ignore",
            "train_policy": "ignore_not_in_frozen_stage2_categories",
        },
        "tracker": {
            "name": "RAFT-forward-warped-mask-IoU greedy tracker",
            "track_iou": args.track_iou,
            "max_age": args.track_max_age,
            "flow_warp_dilation": args.flow_warp_dilation,
        },
        "total_instances": total_instances,
        "unique_tracks": tracker.next_track_id,
        "category_instances": category_counts,
        "ignored": ignored_totals,
        "inference_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0},
        "adapter_ms": safe_stats(frame_times),
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
