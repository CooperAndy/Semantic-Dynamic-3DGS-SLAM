#!/usr/bin/env python3
"""Generate tracked Cityscapes thing instances with Mask2Former only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from adapt_mask2former_to_instances import (
    CANONICAL_CATEGORIES,
    Detection,
    FlowWarpTracker,
    bbox_from_mask,
    load_previous_flow,
    make_preview,
    mask_iou,
    masks_are_near,
    pairing_score,
    sha256_file,
)


DIRECT_CATEGORIES = {"car", "truck", "bus", "person"}
MODEL_THING_CATEGORIES = DIRECT_CATEGORIES | {"rider", "motorcycle", "bicycle", "train"}
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-source-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dense-semantic-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=199)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--overlap-threshold", type=float, default=0.80)
    parser.add_argument("--dedupe-mask-iou", type=float, default=0.30)
    parser.add_argument("--dedupe-containment", type=float, default=0.70)
    parser.add_argument("--cross-category-mask-iou", type=float, default=0.80)
    parser.add_argument("--cross-category-containment", type=float, default=0.95)
    parser.add_argument("--min-dense-agreement", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--rider-pair-radius", type=int, default=24)
    parser.add_argument("--track-iou", type=float, default=0.08)
    parser.add_argument("--track-max-age", type=int, default=2)
    parser.add_argument("--flow-warp-dilation", type=int, default=3)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(np.asarray(values, dtype=np.float64), 95)),
    }


def detection_from_mask(
    category: str,
    score: float,
    mask: np.ndarray,
    source_categories: list[str],
    source_label_ids: list[int],
) -> Detection:
    return Detection(
        category=category,
        confidence=float(score),
        bbox=bbox_from_mask(mask),
        mask=np.ascontiguousarray(mask, dtype=np.bool_),
        source_categories=source_categories,
        source_label_ids=source_label_ids,
    )


def canonicalize_queries(
    raw: list[Detection], pair_radius: int
) -> tuple[list[Detection], dict[str, int]]:
    direct = [item for item in raw if item.category in DIRECT_CATEGORIES]
    riders = [item for item in raw if item.category == "rider"]
    bicycles = [item for item in raw if item.category == "bicycle"]
    motorcycles = [item for item in raw if item.category == "motorcycle"]
    trains = [item for item in raw if item.category == "train"]
    used_bicycles: set[int] = set()
    used_motorcycles: set[int] = set()

    pair_candidates: list[tuple[float, int, str, int]] = []
    for rider_index, rider in enumerate(riders):
        for index, motorcycle in enumerate(motorcycles):
            score = pairing_score(rider.mask, motorcycle.mask, pair_radius)
            if np.isfinite(score):
                pair_candidates.append((score + 0.01, rider_index, "motorcycle", index))
        for index, bicycle in enumerate(bicycles):
            score = pairing_score(rider.mask, bicycle.mask, pair_radius)
            if np.isfinite(score):
                pair_candidates.append((score, rider_index, "bicycle", index))

    rider_pairs: dict[int, tuple[str, int]] = {}
    for _, rider_index, vehicle_type, vehicle_index in sorted(pair_candidates, reverse=True):
        if rider_index in rider_pairs:
            continue
        used = used_motorcycles if vehicle_type == "motorcycle" else used_bicycles
        if vehicle_index in used:
            continue
        rider_pairs[rider_index] = (vehicle_type, vehicle_index)
        used.add(vehicle_index)

    output = list(direct)
    for rider_index, rider in enumerate(riders):
        pair = rider_pairs.get(rider_index)
        if pair is None:
            output.append(
                detection_from_mask(
                    "cyclist",
                    rider.confidence,
                    rider.mask,
                    ["rider"],
                    rider.source_label_ids,
                )
            )
            continue
        vehicle_type, vehicle_index = pair
        vehicle = motorcycles[vehicle_index] if vehicle_type == "motorcycle" else bicycles[vehicle_index]
        category = "motorcycle" if vehicle_type == "motorcycle" else "cyclist"
        output.append(
            detection_from_mask(
                category,
                min(rider.confidence, vehicle.confidence),
                np.logical_or(rider.mask, vehicle.mask),
                ["rider", vehicle_type],
                rider.source_label_ids + vehicle.source_label_ids,
            )
        )

    for index, motorcycle in enumerate(motorcycles):
        if index not in used_motorcycles:
            output.append(motorcycle)

    ignored = {
        "unpaired_bicycle_queries": len(bicycles) - len(used_bicycles),
        "train_queries": len(trains),
    }
    return output, ignored


def dense_semantic_ids(detection: Detection) -> set[int]:
    try:
        return DENSE_SEMANTIC_IDS[detection.category]
    except KeyError as error:
        raise ValueError(f"No dense semantic mapping for {detection.category!r}") from error


def dense_agreement(labels: np.ndarray, detection: Detection) -> float:
    if labels.shape != detection.mask.shape:
        raise ValueError("dense labels/detection mask shape mismatch")
    ids = dense_semantic_ids(detection)
    return float(np.isin(labels[detection.mask], list(ids)).mean())


def mask_containment(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    if intersection == 0:
        return 0.0
    smaller = min(a.sum(dtype=np.int64), b.sum(dtype=np.int64))
    return float(intersection / max(int(smaller), 1))


def filter_and_deduplicate(
    detections: list[Detection],
    dense_labels: np.ndarray,
    min_dense_agreement: float,
    dedupe_mask_iou: float,
    dedupe_containment: float,
    cross_category_mask_iou: float,
    cross_category_containment: float,
) -> tuple[list[Detection], dict[str, int]]:
    agreements = [dense_agreement(dense_labels, item) for item in detections]
    order = sorted(
        range(len(detections)),
        key=lambda index: (
            agreements[index],
            detections[index].confidence,
            int(detections[index].mask.sum()),
            -index,
        ),
        reverse=True,
    )
    kept: list[Detection] = []
    low_agreement = 0
    duplicate = 0
    cross_category_duplicate = 0
    for index in order:
        detection = detections[index]
        if agreements[index] < min_dense_agreement:
            low_agreement += 1
            continue
        duplicate_kind = None
        for previous in kept:
            same_category = detection.category == previous.category
            iou_limit = dedupe_mask_iou if same_category else cross_category_mask_iou
            containment_limit = (
                dedupe_containment if same_category else cross_category_containment
            )
            if (
                mask_iou(detection.mask, previous.mask) >= iou_limit
                or mask_containment(detection.mask, previous.mask) >= containment_limit
            ):
                duplicate_kind = "same" if same_category else "cross"
                break
        if duplicate_kind is not None:
            if duplicate_kind == "same":
                duplicate += 1
            else:
                cross_category_duplicate += 1
            continue
        kept.append(detection)
    return kept, {
        "low_dense_agreement_queries": low_agreement,
        "duplicate_queries": duplicate,
        "cross_category_duplicate_queries": cross_category_duplicate,
    }


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be >= --start")
    for name in (
        "score_threshold",
        "mask_threshold",
        "overlap_threshold",
        "dedupe_mask_iou",
        "dedupe_containment",
        "cross_category_mask_iou",
        "cross_category_containment",
        "min_dense_agreement",
        "track_iou",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)

    load_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(
        args.model_dir, local_files_only=True, use_fast=False
    )
    model, loading_info = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_dir, local_files_only=True, output_loading_info=True
    )
    loading_failures = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(loading_failures.values()):
        raise RuntimeError(f"Checkpoint did not load strictly: {loading_failures}")
    model = model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    id2label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    (args.output_dir / "id2label.json").write_text(
        json.dumps(id2label, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tracker = FlowWarpTracker(args.track_iou, args.track_max_age, args.flow_warp_dilation)
    rows: list[dict[str, Any]] = []
    inference_times: list[float] = []
    end_to_end_times: list[float] = []
    category_counts = {category: 0 for category in sorted(CANONICAL_CATEGORIES)}
    ignored_totals = {
        "unpaired_bicycle_queries": 0,
        "train_queries": 0,
        "low_dense_agreement_queries": 0,
        "duplicate_queries": 0,
        "cross_category_duplicate_queries": 0,
    }
    agreement_values: list[float] = []
    total_instances = 0
    raw_instance_queries = 0
    flow_frames_used = 0

    for frame_index in range(args.start, args.end + 1):
        from resource_guard import wait_for_headroom
        wait_for_headroom("instances", frame_index)
        frame_id = f"{frame_index:06d}"
        image_path = args.image_dir / f"{frame_id}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = bgr.shape[:2]
        dense_labels = np.load(
            args.dense_semantic_dir / "labels" / f"{frame_id}_semantic.npy",
            allow_pickle=False,
        )
        if dense_labels.dtype != np.uint8 or dense_labels.shape != (height, width):
            raise RuntimeError(
                f"Bad dense semantic labels for {frame_id}: "
                f"{dense_labels.dtype} {dense_labels.shape}"
            )

        frame_started = time.perf_counter()
        inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            overlap_mask_area_threshold=args.overlap_threshold,
            target_sizes=[(height, width)],
            return_binary_maps=True,
        )[0]
        if processed["segmentation"] is None:
            binary_masks = np.zeros((0, height, width), dtype=np.bool_)
        else:
            binary_masks = processed["segmentation"].detach().cpu().numpy() > 0.5

        raw: list[Detection] = []
        for info in processed["segments_info"]:
            label_id = int(info["label_id"])
            category = id2label.get(label_id, f"label_{label_id}")
            if category not in MODEL_THING_CATEGORIES:
                continue
            mask_index = int(info["id"])
            mask = np.ascontiguousarray(binary_masks[mask_index], dtype=np.bool_)
            if int(mask.sum()) < args.min_area:
                continue
            raw.append(
                detection_from_mask(
                    category,
                    float(info["score"]),
                    mask,
                    [category],
                    [label_id],
                )
            )

        detections, ignored = canonicalize_queries(raw, args.rider_pair_radius)
        raw_instance_queries += len(detections)
        detections, dedupe_ignored = filter_and_deduplicate(
            detections,
            dense_labels,
            args.min_dense_agreement,
            args.dedupe_mask_iou,
            args.dedupe_containment,
            args.cross_category_mask_iou,
            args.cross_category_containment,
        )
        ignored.update(dedupe_ignored)
        previous_flow = load_previous_flow(args.flow_dir, frame_index)
        if previous_flow is not None:
            flow_frames_used += 1
        tracker.update(frame_index, detections, previous_flow)
        detections.sort(key=lambda item: (item.category, item.track_id))
        agreements = [dense_agreement(dense_labels, item) for item in detections]
        agreement_values.extend(agreements)

        masks = (
            np.stack([item.mask for item in detections]).astype(np.bool_, copy=False)
            if detections
            else np.zeros((0, height, width), dtype=np.bool_)
        )
        np.savez_compressed(args.output_dir / f"{frame_id}_masks.npz", masks=masks)
        end_to_end_ms = (time.perf_counter() - frame_started) * 1000.0
        metadata = {
            "frame_id": frame_id,
            "image_path": str(image_path),
            "image_shape": [height, width],
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
                    "dense_semantic_ids": sorted(dense_semantic_ids(item)),
                    "dense_semantic_agreement": agreements[index],
                    "source_model": "Mask2Former Cityscapes instance",
                }
                for index, item in enumerate(detections)
            ],
            "ignored": ignored,
            "timing_ms": {
                "end_to_end": end_to_end_ms,
                "preprocess_postprocess": end_to_end_ms - inference_ms,
                "inference": inference_ms,
            },
            "flow_warp_used": previous_flow is not None,
        }
        (args.output_dir / f"{frame_id}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if args.preview_every > 0 and (frame_index - args.start) % args.preview_every == 0:
            cv2.imwrite(str(preview_dir / f"{frame_id}.png"), make_preview(bgr, detections))

        total_instances += len(detections)
        inference_times.append(inference_ms)
        end_to_end_times.append(end_to_end_ms)
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
                "inference_ms": inference_ms,
                "end_to_end_ms": end_to_end_ms,
                "mean_dense_agreement": float(np.mean(agreements)) if agreements else None,
                "raw_instance_queries": len(detections)
                + ignored["low_dense_agreement_queries"]
                + ignored["duplicate_queries"]
                + ignored["cross_category_duplicate_queries"],
                "filtered_low_dense_agreement": ignored["low_dense_agreement_queries"],
                "suppressed_duplicates": ignored["duplicate_queries"],
                "suppressed_cross_category_duplicates": ignored[
                    "cross_category_duplicate_queries"
                ],
            }
        )
        print(
            f"[{frame_index - args.start + 1:04d}/{args.end - args.start + 1:04d}] "
            f"{frame_id}: instances={len(detections)} tracks={tracker.next_track_id} "
            f"infer={inference_ms:.1f}ms total={end_to_end_ms:.1f}ms",
            flush=True,
        )

    with (args.output_dir / "per_frame_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    weight = args.model_dir / "model.safetensors"
    if not weight.is_file():
        weight = args.model_dir / "pytorch_model.bin"
    warmup = min(args.warmup_frames, len(inference_times))
    measured_inference = inference_times[warmup:] or inference_times
    measured_end_to_end = end_to_end_times[warmup:] or end_to_end_times
    dense_manifest = args.dense_semantic_dir / "SHA256SUMS.txt"
    summary = {
        "status": "GENERATED_MASK2FORMER_INSTANCES",
        "scientific_status": "UNVALIDATED_REQUIRES_GT_AND_VISUAL_REVIEW",
        "frames": args.end - args.start + 1,
        "frame_range": [args.start, args.end],
        "model": {
            "source_id": args.model_source_id,
            "revision": args.model_revision,
            "local_dir": str(args.model_dir),
            "weight": str(weight),
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": sha256_file(weight),
            "id2label": id2label,
            "loading_info": loading_failures,
        },
        "model_sha256": sha256_file(weight),
        "dense_semantic": {
            "dir": str(args.dense_semantic_dir),
            "manifest_sha256": sha256_file(dense_manifest) if dense_manifest.is_file() else None,
            "instance_mask_agreement": safe_stats(agreement_values),
        },
        "postprocess": {
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "overlap_threshold": args.overlap_threshold,
            "dedupe_mask_iou": args.dedupe_mask_iou,
            "dedupe_containment": args.dedupe_containment,
            "cross_category_mask_iou": args.cross_category_mask_iou,
            "cross_category_containment": args.cross_category_containment,
            "min_dense_agreement": args.min_dense_agreement,
            "dedupe_ranking": "dense_agreement_then_model_confidence_then_area",
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
            "flow_dir": str(args.flow_dir) if args.flow_dir else None,
            "flow_frames_used": flow_frames_used,
        },
        "total_instances": total_instances,
        "raw_instance_queries": raw_instance_queries,
        "unique_tracks": tracker.next_track_id,
        "category_instances": category_counts,
        "ignored": ignored_totals,
        "model_load_seconds": model_load_seconds,
        "warmup_frames_excluded": warmup,
        "inference_ms": safe_stats(measured_inference),
        "end_to_end_ms": safe_stats(measured_end_to_end),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
