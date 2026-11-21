#!/usr/bin/env python3
"""Audit same-class duplicate masks and simulate deterministic mask NMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DENSE_IDS = {
    "person": {11},
    "car": {13},
    "truck": {14},
    "bus": {15},
    "cyclist": {12, 18},
    "motorcycle": {12, 17},
}


def mask_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum(dtype=np.int64))
    if intersection == 0:
        return 0.0, 0.0
    area_a = int(a.sum(dtype=np.int64))
    area_b = int(b.sum(dtype=np.int64))
    union = area_a + area_b - intersection
    return intersection / max(union, 1), intersection / max(min(area_a, area_b), 1)


def keep_indices(
    instances: list[dict],
    masks: np.ndarray,
    dense: np.ndarray,
    iou_threshold: float,
    containment_threshold: float,
    min_dense_agreement: float = 0.0,
) -> list[int]:
    agreements = [
        float(np.isin(dense[masks[index]], list(DENSE_IDS[instance["category"]])).mean())
        for index, instance in enumerate(instances)
    ]
    order = sorted(
        (index for index in range(len(instances)) if agreements[index] >= min_dense_agreement),
        key=lambda index: (
            agreements[index],
            float(instances[index]["confidence"]),
            int(instances[index]["mask_area"]),
            -index,
        ),
        reverse=True,
    )
    kept: list[int] = []
    for index in order:
        duplicate = False
        for kept_index in kept:
            if instances[index]["category"] != instances[kept_index]["category"]:
                continue
            iou, containment = mask_metrics(masks[index], masks[kept_index])
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=19)
    args = parser.parse_args()

    frames: list[tuple[str, list[dict], np.ndarray, np.ndarray]] = []
    pair_ious: list[float] = []
    pair_containments: list[float] = []
    raw_agreements: list[float] = []

    for frame_index in range(args.start, args.end + 1):
        frame_id = f"{frame_index:06d}"
        metadata = json.loads((args.instance_dir / f"{frame_id}.json").read_text())
        masks = np.load(args.instance_dir / f"{frame_id}_masks.npz", allow_pickle=False)["masks"]
        dense = np.load(
            args.dense_dir / "labels" / f"{frame_id}_semantic.npy", allow_pickle=False
        )
        instances = metadata["instances"]
        for left in range(len(instances)):
            ids = DENSE_IDS[instances[left]["category"]]
            raw_agreements.append(float(np.isin(dense[masks[left]], list(ids)).mean()))
            for right in range(left + 1, len(instances)):
                if instances[left]["category"] != instances[right]["category"]:
                    continue
                iou, containment = mask_metrics(masks[left], masks[right])
                if iou > 0:
                    pair_ious.append(iou)
                    pair_containments.append(containment)
        frames.append((frame_id, instances, masks, dense))

    def percentiles(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "p50": float(np.percentile(array, 50)) if array.size else 0.0,
            "p75": float(np.percentile(array, 75)) if array.size else 0.0,
            "p90": float(np.percentile(array, 90)) if array.size else 0.0,
            "p95": float(np.percentile(array, 95)) if array.size else 0.0,
            "max": float(array.max()) if array.size else 0.0,
        }

    result: dict[str, object] = {
        "frames": len(frames),
        "raw_instances": sum(len(item[1]) for item in frames),
        "overlapping_same_class_pair_iou": percentiles(pair_ious),
        "overlapping_same_class_pair_containment": percentiles(pair_containments),
        "corrected_raw_dense_agreement": {
            "mean": float(np.mean(raw_agreements)),
            "median": float(np.median(raw_agreements)),
        },
        "nms_grid": [],
    }

    for iou_threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        for containment_threshold in (0.70, 0.80, 0.90):
            kept_total = 0
            agreements: list[float] = []
            per_frame: list[int] = []
            for _, instances, masks, dense in frames:
                kept = keep_indices(
                    instances, masks, dense, iou_threshold, containment_threshold
                )
                kept_total += len(kept)
                per_frame.append(len(kept))
                for index in kept:
                    ids = DENSE_IDS[instances[index]["category"]]
                    agreements.append(float(np.isin(dense[masks[index]], list(ids)).mean()))
            result["nms_grid"].append(
                {
                    "iou": iou_threshold,
                    "containment": containment_threshold,
                    "kept": kept_total,
                    "removed": result["raw_instances"] - kept_total,
                    "per_frame_range": [min(per_frame), max(per_frame)],
                    "agreement_mean": float(np.mean(agreements)),
                    "agreement_median": float(np.median(agreements)),
                }
            )

    result["agreement_filter_at_iou_0_30_containment_0_70"] = []
    for min_agreement in (0.0, 0.10, 0.25, 0.50, 0.75):
        kept_total = 0
        agreements = []
        category_counts = {category: 0 for category in DENSE_IDS}
        per_frame = []
        for _, instances, masks, dense in frames:
            kept = keep_indices(instances, masks, dense, 0.30, 0.70, min_agreement)
            kept_total += len(kept)
            per_frame.append(len(kept))
            for index in kept:
                category = instances[index]["category"]
                category_counts[category] += 1
                ids = DENSE_IDS[category]
                agreements.append(float(np.isin(dense[masks[index]], list(ids)).mean()))
        result["agreement_filter_at_iou_0_30_containment_0_70"].append(
            {
                "min_agreement": min_agreement,
                "kept": kept_total,
                "removed": result["raw_instances"] - kept_total,
                "per_frame_range": [min(per_frame), max(per_frame)],
                "category_counts": category_counts,
                "agreement_mean": float(np.mean(agreements)),
                "agreement_median": float(np.median(agreements)),
            }
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
