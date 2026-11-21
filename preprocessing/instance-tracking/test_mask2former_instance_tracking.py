#!/usr/bin/env python3
"""Deterministic tests for Mask2Former instance filtering and ID mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from adapt_mask2former_to_instances import Detection, bbox_from_mask  # noqa: E402
from run_mask2former_instance_tracking import (  # noqa: E402
    dense_agreement,
    dense_semantic_ids,
    filter_and_deduplicate,
)


def detection(category: str, confidence: float, mask: np.ndarray, source_id: int) -> Detection:
    return Detection(
        category=category,
        confidence=confidence,
        bbox=bbox_from_mask(mask),
        mask=np.ascontiguousarray(mask, dtype=np.bool_),
        source_categories=[category],
        source_label_ids=[source_id],
    )


def main() -> None:
    shape = (40, 60)
    dense = np.zeros(shape, dtype=np.uint8)
    dense[5:25, 5:25] = 13
    dense[5:25, 30:50] = 13
    dense[28:38, 5:12] = 11

    car_mask = np.zeros(shape, dtype=np.bool_)
    car_mask[5:25, 5:25] = True
    duplicate_car_mask = np.zeros(shape, dtype=np.bool_)
    duplicate_car_mask[6:24, 6:24] = True
    adjacent_car_mask = np.zeros(shape, dtype=np.bool_)
    adjacent_car_mask[5:25, 30:50] = True
    false_person_mask = np.zeros(shape, dtype=np.bool_)
    false_person_mask[5:15, 5:15] = True

    car = detection("car", 0.90, car_mask, 2)
    duplicate_car = detection("car", 0.95, duplicate_car_mask, 2)
    adjacent_car = detection("car", 0.80, adjacent_car_mask, 2)
    false_person = detection("person", 0.99, false_person_mask, 0)

    assert dense_semantic_ids(car) == {13}
    assert dense_agreement(dense, car) == 1.0
    assert dense_agreement(dense, false_person) == 0.0

    kept, ignored = filter_and_deduplicate(
        [car, duplicate_car, adjacent_car, false_person],
        dense,
        min_dense_agreement=0.10,
        dedupe_mask_iou=0.30,
        dedupe_containment=0.70,
    )
    assert len(kept) == 2, [(item.category, item.confidence) for item in kept]
    assert all(item.category == "car" for item in kept)
    assert ignored == {"low_dense_agreement_queries": 1, "duplicate_queries": 1}
    assert sorted(item.confidence for item in kept) == [0.80, 0.95]

    cyclist_mask = np.zeros(shape, dtype=np.bool_)
    cyclist_mask[28:38, 5:12] = True
    cyclist = detection("cyclist", 0.75, cyclist_mask, 1)
    assert dense_semantic_ids(cyclist) == {12, 18}

    print("PASS: dense ID mapping, agreement filtering, and deterministic mask NMS")


if __name__ == "__main__":
    main()
