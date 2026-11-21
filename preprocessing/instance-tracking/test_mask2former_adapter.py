#!/usr/bin/env python3
"""Deterministic unit tests for the Mask2Former pseudo-instance adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from adapt_mask2former_to_instances import (  # noqa: E402
    FlowWarpTracker,
    extract_pseudo_instances,
)


def main() -> None:
    shape = (80, 120)
    labels = np.zeros(shape, dtype=np.uint8)
    confidence = np.full(shape, 0.95, dtype=np.float32)
    labels[20:45, 15:45] = 13  # car
    labels[20:35, 70:78] = 12  # rider
    labels[34:50, 65:88] = 18  # bicycle, paired with rider
    labels[60:66, 100:110] = 18  # unpaired bicycle, must be ignored

    detections, ignored = extract_pseudo_instances(labels, confidence, 0.5, 16, 8)
    categories = sorted(item.category for item in detections)
    assert categories == ["car", "cyclist"], categories
    assert ignored["unpaired_bicycle_components"] == 1, ignored
    cyclist = next(item for item in detections if item.category == "cyclist")
    assert cyclist.source_categories == ["rider", "bicycle"]

    tracker = FlowWarpTracker(threshold=0.08, max_age=2, dilation=1)
    tracker.update(0, detections, previous_flow=None)
    original_tracks = {item.category: item.track_id for item in detections}

    labels_next = np.zeros(shape, dtype=np.uint8)
    labels_next[20:45, 20:50] = 13
    labels_next[20:35, 75:83] = 12
    labels_next[34:50, 70:93] = 18
    detections_next, _ = extract_pseudo_instances(labels_next, confidence, 0.5, 16, 8)
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[..., 0] = 5.0
    tracker.update(1, detections_next, previous_flow=flow)
    next_tracks = {item.category: item.track_id for item in detections_next}
    assert next_tracks == original_tracks, (original_tracks, next_tracks)
    assert all(item.category in {"car", "cyclist"} for item in detections_next)
    print("PASS: extraction, bicycle policy, rider pairing, and flow-warp tracking")


if __name__ == "__main__":
    main()
