"""Freeze a targeted, GT-blind observed-view recovery probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPLAY = PARENT / "grid_replay_v1/receipt.json"
receipt = json.loads(REPLAY.read_text(encoding="utf-8"))
if receipt["status"] != "COMPLETED":
    raise ValueError("Grid replay is not sealed")
events = [
    {"before": 73, "after": 74, "future_case": 75, "views": [72, 73, 74], "role": "targeted_case_A"},
    {"before": 163, "after": 164, "future_case": 165, "views": [162, 163, 164], "role": "targeted_case_B"},
]
snaps = sorted({s for e in events for s in (e["before"], e["after"])})
protocol = {
    "task": "M1.5 targeted observed-view map-update recovery probe",
    "phase": "m15_anchor_gate_targeted_v1",
    "date": "2026-10-28",
    "status": "FROZEN_BEFORE_RENDER",
    "scope": "Exploratory two-event case study on existing capped G08 KITTI00 DEV snapshots; no independent test",
    "selection_note": "The two cases were selected because prior sealed next-view scoring showed contrasting outcomes. This targeted diagnostic is post hoc and cannot validate general predictive performance.",
    "g08_root": "/root/autodl-tmp/stage3_m15_capped_grid_20261027",
    "num_frames": 200,
    "replay_receipt_sha256": hashlib.sha256(REPLAY.read_bytes()).hexdigest(),
    "input_manifest_sha256": "476bac19e98afd225d344b49259ccd2225a5ac5416d1a78e382a7c9d552771c6",
    "orb_pose_sha256": "119ce89f24bfbbbb9b6ee87c9bb3de8c076e452fc04ec8afcfd3b3b69d4be433",
    "registered_observed_views": sorted({f for e in events for f in e["views"]}),
    "variants": {"G00_S0": {str(s): receipt["map_sha256"][str(s)] for s in snaps}},
    "transitions": events,
    "support": {
        "alpha_threshold": 0.99, "depth_min_m": 3.0, "depth_max_m": 60.0,
        "pixel_stride": 4, "min_common_pixels_for_pose_probe": 1000,
        "definition": "Both-map finite RGB/depth, alpha>0.99, observed/rendered E6a depth 3-60m and identical stride-4 pixels at ORB P2 anchor",
        "abstain": "Keep all six registered view rows; do not replace low-support views. An event is decidable only if at least two of its three views are eligible.",
    },
    "pose_probe": {
        "perturbations": [
            {"id": "yaw_m1", "kind": "yaw_deg", "value": -1.0},
            {"id": "yaw_p1", "kind": "yaw_deg", "value": 1.0},
            {"id": "tx_m5cm", "kind": "tx_m", "value": -0.05},
            {"id": "tx_p5cm", "kind": "tx_m", "value": 0.05},
        ],
        "steps": 160, "seed": 0, "lr_rotation_rad": 0.002, "lr_translation_m": 0.005,
        "rgb_weight": 0.5, "depth_weight": 1.0, "pixel_stride": 4,
        "selection": "Minimum fixed-support RGB+E6a-depth L1 over pre-update steps",
    },
    "predeclared_diagnostic_rule": {
        "per_view": "Mean over four perturbations of translation distance from the selected pose to that view's fixed ORB anchor, separately for before and after maps",
        "event_flag": "After-map mean anchor-recovery error exceeds before-map mean by more than 0.003m on at least two eligible views",
        "abstain": "Fewer than two eligible views means no event decision",
        "purpose": "Test whether past/current-view recovery degradation flags the already observed future-view harmful case while not flagging the contrast case; targeted exploratory falsification only",
    },
    "gt_access": "No GT in support or GPU pose worker. Compare diagnostic calls only with previously sealed future-frame labels after local outputs are sealed.",
}
path = HERE / "protocol_support_v1.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(hashlib.sha256(path.read_bytes()).hexdigest(), "views", protocol["registered_observed_views"])
