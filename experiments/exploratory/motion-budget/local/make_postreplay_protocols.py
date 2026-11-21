"""Freeze support and feature protocols from sealed replay receipt; no outcomes read."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = "/root/autodl-tmp/stage3_m15_capped_multi_20261027"
registration = json.loads((HERE / "protocol_replay_v1.json").read_text(encoding="utf-8"))
receipt_path = HERE / "replay_v1/receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt["status"] != "COMPLETED" or receipt["protocol_sha256"] != hashlib.sha256(
    (HERE / "protocol_replay_v1.json").read_bytes()).hexdigest():
    raise ValueError("Replay was not sealed against registration")
if receipt["saved_snapshots"] != [i for event in registration["events"]
                                  for i in (event["before"], event["after"])]:
    raise ValueError("Registered snapshots missing")
receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

support = {
    "task": "M1.5 G08-capped adjacent map future support",
    "date": "2026-10-27", "status": "FROZEN_BEFORE_RENDER",
    "scope": "Exposed KITTI00 DEV, capped G08 G00_S0 200-frame replay, no GT or online tracking claim",
    "g08_root": root,
    "input_manifest_sha256": registration["input_manifest_sha256"],
    "orb_pose_sha256": "119ce89f24bfbbbb9b6ee87c9bb3de8c076e452fc04ec8afcfd3b3b69d4be433",
    "variants": {"G00_S0": receipt["map_sha256"]},
    "transitions": [dict(before=e["before"], after=e["after"], future_holdouts=[e["future"]],
                         role=e["role"]) for e in registration["events"]],
    "support": {
        "alpha_threshold": 0.99, "depth_min_m": 3.0, "depth_max_m": 60.0,
        "pixel_stride": 4, "min_common_pixels_for_pose_probe": 1000,
        "definition": "Both-map finite RGB/depth, alpha>0.99, observed/rendered E6a depth 3-60m and stride-4 pixels at identical ORB P2 camera",
    },
    "decision": "Retain all 10 registered future-view pairs; mark low support unscorable without changing threshold or frame."
}
feature = {
    "task": "M1.5 capped map pre-future features",
    "date": "2026-10-27", "status": "FROZEN_BEFORE_FEATURE_RUN",
    "scope": "GT-blind feature values on already observed audit/current frames, before future holdout is read",
    "root": root, "replay_receipt_sha256": receipt_sha,
    "orb_pose_sha256": support["orb_pose_sha256"],
    "events": registration["events"],
    "feature": {
        "audit_rule": "audit=before snapshot frame=t-1; current=after snapshot frame=t",
        "risk_score_primary": "basin_margin_before - basin_margin_after at audit frame; larger means weaker candidate-map pose basin",
        "basin_margin": "Mean fixed-support loss at four perturbations minus loss at ORB anchor",
        "probes": [
            {"id": "yaw_m1", "kind": "yaw_deg", "value": -1.0},
            {"id": "yaw_p1", "kind": "yaw_deg", "value": 1.0},
            {"id": "tx_m5cm", "kind": "tx_m", "value": -0.05},
            {"id": "tx_p5cm", "kind": "tx_m", "value": 0.05},
        ],
        "rgb_weight": 0.5, "depth_weight": 1.0,
        "alpha_threshold": 0.99, "depth_min_m": 3.0, "depth_max_m": 60.0,
        "pixel_stride": 4, "min_common_pixels": 1000,
        "secondary": ["audit RGB PSNR delta", "audit depth MAE delta", "current RGB PSNR delta",
                      "current depth MAE delta", "alpha coverage delta", "Gaussian count delta fraction"],
        "decision": "If audit support <1000, primary risk unavailable; never substitute future view or reselect audit view."
    },
}
for name, value in (("protocol_support_v1.json", support), ("protocol_feature_v1.json", feature)):
    path = HERE / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())
