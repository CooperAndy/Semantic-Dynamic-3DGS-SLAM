"""Freeze pose trials from the pre-registered support screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
support_protocol_path = HERE / "protocol_support_v1.json"
support_receipt_path = HERE / "anchor_support_v1/receipt.json"
support_rows_path = HERE / "anchor_support_v1/support.jsonl"
support_protocol = json.loads(support_protocol_path.read_text(encoding="utf-8"))
support_receipt = json.loads(support_receipt_path.read_text(encoding="utf-8"))
if support_receipt["status"] != "COMPLETED":
    raise ValueError("Support screen incomplete")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if support_receipt["protocol_sha256"] != sha(support_protocol_path):
    raise ValueError("Support protocol hash mismatch")
if support_receipt["support_sha256"] != sha(support_rows_path):
    raise ValueError("Support rows hash mismatch")
rows = [json.loads(line) for line in support_rows_path.read_text(encoding="utf-8").splitlines()]
expected = {(e["before"], e["after"], f, e["role"])
            for e in support_protocol["transitions"] for f in e["views"]}
if {(x["before"], x["after"], x["frame"], x["role"]) for x in rows} != expected:
    raise ValueError("Support rows do not match pre-registered views")
pairs, unscorable = [], []
for row in rows:
    item = {"before": row["before"], "after": row["after"], "frame": row["frame"],
            "role": row["role"], "support_pixels": row["common_fixed_support_pixels"]}
    (pairs if row["pose_probe_eligible"] else unscorable).append(item)
for event in support_protocol["transitions"]:
    eligible_count = sum(p["role"] == event["role"] for p in pairs)
    if eligible_count < 2:
        raise ValueError(f"Abstain for {event['role']}: fewer than two eligible views")
protocol = {
    "phase": "m15_anchor_gate_targeted_pose_v1",
    "date": "2026-10-28",
    "status": "FROZEN_BEFORE_POSE_RUN",
    "scope": support_protocol["scope"],
    "root": support_protocol["g08_root"],
    "num_frames": support_protocol["num_frames"],
    "registered_observed_views": support_protocol["registered_observed_views"],
    "events": support_protocol["transitions"],
    "pairs": pairs,
    "unscorable_support": unscorable,
    "replay_receipt_sha256": support_protocol["replay_receipt_sha256"],
    "support_protocol_sha256": sha(support_protocol_path),
    "support_receipt_sha256": sha(support_receipt_path),
    "support_rows_sha256": sha(support_rows_path),
    "optimization": {
        "steps": 160, "seed": 0, "lr_rotation_rad": 0.002,
        "lr_translation_m": 0.005, "rgb_weight": 0.5, "depth_weight": 1.0,
        "pixel_stride": 4, "min_common_pixels": 1000,
        "alpha_threshold": 0.99, "depth_min_m": 3.0, "depth_max_m": 60.0,
        "selection": "Minimum fixed-support RGB+E6a-depth L1 loss over pre-update steps",
    },
    "perturbations": [
        {"id": "yaw_m1", "kind": "yaw_deg", "value": -1.0},
        {"id": "yaw_p1", "kind": "yaw_deg", "value": 1.0},
        {"id": "tx_m5cm", "kind": "tx_m", "value": -0.05},
        {"id": "tx_p5cm", "kind": "tx_m", "value": 0.05},
    ],
    "decision_rule": support_protocol["predeclared_diagnostic_rule"],
    "gt_access": "GPU worker blocks GT paths; diagnostics scored relative to ORB anchor only",
}
out = HERE / "protocol_pose_v1.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(sha(out), "eligible", len(pairs), "unscorable", len(unscorable))
