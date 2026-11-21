"""Freeze all support-eligible registered events for GT-blind pose recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


registration = read(HERE / "protocol_grid_replay_v1.json")
replay_receipt_path = HERE / "grid_replay_v1/receipt.json"
support_receipt_path = HERE / "grid_support_v1/receipt.json"
support_rows_path = HERE / "grid_support_v1/support.jsonl"
replay = read(replay_receipt_path)
support_receipt = read(support_receipt_path)
if replay["status"] != "COMPLETED" or support_receipt["status"] != "COMPLETED":
    raise ValueError("Incomplete replay or support")
if support_receipt["protocol_sha256"] != sha(HERE / "protocol_grid_support_v1.json"):
    raise ValueError("Support protocol hash mismatch")
if support_receipt["support_sha256"] != sha(support_rows_path):
    raise ValueError("Support row hash mismatch")
rows = [json.loads(line) for line in support_rows_path.read_text(encoding="utf-8").splitlines()]
if len(rows) != len(registration["events"]):
    raise ValueError("Registered event count mismatch")
pairs, unscorable = [], []
for event, row in zip(registration["events"], rows):
    if (event["before"], event["after"], event["future"]) != (
        row["before"], row["after"], row["frame"]
    ):
        raise ValueError("Event/support order mismatch")
    item = {"before": event["before"], "after": event["after"],
            "frame": event["future"], "role": event["role"],
            "support_pixels": row["common_fixed_support_pixels"]}
    (pairs if row["pose_probe_eligible"] else unscorable).append(item)
protocol = {
    "phase": "m15_capped_grid_pose_exploratory_v1", "date": "2026-10-27",
    "status": "FROZEN_BEFORE_POSE_RUN",
    "scope": "Exposed KITTI00 DEV, exploratory G08 capped-native grid, offline GT-blind next-frame pose recovery",
    "root": "/root/autodl-tmp/stage3_m15_capped_grid_20261027",
    "num_frames": registration["num_frames"],
    "all_query_frames": [e["future"] for e in registration["events"]],
    "replay_receipt_sha256": sha(replay_receipt_path),
    "support_receipt_sha256": sha(support_receipt_path),
    "pairs": pairs, "unscorable_support": unscorable,
    "perturbations": [
        {"id": "yaw_m1", "kind": "yaw_deg", "value": -1.0},
        {"id": "yaw_p1", "kind": "yaw_deg", "value": 1.0},
        {"id": "tx_p5cm", "kind": "tx_m", "value": 0.05},
    ],
    "optimization": {
        "steps": 160, "seed": 0, "lr_rotation_rad": 0.002,
        "lr_translation_m": 0.005, "rgb_weight": 0.5, "depth_weight": 1.0,
        "pixel_stride": 4, "min_common_pixels": 1000,
        "alpha_threshold": 0.99, "depth_min_m": 3.0, "depth_max_m": 60.0,
        "selection": "Minimum fixed-support RGB+E6a-depth L1 loss over all pre-update steps, no GT access",
    },
    "evaluation": {
        "primary": "After sealing GPU output, paired after-minus-before GT translation/rotation error",
        "alignment": "First 400 ORB frames excluding 20 mapping holdouts and all 16 grid query frames, fixed SE3, no scale",
        "gt_sha256": "90791a4113df979b149fa9e1104e960ea59f525a8318a202dbb6aec1a3d88793",
        "calib_sha256": "7cfc9ab047da6bd829d669e31b2a0d72c278d084fa0c02f2f0320795d51a5d01",
        "harm_label": "Per-event mean over three perturbations: after-minus-before GT translation > 0.003 m",
        "role": "adaptive exploratory grid after first support screen, not confirmatory",
        "independent_test": False,
    },
    "interpretation": "Only support-eligible events included by frozen pre-GT rule. All 16 registered grid events stay in feasibility denominator. Adaptive exploratory design and correlated sequence cannot authorize M2.",
}
path = HERE / "protocol_grid_pose_v1.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(sha(path), "eligible", len(pairs), "unscorable", len(unscorable))
