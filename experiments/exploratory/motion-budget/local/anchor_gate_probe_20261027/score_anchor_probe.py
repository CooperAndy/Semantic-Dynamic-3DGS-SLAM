"""Audit sealed GT-blind anchor recovery and evaluate the frozen event rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol_path = HERE / "protocol_pose_v1.json"
protocol = read(protocol_path)
support_dir = HERE / "anchor_support_v1"
run_dir = HERE / "anchor_pose_v1"
support_receipt_path = support_dir / "receipt.json"
support_rows_path = support_dir / "support.jsonl"
support_receipt = read(support_receipt_path)
receipt_path = run_dir / "receipt.json"
receipt = read(receipt_path)
trials_path = run_dir / "trials.jsonl"
quality_path = run_dir / "anchor_quality.jsonl"
if receipt["status"] != "COMPLETED" or receipt["phase"] != protocol["phase"]:
    raise ValueError("Pose run not sealed or phase mismatch")
if receipt["protocol_sha256"] != sha(protocol_path):
    raise ValueError("Pose protocol hash mismatch")
if sha(support_receipt_path) != protocol["support_receipt_sha256"]:
    raise ValueError("Support receipt hash mismatch")
if support_receipt["status"] != "COMPLETED" or sha(support_rows_path) != support_receipt["support_sha256"]:
    raise ValueError("Support output not sealed")
for path, key in ((trials_path, "trials_sha256"), (quality_path, "quality_sha256")):
    if sha(path) != receipt[key]:
        raise ValueError(f"Pose output hash mismatch: {path.name}")
if sha(HERE / "pose_worker_anchor.py") != receipt["worker_sha256"]:
    raise ValueError("Worker hash mismatch")
if sha(HERE / "m12_worker.py") != receipt["m12_worker_sha256"]:
    raise ValueError("Shared worker hash mismatch")
if sha(HERE / "pose_update.py") != receipt["pose_update_sha256"]:
    raise ValueError("Pose update hash mismatch")
for snap, expected in receipt["map_sha256"].items():
    path = PARENT / "grid_maps_G00_S0" / f"params{snap}.npz"
    if sha(path) != expected:
        raise ValueError(f"Map hash mismatch {snap}")
if len(receipt["map_sha256"]) != 4:
    raise ValueError("Expected four adjacent snapshots")

trials = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
if len(trials) != receipt["expected_trials"] or len(trials) != 40:
    raise ValueError("Incomplete registered trial set")
if any(row["status"] != "COMPLETED" or row["steps_completed"] != protocol["optimization"]["steps"]
       for row in trials):
    raise ValueError("Incomplete optimizer trial")
expected = {(p["frame"], c["id"], arm) for p in protocol["pairs"]
            for c in protocol["perturbations"] for arm in ("before", "after")}
actual = {(row["frame"], row["condition"]["id"], row["arm"]) for row in trials}
if actual != expected:
    raise ValueError("Trial identities differ from frozen protocol")


def pose_error_to_anchor(row):
    anchor = np.asarray(row["anchor_w2c"], dtype=np.float64)
    selected = np.asarray(row["selected_w2c"], dtype=np.float64)
    anchor_c2w, selected_c2w = np.linalg.inv(anchor), np.linalg.inv(selected)
    translation_m = float(np.linalg.norm(selected_c2w[:3, 3] - anchor_c2w[:3, 3]))
    relative_r = anchor_c2w[:3, :3].T @ selected_c2w[:3, :3]
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(relative_r) - 1) / 2, -1, 1))))
    return translation_m, angle_deg


index = {}
for row in trials:
    index[(row["frame"], row["condition"]["id"], row["arm"])] = row
per_view = []
for pair in protocol["pairs"]:
    by_arm = {arm: [pose_error_to_anchor(index[(pair["frame"], cond["id"], arm)])
                    for cond in protocol["perturbations"]] for arm in ("before", "after")}
    before_mean = float(np.mean([x[0] for x in by_arm["before"]]))
    after_mean = float(np.mean([x[0] for x in by_arm["after"]]))
    before_rot = float(np.mean([x[1] for x in by_arm["before"]]))
    after_rot = float(np.mean([x[1] for x in by_arm["after"]]))
    per_view.append({
        "event_role": pair["role"], "frame": pair["frame"], "support_pixels": pair["support_pixels"],
        "before_mean_anchor_translation_error_m": before_mean,
        "after_mean_anchor_translation_error_m": after_mean,
        "after_minus_before_m": after_mean - before_mean,
        "before_mean_anchor_rotation_error_deg": before_rot,
        "after_mean_anchor_rotation_error_deg": after_rot,
        "after_minus_before_rotation_deg": after_rot - before_rot,
        "per_perturbation_after_minus_before_m": {
            cond["id"]: pose_error_to_anchor(index[(pair["frame"], cond["id"], "after")])[0]
            - pose_error_to_anchor(index[(pair["frame"], cond["id"], "before")])[0]
            for cond in protocol["perturbations"]
        },
    })

events = []
for event in protocol["events"]:
    views = [row for row in per_view if row["event_role"] == event["role"]]
    flagged_views = [row["frame"] for row in views if row["after_minus_before_m"] > 0.003]
    eligible = len(views)
    flagged = eligible >= 2 and len(flagged_views) >= 2
    future_row = next(row for row in read(PARENT / "grid_scored_v1/summary.json")["by_frame"]
                      if row["frame"] == event["future_case"])
    events.append({
        "role": event["role"], "future_frame": event["future_case"],
        "eligible_observed_views": eligible, "registered_observed_views": event["views"],
        "unscorable_views": [f for f in event["views"] if f not in {r["frame"] for r in views}],
        "per_view_flagged_frames": flagged_views, "probe_flag": flagged,
        "previously_sealed_future_gt_translation_delta_mm": 1000 * future_row["after_minus_before_gt_translation_mean_m"],
        "future_harm_over_3mm": future_row["after_minus_before_gt_translation_mean_m"] > 0.003,
    })

out = {
    "status": "DERIVED_AFTER_GPU_OUTPUT_SEAL",
    "worker_GT_access": receipt["GT_access"],
    "support_protocol_sha256": sha(HERE / "protocol_support_v1.json"),
    "pose_protocol_sha256": sha(protocol_path),
    "pose_receipt_sha256": sha(receipt_path),
    "trial_count": len(trials), "eligible_view_count": len(per_view),
    "per_view": per_view, "events": events,
    "interpretation": "Targeted two-case post hoc falsification only; view-level local recovery to the fixed ORB anchor is not GT accuracy, and the cases were selected after prior future-view outcomes were known.",
}
path = HERE / "anchor_gate_summary.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(events, ensure_ascii=False, indent=2))
