"""CPU-only GT scorer for sealed M1.5 adjacent-update future pose recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    out = np.repeat(np.eye(4)[None], len(data), axis=0)
    out[:, :3, :] = data
    return out


def rigid_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd((target - target.mean(0)).T @ (source - source.mean(0)))
    sign = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    return u @ sign @ vt


def angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "protocol", "support", "manifest", "orb", "gt", "calib", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    receipt = json.loads((args.run / "receipt.json").read_text(encoding="utf-8"))
    support_receipt = json.loads((args.support / "receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETED" or receipt["phase"] != protocol["phase"]:
        raise ValueError("Pose run not sealed")
    if (sha256(args.gt) != protocol["evaluation"]["gt_sha256"]
            or sha256(args.calib) != protocol["evaluation"]["calib_sha256"]):
        raise ValueError("GT or calibration version differs from frozen scoring protocol")
    checks = [
        (args.protocol, receipt["protocol_sha256"]),
        (args.support / "receipt.json", receipt["support_receipt_sha256"]),
        (args.support / "support.jsonl", support_receipt["support_sha256"]),
        (args.manifest, receipt["manifest_sha256"]),
        (args.orb, receipt["orb_pose_sha256"]),
        (args.run / "trials.jsonl", receipt["trials_sha256"]),
        (args.run / "anchor_quality.jsonl", receipt["quality_sha256"]),
    ]
    for path, expected in checks:
        if sha256(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")
    if any(value != support_receipt["map_sha256"]["G00_S0"].get(key)
           for key, value in receipt["map_sha256"].items()):
        raise ValueError("Map hashes differ from sealed support screen")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {int(row["frame_id"]): row for row in manifest["rows"]}
    trials = [json.loads(s) for s in (args.run / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    quality = [json.loads(s) for s in (args.run / "anchor_quality.jsonl").read_text(encoding="utf-8").splitlines()]
    support = [json.loads(s) for s in (args.support / "support.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(s)["pose_probe_eligible"]]
    expected = {(p["frame"], c["id"], arm) for p in protocol["pairs"]
                for c in protocol["perturbations"] for arm in ("before", "after")}
    actual = {(x["frame"], x["condition"]["id"], x["arm"]) for x in trials}
    if actual != expected or len(trials) != len(expected) or len(trials) != receipt["expected_trials"]:
        raise ValueError("Incomplete trial identities")
    if any(x["status"] != "COMPLETED" or x["steps_completed"] != protocol["optimization"]["steps"]
           for x in trials):
        raise ValueError("Incomplete pose optimization")
    qindex = {(x["frame"], x["arm"]): x for x in quality}
    if len(qindex) != 2 * len(protocol["pairs"]):
        raise ValueError("Incomplete anchor quality")
    if [x["common_fixed_support_pixels"] for x in support] != [x["support_pixels"] for x in protocol["pairs"]]:
        raise ValueError("Support mismatch")

    orb_c2w = poses(args.orb)
    orb_w2c = np.linalg.inv(orb_c2w)
    gt = poses(args.gt)
    calib = {}
    for line in args.calib.read_text(encoding="utf-8").splitlines():
        key, values = line.split(":", 1)
        calib[key] = np.fromstring(values, sep=" ").reshape(3, 4)

    def offset(key: str) -> np.ndarray:
        return np.linalg.solve(calib[key][:, :3], calib[key][:, 3])

    p2_to_p0 = offset("P2") - offset("P0")
    train = np.asarray([i for i in range(400) if i not in manifest["mapping_holdout_frames"]], dtype=int)
    if len(train) != 380:
        raise ValueError("Expected 380 nonholdout alignment frames")
    raw_ids = np.asarray([manifest_rows[int(i)]["source_frame_id"] for i in train], dtype=int)
    orb_p0 = orb_c2w[train, :3, 3] + np.einsum("nij,j->ni", orb_c2w[train, :3, :3], p2_to_p0)
    align_r = rigid_rotation(orb_p0, gt[raw_ids, :3, 3])
    align_t = gt[raw_ids, :3, 3].mean(axis=0) - align_r @ orb_p0.mean(axis=0)

    def errors(pose_w2c: np.ndarray, frame: int) -> dict:
        c2w = np.linalg.inv(pose_w2c)
        raw = manifest_rows[frame]["source_frame_id"]
        local_t = float(np.linalg.norm(c2w[:3, 3] - orb_c2w[frame, :3, 3]))
        local_r = angle_deg(orb_c2w[frame, :3, :3].T @ c2w[:3, :3])
        p0_t = c2w[:3, 3] + c2w[:3, :3] @ p2_to_p0
        aligned_t = align_r @ p0_t + align_t
        aligned_r = align_r @ c2w[:3, :3]
        return {
            "gt_translation_m": float(np.linalg.norm(aligned_t - gt[raw, :3, 3])),
            "gt_rotation_deg": angle_deg(gt[raw, :3, :3].T @ aligned_r),
            "local_translation_m": local_t,
            "local_rotation_deg": local_r,
        }

    scored = []
    for row in trials:
        frame = int(row["frame"])
        if not np.allclose(np.asarray(row["anchor_w2c"]), orb_w2c[frame], rtol=0, atol=1e-6):
            raise ValueError("Trial anchor differs from sealed ORB input")
        scored.append({
            "frame": frame, "condition": row["condition"]["id"], "arm": row["arm"],
            "initial": errors(np.asarray(row["initial_w2c"]), frame),
            "selected": errors(np.asarray(row["selected_w2c"]), frame),
            "selected_pre_step": row["selected_pre_step"],
            "selected_loss": row["selected_loss"],
        })
    index = {(x["frame"], x["condition"], x["arm"]): x for x in scored}
    paired = []
    for pair in protocol["pairs"]:
        frame = pair["frame"]
        for condition in protocol["perturbations"]:
            a = index[(frame, condition["id"], "before")]
            b = index[(frame, condition["id"], "after")]
            if not np.allclose(a["initial"]["gt_translation_m"], b["initial"]["gt_translation_m"], atol=1e-9):
                raise ValueError("Paired initial GT error differs")
            paired.append({
                "frame": frame, "condition": condition["id"],
                "after_minus_before_gt_translation_m": b["selected"]["gt_translation_m"] - a["selected"]["gt_translation_m"],
                "after_minus_before_gt_rotation_deg": b["selected"]["gt_rotation_deg"] - a["selected"]["gt_rotation_deg"],
                "before_gt_translation_m": a["selected"]["gt_translation_m"],
                "after_gt_translation_m": b["selected"]["gt_translation_m"],
            })
    by_frame = []
    for pair in protocol["pairs"]:
        frame = pair["frame"]
        subset = [x for x in paired if x["frame"] == frame]
        before, after = qindex[(frame, "before")], qindex[(frame, "after")]
        by_frame.append({
            "frame": frame, "support_pixels": pair["support_pixels"],
            "after_minus_before_gt_translation_mean_m": mean([x["after_minus_before_gt_translation_m"] for x in subset]),
            "after_minus_before_gt_rotation_mean_deg": mean([x["after_minus_before_gt_rotation_deg"] for x in subset]),
            "after_minus_before_anchor_psnr_db": after["anchor_psnr_db"] - before["anchor_psnr_db"],
            "after_minus_before_anchor_depth_mae_m": after["anchor_depth_mae_m"] - before["anchor_depth_mae_m"],
            "before_coverage099": before["coverage099"],
            "after_coverage099": after["coverage099"],
        })
    summary = {
        "status": "CPU_SCORED", "scope": protocol["scope"],
        "run_receipt_sha256": sha256(args.run / "receipt.json"),
        "scorer_sha256": sha256(Path(__file__)),
        "gt_sha256": sha256(args.gt), "calib_sha256": sha256(args.calib),
        "alignment": "Fixed SE3 on first 400 frames excluding 20 mapping holdouts, P2-to-P0, no scale",
        "trials": len(scored), "pairs": len(paired), "distinct_update_events": len(by_frame),
        "after_minus_before_gt_translation_mean_m": mean([x["after_minus_before_gt_translation_m"] for x in paired]),
        "after_better_gt_translation_pairs": sum(x["after_minus_before_gt_translation_m"] < 0 for x in paired),
        "by_frame": by_frame,
        "limitations": [f"{len(by_frame)} distinct but correlated adjacent updates from one exposed KITTI00 DEV replay",
                        "Three perturbations per frame are correlated",
                        "Fixed-map offline recovery, not online tracking or independent Test",
                        "E6a depth is a proxy; GT accessed only by this CPU scorer after GPU outputs were sealed"],
    }
    args.out.mkdir(parents=True)
    for filename, data in (("scored_trials.jsonl", scored), ("paired.jsonl", paired)):
        (args.out / filename).write_text("".join(json.dumps(x) + "\n" for x in data), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
