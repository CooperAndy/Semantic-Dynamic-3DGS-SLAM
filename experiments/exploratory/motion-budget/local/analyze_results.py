"""Summarize sealed support, pre-GT features, and post-GT pose scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rows(directory: str, row_name: str, hash_key: str):
    receipt = read(HERE / directory / "receipt.json")
    path = HERE / directory / row_name
    if receipt["status"] != "COMPLETED" or sha(path) != receipt[hash_key]:
        raise ValueError(f"Unsealed or corrupt {directory}")
    return rows(path), receipt


def arm(label: str, support_dir: str, feature_dir: str, score_dir: str):
    support, _ = check_rows(support_dir, "support.jsonl", "support_sha256")
    features, _ = check_rows(feature_dir, "features.jsonl", "features_sha256")
    score = read(HERE / score_dir / "summary.json")
    if score["status"] != "CPU_SCORED":
        raise ValueError(f"Unscored {score_dir}")
    support_by_frame = {x["frame"]: x for x in support}
    feature_by_frame = {x["event"]["future"]: x for x in features}
    scored_by_frame = {x["frame"]: x for x in score["by_frame"]}
    if set(support_by_frame) != set(feature_by_frame):
        raise ValueError("Support and feature event sets differ")
    if {f for f, x in support_by_frame.items() if x["pose_probe_eligible"]} != set(scored_by_frame):
        raise ValueError("Scored event set differs from frozen support screen")
    out = []
    for frame, s in support_by_frame.items():
        f = feature_by_frame[frame]
        p = scored_by_frame.get(frame)
        out.append({
            "frame": frame,
            "role": f["event"]["role"],
            "support_pixels": s["common_fixed_support_pixels"],
            "eligible": s["pose_probe_eligible"],
            "risk_score_primary": f.get("risk_score_primary"),
            "pose_delta_mm": None if p is None else 1000 * p["after_minus_before_gt_translation_mean_m"],
            "harm_gt3mm": None if p is None else p["after_minus_before_gt_translation_mean_m"] > 0.003,
            "anchor_psnr_delta_db": None if p is None else p["after_minus_before_anchor_psnr_db"],
            "anchor_depth_mae_delta_m": None if p is None else p["after_minus_before_anchor_depth_mae_m"],
        })
    return {"name": label, "registered": len(support), "eligible": len(scored_by_frame),
            "rows": out}


def auc(scored):
    positive = [x for x in scored if x["harm_gt3mm"]]
    negative = [x for x in scored if not x["harm_gt3mm"]]
    if not positive or not negative:
        return None
    return sum((p["risk_score_primary"] > n["risk_score_primary"])
               + 0.5 * (p["risk_score_primary"] == n["risk_score_primary"])
               for p in positive for n in negative) / (len(positive) * len(negative))


holdout = arm("frozen_original_holdout", "support_v1", "features_v1", "scored_v1_corrected")
grid = arm("adaptive_exploratory_grid", "grid_support_v1", "grid_features_v1", "grid_scored_v1")
for group in (holdout, grid):
    scoreable = [x for x in group["rows"] if x["eligible"]]
    group["harm_gt3mm_count"] = sum(x["harm_gt3mm"] for x in scoreable)
    group["risk_auc_descriptive"] = auc(scoreable) if group is grid else None
    group["risk_ranking_desc"] = [x["frame"] for x in sorted(scoreable,
                                              key=lambda x: x["risk_score_primary"], reverse=True)]
holdout["prospective"] = {
    "registered": sum(x["role"] == "prospective_dev" for x in holdout["rows"]),
    "eligible": sum(x["eligible"] and x["role"] == "prospective_dev" for x in holdout["rows"]),
    "harm_gt3mm_count": sum(x["harm_gt3mm"] is True and x["role"] == "prospective_dev" for x in holdout["rows"]),
    "risk_auc_descriptive": auc([x for x in holdout["rows"] if x["eligible"] and x["role"] == "prospective_dev"]),
}
holdout["process_control"] = {
    "registered": sum(x["role"] == "process_control" for x in holdout["rows"]),
    "eligible": sum(x["eligible"] and x["role"] == "process_control" for x in holdout["rows"]),
    "harm_gt3mm_count": sum(x["harm_gt3mm"] is True and x["role"] == "process_control" for x in holdout["rows"]),
}
out = {"status": "DERIVED_FROM_SEALED_RECEIPTS", "holdout": holdout, "grid": grid,
       "interpretation_boundary": "Never pool process controls, frozen holdouts, or adaptive grid as one confirmatory sample."}
path = HERE / "analysis_summary.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: {"registered": v["registered"], "eligible": v["eligible"],
                      "harm": v["harm_gt3mm_count"], "auc": v["risk_auc_descriptive"],
                      "rank": v["risk_ranking_desc"]} for k, v in
                  (("holdout", holdout), ("grid", grid))}, indent=2))
