"""Separate vehicle recognition and closed-loop comparison against frozen S23."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import shutil

import numpy as np

from s23p_inputs import ROOT, S23, VEHICLES, read, sha, write

sys.path.insert(0, str(S23))
from automatic_gt import official_gt
from experiment import LIMITS, design
from http_range_zip import RangeZip

MOVING, STATIC = [252, 258, 257], [10, 18, 13]
FIXED = set(design()["evaluation_frames"])
URLS = {
    "velodyne": ("bin", "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"),
    "labels": ("label", "https://semantic-kitti.org/assets/data_odometry_labels.zip"),
}


def measure(truth, allowed, states, detected):
    p = states == 2
    return np.array([(allowed & truth & p).sum(), (allowed & ~truth & p).sum(),
                     (allowed & truth & ~p).sum(), (allowed & ~truth & ~p).sum(),
                     (allowed & ~detected).sum(), (allowed & (states == 1)).sum()], dtype=np.int64)


def recognition(sequence):
    base = ROOT / "inputs" / sequence
    ready = read(base / "prediction_complete.json")
    assert ready["manifest_sha256"] == sha(base / "observations/manifest.json")
    original = ROOT / "restored/bundles" / sequence / "observations"
    reference = S23 / "evaluation_only" / f"K{sequence}"
    out = ROOT / "recognition" / sequence
    out.mkdir(parents=True, exist_ok=False)
    runtime = read(base / "runtime.json")
    shape = runtime["H"], runtime["W"]
    archives = {}
    records = []
    download_error = None
    with (out / "frames.jsonl").open("x", buffering=1) as ledger:
        for frame in range(1000):
            name = f"{frame:06}"
            try:
                sources = {}
                for kind, (ext, url) in URLS.items():
                    p = reference / kind / f"{name}.{ext}"
                    if not p.exists():
                        if download_error:
                            raise RuntimeError("Official range archive unavailable")
                        if kind not in archives:
                            archives[kind] = RangeZip(url, 6 * 2**30)
                        archive = archives[kind]
                        if not isinstance(archive, tuple):
                            names = {}
                            for member in archive.info:
                                if member.endswith(f".{ext}") and f"/{sequence}/{kind}/" in member:
                                    names[Path(member).stem] = member
                            archive = archives[kind] = (archive, names)
                        p = out / f"{kind}.{ext}"
                        p.write_bytes(archive[0].read(archive[1][name]))
                    sources[kind] = p
            except Exception as error:
                download_error = repr(error) if download_error is None else download_error
                continue
            gt, _ = official_gt(sources["velodyne"], sources["labels"], reference / "calib.txt", shape)
            labels = gt["raw_semantic"]
            truth = np.isin(labels, MOVING)
            support = (gt["point_index"] >= 0) & (truth | np.isin(labels, STATIC))
            with (np.load(original / f"{name}.npz", allow_pickle=False) as old,
                  np.load(base / "observations" / f"{name}.npz", allow_pickle=False) as new):
                assert np.array_equal(old["instance_id"], new["instance_id"])
                owner = old["instance_id"]
                instances = read(original.parent / "instances" / f"{name}.json")["instances"]
                detected = np.isin(owner, [int(x["track_id"]) for x in instances
                                           if x["category"].lower() in VEHICLES])
                record = dict(frame=frame, points_sha256=sha(sources["velodyne"]), labels_sha256=sha(sources["labels"]))
                for tag, allowed in (("all", support), ("detected", support & detected)):
                    record[tag] = {method: measure(truth, allowed, source["state"], detected).tolist()
                                   for method, source in (("TSDD", old), ("Pi3MOS", new))}
            records.append(record)
            ledger.write(json.dumps(record) + "\n")
            if frame % 100 == 0:
                print(sequence, frame, "recognition frames", len(records), flush=True)
    result = {"status": "FULL_WINDOW" if len(records) == 1000 else "NOT_EVALUABLE_FULL_WINDOW",
              "frames": len(records), "download_error": download_error, "scopes": {}}
    ratio = lambda a, b: float(a / b) if b else None
    for tag in ("all", "detected"):
        for cohort, predicate in (("full", lambda x: True), ("warmup", lambda x: x < 7),
                                  ("postwarmup", lambda x: x >= 7), ("fixed20", lambda x: x in FIXED)):
            chosen = [row for row in records if predicate(row["frame"])]
            scope = {}
            for method in ("TSDD", "Pi3MOS"):
                tp, fp, fn, tn, absent, unknown = map(int, np.sum(
                    [r[tag][method] for r in chosen], axis=0)) if chosen else (0,) * 6
                scope[method] = dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=ratio(tp,tp+fp),
                    recall=ratio(tp,tp+fn), F1=ratio(2*tp,2*tp+fp+fn), IoU=ratio(tp,tp+fp+fn),
                    static_false_removal=ratio(fp,fp+tn), uncertain_or_unknown=ratio(unknown,tp+fp+fn+tn),
                    detected_coverage=ratio(tp+fp+fn+tn-absent,tp+fp+fn+tn))
            result["scopes"][f"{tag}_{cohort}"] = {"frames": len(chosen), "methods": scope}
    write(ROOT / "producer_results" / f"{sequence}.json", result)


def map_means(run, cohort):
    return next(row["means"] for row in run["summary"]
                if row["mode"] == "COMMON_GT" and row["cohort"] == cohort)


def pairs(path, control):
    from dynamic_pair_audit import input_pairs
    from collections import Counter
    counts = Counter()
    for i in range(1, 1000):
        name = f"{i:06}.json"
        first, second = (read(run / "proposals" / name) for run in (path, control))
        # Restored caches have different paths but retain the original bytes.
        reference_keys = ("frame", "current_frame", "depth_sha256", "confidence_sha256")
        assert all(first["causal_stereo_reference"][k] == second["causal_stereo_reference"][k]
                   for k in reference_keys)
        a, b = input_pairs(first), input_pairs(second)
        counts["new_only"] += sum((a-b).values())
        counts["control_only"] += sum((b-a).values())
        counts["changed_steps"] += int(a != b)
    return dict(counts)


def summary():
    out = ROOT / "report"
    out.mkdir(exist_ok=False)
    observations = read(ROOT / "progress.json")["completed"]
    assert len(observations) == 12
    comparisons = []
    render = out / "views"
    render.mkdir()
    for item in observations:
        seq, seed, group = (item[key] for key in ("sequence", "seed", "group"))
        new = S23 / item["cell"] / "runs" / group
        old_root = S23 / f"E3_K{seq}_S{seed}_R3_serial/runs"
        old, off = old_root / group, old_root / ("B" if group == "D" else "TM")
        result, base, disabled = (read(p / "evaluation/results.json") for p in (new,old,off))
        assert sha(new / "evaluation/results.json") == item["evaluation"]
        m, b, o = (map_means(e,"ALL") for e in (result,base,disabled))
        assert len(result["rows"]) == len(base["rows"]) == 60
        for a, previous in zip(result["rows"],base["rows"]):
            for key in ("mode","frame","source_frame","static_GT_pixels","semantic_GT_pixels"):
                assert a[key] == previous[key], key
        row = {key:item[key] for key in ("sequence","seed","group","archive_sha256")}
        for field in ("ATE_RMSE_m","RPE_translation_RMSE_m_1frame","RPE_rotation_RMSE_deg_1frame"):
            a, p, d = (e["trajectory"][field] for e in (result,base,disabled))
            row.update({f"{field}_old":p, f"{field}_new":a, f"{field}_off":d, f"{field}_delta":a-p,
                        f"{field}_improvement":1-a/p})
        for field in ("PSNR_dB","SSIM","LPIPS","coverage","static_RGB_PSNR_dB","static_depth_MAE_m",
                      "static_depth_coverage","sparse_mIoU","semantic_label_coverage","static_vehicle_depth_coverage",
                      "static_vehicle_semantic_accuracy"):
            a,p,d = m.get(field),b.get(field),o.get(field)
            row.update({f"{field}_old":p,f"{field}_new":a,f"{field}_off":d,
                        f"{field}_delta":a-p if a is not None and p is not None else None})
        row["depth_ratio"] = m["static_depth_MAE_m"] / b["static_depth_MAE_m"]
        for field in ("PSNR_dB","LPIPS"):
            row[f"recent_{field}_delta"] = map_means(result,"RECENT")[field] - map_means(base,"RECENT")[field]
        row["recent_LPIPS_tail_delta"] = max(a["LPIPS"]-p["LPIPS"] for a,p in zip(result["rows"],base["rows"])
             if a["mode"]=="COMMON_GT" and a["cohort"]=="RECENT")
        for field in ("geometry_failure_steps","prediction_fallback_steps","accepted_steps","geometry_qualified_steps"):
            row[f"{field}_delta"] = result["tracking_diagnostics"][field]-base["tracking_diagnostics"][field]
        for method, ev, path in (("old",base,old),("new",result,new)):
            row[f"{method}_wall_seconds"] = read(path/"receipt.json")["wall_seconds"]
            row[f"{method}_gpu_overlap_samples"] = ev["resource_audit"]["overlap_samples"]
            for key, value in ev["tracking_diagnostics"]["dynamic_consumption"].items():
                row[f"{method}_{key}"] = value
            for figure in ev["publication_views"]["files"]:
                source = S23 / figure["path"]
                assert sha(source) == figure["sha256"]
                shutil.copy2(source, render / f"K{seq}_S{seed}_{group}_{method}_{source.name}")
        row["new_vs_old_actual_pairs"] = pairs(new,old)
        row["new_vs_off_actual_pairs"] = pairs(new,off)
        comparisons.append(row)
    combined = {}
    for group in ("D","TMD"):
        windows = {}
        for seq in ("05","10"):
            rows = [r for r in comparisons if r["sequence"]==seq and r["group"]==group]
            fields = {k for r in rows for k in r if k.endswith(("_delta","_improvement")) or k=="depth_ratio"}
            stats = {}
            for field in sorted(fields):
                vals = [r[field] for r in rows if r.get(field) is not None]
                if vals:
                    stats[field] = dict(mean=mean(vals), std=stdev(vals) if len(vals)>1 else None, seeds=len(vals))
            windows[seq] = stats
        def score(seq,key):
            return windows[seq][key]["mean"]
        equal = {k:mean(score(s,k) for s in windows) for k in set(windows["05"]) & set(windows["10"])}
        ate, rpe = "ATE_RMSE_m_improvement", "RPE_translation_RMSE_m_1frame_improvement"
        track = equal[ate]>=LIMITS["ATE_improvement"] and equal[rpe]>=LIMITS["RPE_improvement"] and all(
            min(score(s,ate),score(s,rpe))>=-LIMITS["window_max_regression"] and
            score(s,"geometry_failure_steps_delta")<=0 for s in windows)
        maps = all(
            score(s,"static_RGB_PSNR_dB_delta")>=LIMITS["static_PSNR_gain_dB"] and
            score(s,"LPIPS_delta")<=LIMITS["static_LPIPS_max_delta"] and
            score(s,"SSIM_delta")>=LIMITS["SSIM_min_delta"] and score(s,"coverage_delta")>=LIMITS["coverage_min_delta"] and
            score(s,"depth_ratio")<=LIMITS["static_depth_MAE_max_ratio"] and
            score(s,"recent_PSNR_dB_delta")>=LIMITS["recent_PSNR_min_delta"] and
            score(s,"recent_LPIPS_delta")<=LIMITS["recent_LPIPS_max_delta"] and
            max(r["recent_LPIPS_tail_delta"] for r in comparisons if r["group"]==group and r["sequence"]==s)
            <= LIMITS["recent_LPIPS_tail_max_delta"] for s in windows)
        def direction(values):
            if min(values) >= 0 and max(values) > 0:
                return "Pi3MOS"
            if max(values) <= 0 and min(values) < 0:
                return "TSDD"
            return "TRADEOFF_OR_TIE"

        combined[group] = dict(windows=windows, window_equal=equal, passes_tracking=track, passes_mapping=maps,
            tracking_direction=direction([equal[ate],equal[rpe],-equal["geometry_failure_steps_delta"]]),
            mapping_direction=direction([equal["static_RGB_PSNR_dB_delta"],-equal["LPIPS_delta"],
                                         -equal["static_depth_MAE_m_delta"]]), passes_joint_adoption=track and maps)
    motion = {s:read(ROOT / "producer_results" / f"{s}.json") for s in ("05","10")}
    support = [motion[s]["scopes"]["all_full"]["methods"] for s in motion]
    motion_winner = "NOT_EVALUABLE"
    if all(m["status"]=="FULL_WINDOW" for m in motion.values()) and all(
        all(x[g]["F1"] is not None and x[g]["static_false_removal"] is not None for g in ("TSDD","Pi3MOS"))
        for x in support):
        scores = {g:mean(x[g]["F1"] for x in support) for g in ("TSDD","Pi3MOS")}
        motion_winner = "TRADEOFF_OR_TIE"
        for a,b in (("Pi3MOS","TSDD"),("TSDD","Pi3MOS")):
            if scores[a]>scores[b] and all(x[a]["static_false_removal"]<=x[b]["static_false_removal"] for x in support):
                motion_winner = a
    inference = {}
    for s in motion:
        old_timing = S23 / "bundles" / s / "producer_timings.jsonl"
        old = [json.loads(x) for x in old_timing.read_text().splitlines()]
        inference[s] = dict(Pi3MOS=read(ROOT/"inputs"/s/"prediction_complete.json"),
                            TSDD_and_flow=[x for x in old if x["log"] in {"step2.log","step3.log"}])
    shutil.copy2(ROOT/"model_identity.json", out/"model_identity.json")
    write(out / "comparison.json", dict(recognition=motion, recognition_winner=motion_winner,
         paired=comparisons, joint=combined, inference=inference))
    keys = sorted({k for r in comparisons for k in r if not isinstance(r[k],dict)})
    with (out / "closed_loop_pairs.tsv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,delimiter="\t",extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    text = ["# S23-P 新旧车辆动静对照", "", f"独立识别判定为 {motion_winner}。", "",
            "| 组别 | ATE 改善 | 平移 RPE 改善 | 静态 PSNR 差值 | LPIPS 差值 | 定位方向 | 建图方向 | 联合采用门 |",
            "|---|---:|---:|---:|---:|---|---|---|"]
    for group,result in combined.items():
        e=result["window_equal"]
        text.append(f"| {group} | {e['ATE_RMSE_m_improvement']:.2%} | {e['RPE_translation_RMSE_m_1frame_improvement']:.2%} | "
                    f"{e['static_RGB_PSNR_dB_delta']:+.3f} dB | {e['LPIPS_delta']:+.4f} | "
                    f"{result['tracking_direction']} | {result['mapping_direction']} | "
                    f"{'通过' if result['passes_joint_adoption'] else '未通过'} |")
    (out/"comparison.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":
    {"recognition":recognition,"summary":summary}[sys.argv[1]](*sys.argv[2:])
