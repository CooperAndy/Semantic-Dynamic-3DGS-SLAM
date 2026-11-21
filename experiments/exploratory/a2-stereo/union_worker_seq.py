"""三重组合 worker：门控 0.6 + 240 步 + 6000 点。

目的：确认"门控 0.6"能不能在"240 步 + 6000 点"之上再加收益（按容量互相替代的规律，预期很小）。

与 combo_worker 的唯一差别：数据走 `gate06` 的门控清单（T=0.6），
通过为 gate06 写一份 data_cfg.yaml 并把 `cfg['data']['gradslam_data_cfg']` 指过去实现
（mapper 用 ManifestDataset 读 yaml 里的 input_manifest）。

用法：python combo_gate_worker.py <steps> <cap> <count> <seed> <gate_tag>
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PARENT = Path("/root/autodl-tmp/A2_union_20261029")
SEQ = sys.argv[7]
assert SEQ in ("03", "08")
ROOT = PARENT / f"seq{SEQ}"
SOURCE = PARENT / "source"
BASE = Path("/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200")
BASE_MANIFEST = Path("/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1/input_manifest.json") if SEQ == "03" else BASE / "inputs/runtime/input_manifest.json"
FB = 384.37173985194534 if SEQ == "03" else 378.6112506585076
DEPTH_ROOT = Path("/root/autodl-tmp/E2_depth")

STEPS = int(sys.argv[1])
CAP = int(sys.argv[2])
COUNT = int(sys.argv[3])
SEED = int(sys.argv[4])
GATE = sys.argv[5]
ARM = sys.argv[6]
assert ARM in ("C0", "C2")
assert STEPS in (60, 240) and CAP in (3000, 6000) and COUNT in (25, 200)
assert GATE in ("06", "08")

MANIFEST = (PARENT / f"seq{SEQ}_gate06" / "manifest.json") if GATE == "06" else BASE_MANIFEST
OUT = ROOT / "runs" / f"{ARM}_g{GATE}_{STEPS}_{CAP}_{COUNT}_s{SEED}"
OUT.mkdir(parents=True, exist_ok=False)

start = time.monotonic()
sys.path.insert(0, "/root/autodl-tmp/A2_stereo_robust_geometry/factorial_v1")
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)
torch.set_num_threads(2)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


manifest = json.loads(MANIFEST.read_text())
rows = {r["frame_id"]: r for r in manifest["rows"]}
if sha(manifest["pose_file"]) != manifest["pose_sha256"]:
    raise RuntimeError("pose mismatch")
for i in range(COUNT):
    for k in ("rgb", "depth", "confidence"):
        if sha(rows[i][k]) != rows[i]["hashes"][k]:
            raise RuntimeError(f"input mismatch {i} {k}")

cfg = load("combo_gate_config", BASE / "source/configs/kitti/G00_S0.py").config
cfg["workdir"] = str(ROOT / "runs")
cfg["run_name"] = OUT.name
cfg["seed"] = SEED
cfg["data"]["num_frames"] = COUNT
cfg["mapping"]["num_iters"] = STEPS
cfg["mapping"]["residual_fb"] = FB
cfg["mapping"]["sampling_seed"] = SEED
if SEQ == "03":
    cfg["data"].update(sequence="03", desired_image_height=375, desired_image_width=1242, gradslam_data_cfg="/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1/seed_0/data.yaml")
cfg["eval_only_frames"] = [i for i in cfg["eval_only_frames"] if i < COUNT]

# Only 0.6 changes the manifest. Original 0.8 uses the original configuration.
if GATE == "06":
    _src = Path(cfg["data"]["gradslam_data_cfg"])
    _dst = ROOT / "gate06_data_cfg.yaml"
    _txt = _src.read_text()
    _txt, nsub = re.subn(r"(?m)^input_manifest:.*$", f"input_manifest: {MANIFEST}", _txt)
    if nsub != 1: raise RuntimeError("invalid manifest yaml")
    if not _dst.exists(): _dst.write_text(_txt)
    if _dst.read_text() != _txt: raise RuntimeError("data cfg mismatch")
    cfg["data"]["gradslam_data_cfg"] = str(_dst)
print("DATASET_CFG", cfg["data"]["gradslam_data_cfg"], flush=True)

heldout = set(cfg["eval_only_frames"])
core = load("combo_gate_core", SOURCE / "scripts/splatam_external_tm.py")
core.seed_everything(SEED)

from c04n_addition import add_new_gaussians as selected_addition
import utils.slam_external as external
from external_tracking import map_hash

state = {"births": 0, "deleted": 0, "prune_events": 0, "initial_hash": None,
         "loss_frames": {}, "additions": [], "calls": 0}
holder = [None]


def assert_len(p, v):
    assert len(v["stable_id"]) == len(p["means3D"])


init = core.initialize_first_timestep


def init_hook(*a, **kw):
    result = init(*a, **kw)
    p, v = result[:2]
    n = len(p["means3D"])
    v["stable_id"] = torch.arange(n, device="cuda", dtype=torch.int64)
    state["births"] = n
    state["initial_hash"] = map_hash(p)
    state["initial_points"] = n
    holder[0] = v
    assert_len(p, v)
    return result


core.initialize_first_timestep = init_hook


def add_hook(*a, **kw):
    p, v = a[:2]
    frame = int(a[4])
    assert frame not in heldout
    before = len(p["means3D"])
    p, v = selected_addition(*a, **kw, core=core, mode="native", cap=CAP, seed=SEED,
                             log_path=OUT / "addition_history.jsonl", fb=cfg["mapping"]["residual_fb"])
    count = len(p["means3D"]) - before
    if count > CAP:
        raise RuntimeError(f"budget exceeded frame={frame}")
    state["additions"].append({"frame": frame, "actual": count})
    v["stable_id"] = torch.cat((v["stable_id"], torch.arange(
        state["births"], state["births"] + count, device="cuda", dtype=torch.int64)))
    state["births"] += count
    holder[0] = v
    assert_len(p, v)
    return p, v


core.add_new_gaussians = add_hook
remove = external.remove_points


def remove_hook(mask, p, v, opt):
    saved = v["stable_id"][~mask].clone()
    state["deleted"] += int(mask.sum())
    state["prune_events"] += 1
    p, v = remove(mask, p, v, opt)
    v["stable_id"] = saved
    holder[0] = v
    assert_len(p, v)
    return p, v


external.remove_points = remove_hook
orig_loss = core.get_loss


def loss_hook(*a, **kw):
    p, data, v, frame = a[:4]
    if kw.get("mapping"):
        online = int(sys._getframe(1).f_locals["time_idx"])
        assert online not in heldout and int(frame) not in heldout
        state["loss_frames"][str(online)] = state["loss_frames"].get(str(online), 0) + 1
        if ARM == "C2":
            data["a2_mode"] = "C2"
            data["a2_fb"] = FB
            data["a2_confidence"] = data["depth"]
    state["calls"] += 1
    result = orig_loss(*a, **kw)
    if not torch.isfinite(result[0]):
        raise RuntimeError("nonfinite loss")
    if time.monotonic() - start > 21600:
        raise RuntimeError("time budget")
    return result


core.get_loss = loss_hook

try:
    core.rgbd_slam(cfg)
    torch.cuda.synchronize()
    z = np.load(OUT / "params.npz")
    n = len(z["means3D"])
    assert n == state["births"] - state["deleted"]
    assert sum(state["loss_frames"].values()) == STEPS * (COUNT - len(heldout))
    assert all(x == STEPS for x in state["loss_frames"].values())
    assert not (set(map(int, state["loss_frames"])) & heldout)
    adds = [a["actual"] for a in state["additions"]]
    assert len(adds) == COUNT - len(heldout) - 1

    colors = np.asarray(z["rgb_colors"], dtype=np.float32)
    receipt = {
        "status": "PASS_ENGINEERING", "arm": ARM, "steps": STEPS, "cap": CAP, "gate": GATE,
        "frames": COUNT, "seed": SEED,
        "initial_hash": state["initial_hash"], "initial_points": state.get("initial_points"),
        "births_total": state["births"], "deleted": state["deleted"], "survivors": n,
        "prune_events": state["prune_events"], "n_additions": len(adds),
        "addition_total": int(sum(adds)), "addition_mean": float(np.mean(adds)),
        "addition_min": int(min(adds)),
        "budget_utilization": float(sum(adds) / (CAP * len(adds))),
        "mapping_loss_per_frame": STEPS, "holdout": sorted(heldout),
        "seconds": time.monotonic() - start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "map_sha256": sha(OUT / "params.npz"),
        "final_out_of_range_frac": float(np.mean((colors < 0) | (colors > 1))),
        "quality_adoption": False,
    }
    (OUT / "combo_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("COMBO3_COMPLETE", json.dumps({k: receipt[k] for k in (
        "status", "gate", "steps", "cap", "seed", "initial_points", "survivors",
        "addition_mean", "seconds", "peak_cuda_bytes")}))
except BaseException as e:
    (OUT / "combo_receipt.json").write_text(json.dumps(
        {"status": "FAILED_PRESERVED", "arm": ARM, "steps": STEPS, "cap": CAP, "gate": GATE, "seed": SEED,
         "error": repr(e), "traceback": traceback.format_exc(), "state": state}, indent=2), encoding="utf-8")
    raise
