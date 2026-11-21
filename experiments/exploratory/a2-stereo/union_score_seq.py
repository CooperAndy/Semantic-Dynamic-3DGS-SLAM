"""三重组合评分：门控 0.6 能不能在「240 步 + 6000 点」之上再加收益。

四个臂（同窗口、同 3 种子）：
  A00  = T0.8 / 60 步 / 3000 点   → E2_iter/runs/iter60_200_s<S>
  A06  = T0.6 / 60 步 / 3000 点   → E2_depth/runs/gate06_200_s<S>
  A11  = T0.8 / 240 步 / 6000 点  → E2_combo/runs/combo_240_6000_200_s<S>
  A111 = T0.6 / 240 步 / 6000 点  → E2_combo3/runs/combo_g06_240_6000_200_s<S>   ← 本轮新跑

用法：python combo3_score.py <count> <seed>
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEQ = sys.argv[3]
assert SEQ in ("03", "08")
BASE = Path("/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200")
BASE_MANIFEST = Path("/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1/input_manifest.json") if SEQ == "03" else BASE / "inputs/runtime/input_manifest.json"
ITER = Path("/root/autodl-tmp/E2_iter")
DEPTH = Path("/root/autodl-tmp/E2_depth")
COMBO = Path("/root/autodl-tmp/E2_combo")
COMBO3 = Path("/root/autodl-tmp/A2_union_20261029") / f"seq{SEQ}"

COUNT = int(sys.argv[1])
SEED = int(sys.argv[2])
assert COUNT in (25, 200), COUNT

ARMS = {
    "B": (COMBO3 / "runs" / f"C0_g08_60_3000_{COUNT}_s{SEED}", "combo_receipt.json"),
    "A": (COMBO3 / "runs" / f"C2_g08_60_3000_{COUNT}_s{SEED}", "combo_receipt.json"),
    "F": (COMBO3 / "runs" / f"C0_g06_240_6000_{COUNT}_s{SEED}", "combo_receipt.json"),
    "AF": (COMBO3 / "runs" / f"C2_g06_240_6000_{COUNT}_s{SEED}", "combo_receipt.json"),
}

frames = list(range(19, COUNT, 20))


def sha(p):
    import hashlib

    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    m = json.loads(BASE_MANIFEST.read_text())
    rows = {r["frame_id"]: r for r in m["rows"]}
    src = Path("/root/autodl-tmp/E2_appearance/source")
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("c3_core", src / "scripts/splatam_external_tm.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    from external_tracking import render
    from utils.recon_helpers import setup_camera
    from utils.slam_external import calc_ssim

    maps, checks, ref = {}, {}, None
    for name, (path, rname) in ARMS.items():
        rec = json.loads((path / rname).read_text())
        p = path / "params.npz"
        if rec["status"] != "PASS_ENGINEERING" or sha(p) != rec["map_sha256"] or rec["seed"] != SEED:
            raise RuntimeError(f"unsealed map {name}")
        with np.load(p) as z:
            assert not np.isin(z["timestep"], frames).any()
            meta = {k: z[k].copy() for k in ("cam_unnorm_rots", "cam_trans", "intrinsics", "w2c")}
            if ref is None:
                ref = meta
            else:
                for k in meta:
                    if not np.array_equal(meta[k], ref[k]):
                        raise RuntimeError("camera mismatch " + k)
            maps[name] = {k: torch.tensor(z[k], device="cuda") for k in
                          ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales")}
            maps[name].update({k: torch.tensor(ref[k], device="cuda") for k in ("cam_unnorm_rots", "cam_trans")})
        checks[name] = {"sha256": rec["map_sha256"], "survivors": rec["survivors"],
                        "seconds": rec["seconds"], "steps": rec.get("steps"),
                        "cap": rec.get("cap"), "gate": rec.get("gate", "08"),
                        "initial_points": rec.get("initial_points")}

    data = {"cam": setup_camera(m["W"], m["H"], ref["intrinsics"], ref["w2c"]),
            "w2c": torch.tensor(ref["w2c"], device="cuda")}
    result = []
    for f in frames:
        row = rows[f]
        with Image.open(row["rgb"]) as im:
            target = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
        d = np.load(row["depth"])
        cf = np.load(row["confidence"])
        common = np.isfinite(d) & (d > 0.5) & (d < 80) & np.isfinite(cf) & (cf >= 0.8)
        renders = {}
        for name in ARMS:
            rgb, ds = render(maps[name], data, f, core)
            if not np.isfinite(rgb).all():
                raise RuntimeError("nonfinite render")
            depth = ds[0] / np.maximum(ds[1], 1e-6)
            common &= (ds[1] > 0.99) & np.isfinite(depth) & (depth > 0.5) & (depth < 80)
            renders[name] = (rgb, ds, depth)
            export = COMBO3 / f"render_s{SEED}"
            export.mkdir(exist_ok=True)
            np.savez_compressed(export / f"{f:03d}_{name}.npz", depth=depth.astype(np.float32), alpha=ds[1].astype(np.float32))
        fr = {"frame": f, "arms": {}}
        for name, (rgb, ds, depth) in renders.items():
            pred = np.clip(rgb, 0, 1)
            mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
            with torch.no_grad():
                ssim = float(calc_ssim(torch.tensor(pred).unsqueeze(0), torch.tensor(target).unsqueeze(0)))
            fr["arms"][name] = {
                "clipped_psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
                "ssim": ssim,
                "alpha099_coverage": float((ds[1] > 0.99).mean()),
                "out_of_range_frac": float(np.mean((rgb < 0) | (rgb > 1))),
                "depth_proxy_common_mae_m": (float(np.mean(np.abs(depth[common] - d[common]))) if common.any() else None),
            }
        result.append(fr)

    means = {}
    for name in ARMS:
        means[name] = {k: float(np.mean([r["arms"][name][k] for r in result]))
                       for k in ("clipped_psnr_db", "ssim", "alpha099_coverage", "out_of_range_frac")}
        vals = [r["arms"][name]["depth_proxy_common_mae_m"] for r in result
                if r["arms"][name]["depth_proxy_common_mae_m"] is not None]
        means[name]["depth_proxy_common_mae_m"] = float(np.mean(vals)) if vals else None
        means[name].update(checks[name])

    report = {"status": "COMPLETE", "seed": SEED, "frames": result, "means": means, "checks": checks}
    (COMBO3 / f"evaluation_{COUNT}_s{SEED}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name in ARMS:
        (COMBO3 / f"eval_{name}_s{SEED}.json").write_text(json.dumps({
            "arm": name, "seed": str(SEED),
            "psnr_clip_db_mean": means[name]["clipped_psnr_db"],
            "coverage_mean": means[name]["alpha099_coverage"],
            "ssim_mean": means[name]["ssim"],
            "survivors": means[name]["survivors"],
            "seconds": means[name]["seconds"],
        }, indent=2), encoding="utf-8")

    print(json.dumps({k: means[k]["clipped_psnr_db"] for k in ARMS}, indent=2))

if __name__ == "__main__":
    main()
