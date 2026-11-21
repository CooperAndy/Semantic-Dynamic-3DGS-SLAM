"""GT-blind, pre-future features for registered capped map updates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

MAP_KEYS = ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.protocol = args.protocol.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_FEATURE_RUN":
        raise ValueError("Wrong feature protocol")
    root = Path(protocol["root"])
    replay = json.loads((root / "G00_S0/receipt.json").read_text(encoding="utf-8"))
    if replay["status"] != "COMPLETED" or sha256(root / "G00_S0/receipt.json") != protocol["replay_receipt_sha256"]:
        raise ValueError("Replay receipt mismatch")
    original = json.loads((root / "protocol_grid_replay_v1.json").read_text(encoding="utf-8"))
    if sha256(root / "protocol_grid_replay_v1.json") != replay["protocol_sha256"]:
        raise ValueError("Replay protocol mismatch")
    manifest_path = root / "inputs/runtime/input_manifest.json"
    pose_path = root / "inputs/runtime/external_orb_P2_c2w.txt"
    if sha256(manifest_path) != original["input_manifest_sha256"] or sha256(pose_path) != protocol["orb_pose_sha256"]:
        raise ValueError("Input mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {int(row["frame_id"]): row for row in manifest["rows"]}
    if protocol["events"] != original["events"]:
        raise ValueError("Event list changed")
    for event in protocol["events"]:
        for snap in (event["before"], event["after"]):
            if sha256(root / "G00_S0" / f"params{snap}.npz") != replay["map_sha256"][str(snap)]:
                raise ValueError(f"Map hash mismatch: {snap}")
        for frame in (event["before"], event["after"]):
            for key in ("rgb", "depth"):
                if sha256(Path(rows[frame][key])) != rows[frame]["hashes"][key]:
                    raise ValueError(f"Input hash mismatch: {frame} {key}")

    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/poses/" in path or "/evaluation_only/" in path:
                raise RuntimeError("GT/evaluation path blocked in feature probe")
    sys.addaudithook(guard)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    source = root / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.recon_helpers import setup_camera
    from external_tracking import render, set_external
    core = load("m15_multi_feature_core", source / "scripts/splatam_external_tm.py")
    sys.path.insert(0, str(root))
    from m12_worker import delta_w2c
    pose_rows = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 3, 4)
    c2w = np.repeat(np.eye(4)[None], len(pose_rows), axis=0)
    c2w[:, :3, :] = pose_rows
    w2c = np.linalg.inv(c2w)
    data = {"cam": setup_camera(manifest["W"], manifest["H"], np.asarray(manifest["K"]), np.eye(4)),
            "w2c": torch.eye(4, device="cuda")}
    cfg = protocol["feature"]
    yy, xx = np.indices((manifest["H"], manifest["W"]))
    stride = (yy % cfg["pixel_stride"] == 0) & (xx % cfg["pixel_stride"] == 0)
    probe_conditions = [
        {"id": "yaw_m1", "kind": "yaw_deg", "value": -1.0},
        {"id": "yaw_p1", "kind": "yaw_deg", "value": 1.0},
        {"id": "tx_m5cm", "kind": "tx_m", "value": -0.05},
        {"id": "tx_p5cm", "kind": "tx_m", "value": 0.05},
    ]
    if cfg["probes"] != probe_conditions:
        raise ValueError("Registered perturbations changed")
    args.out.mkdir(parents=True)
    output = args.out / "features.jsonl"
    started = time.monotonic()

    def observe(frame: int):
        row = rows[frame]
        with Image.open(row["rgb"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
        depth = np.load(row["depth"], allow_pickle=False).astype(np.float32).squeeze()
        return rgb, depth

    def fixed_mask(base, observed_depth):
        mask = np.isfinite(observed_depth) & (observed_depth >= cfg["depth_min_m"]) & (observed_depth <= cfg["depth_max_m"])
        for rgb, ds in base.values():
            mask &= (np.isfinite(rgb).all(axis=0) & np.isfinite(ds).all(axis=0)
                     & (ds[1] > cfg["alpha_threshold"])
                     & (ds[0] >= cfg["depth_min_m"]) & (ds[0] <= cfg["depth_max_m"]))
        return mask & stride

    def measures(rgb, ds, target_rgb, target_depth, mask):
        diff = target_rgb[:, mask] - rgb[:, mask]
        dep = target_depth[mask] - ds[0, mask]
        mse = float(np.mean(diff ** 2))
        return {
            "rgb_psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
            "rgb_mae": float(np.mean(np.abs(diff))),
            "depth_mae_m": float(np.mean(np.abs(dep))),
            "loss_per_pixel": float(cfg["rgb_weight"] * np.abs(diff).sum() / int(mask.sum())
                                    + cfg["depth_weight"] * np.abs(dep).sum() / int(mask.sum())),
            "coverage099": float(np.mean(ds[1] > cfg["alpha_threshold"])),
        }

    with output.open("w", encoding="utf-8") as stream:
        for event in protocol["events"]:
            audit = event["before"]
            current = event["after"]
            maps = {}
            for arm, snap in (("before", audit), ("after", current)):
                with np.load(root / "G00_S0" / f"params{snap}.npz", allow_pickle=False) as archive:
                    params = {key: torch.as_tensor(archive[key], device="cuda") for key in MAP_KEYS}
                    params["cam_unnorm_rots"] = torch.as_tensor(archive["cam_unnorm_rots"], device="cuda").clone()
                    params["cam_trans"] = torch.as_tensor(archive["cam_trans"], device="cuda").clone()
                maps[arm] = params
            result = {"event": event, "before_gaussians": int(len(maps["before"]["means3D"])),
                      "after_gaussians": int(len(maps["after"]["means3D"]))}
            result["point_delta_fraction"] = ((result["after_gaussians"] - result["before_gaussians"])
                                               / result["before_gaussians"])
            for view_name, frame in (("audit", audit), ("current", current)):
                target_rgb, target_depth = observe(frame)
                anchor = torch.as_tensor(w2c[frame], device="cuda", dtype=torch.float32)
                base = {}
                for arm in ("before", "after"):
                    set_external(maps[arm], anchor, frame, core)
                    base[arm] = render(maps[arm], data, frame, core)
                mask = fixed_mask(base, target_depth)
                support = int(mask.sum())
                view = {"fixed_support_pixels": support, "eligible": support >= cfg["min_common_pixels"]}
                if view["eligible"]:
                    view["before"] = measures(*base["before"], target_rgb, target_depth, mask)
                    view["after"] = measures(*base["after"], target_rgb, target_depth, mask)
                    if view_name == "audit":
                        margins = {}
                        for arm in ("before", "after"):
                            losses = []
                            for condition in probe_conditions:
                                disturbed = delta_w2c(condition) @ w2c[frame]
                                set_external(maps[arm], torch.as_tensor(disturbed, device="cuda", dtype=torch.float32), frame, core)
                                rgb, ds = render(maps[arm], data, frame, core)
                                if not (np.isfinite(rgb[:, mask]).all() and np.isfinite(ds[0, mask]).all()):
                                    raise ValueError("Nonfinite probe render")
                                losses.append(measures(rgb, ds, target_rgb, target_depth, mask)["loss_per_pixel"])
                            margins[arm] = float(np.mean(losses) - view[arm]["loss_per_pixel"])
                        view["basin_margin"] = margins
                        result["risk_score_primary"] = margins["before"] - margins["after"]
                result[view_name] = view
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"future": event["future"], "risk": result.get("risk_score_primary"),
                              "audit_support": result["audit"]["fixed_support_pixels"]}), flush=True)
            del maps
            torch.cuda.empty_cache()
    receipt = {
        "status": "COMPLETED", "protocol_sha256": sha256(args.protocol),
        "worker_sha256": sha256(Path(__file__)),
        "replay_receipt_sha256": sha256(root / "G00_S0/receipt.json"),
        "features_sha256": sha256(output), "rows": len(protocol["events"]),
        "wall_seconds": time.monotonic() - started, "GT_access": "audit_hook_blocks_GT_paths",
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
