"""GT-blind support screen for temporally future grid frames."""

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


MAP_KEYS = ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_RENDER":
        raise ValueError("Unexpected protocol status")
    root = Path(protocol["g08_root"])
    manifest_file = root / "inputs/runtime/input_manifest.json"
    pose_file = root / "inputs/runtime/external_orb_P2_c2w.txt"
    if sha256(manifest_file) != protocol["input_manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    if sha256(pose_file) != protocol["orb_pose_sha256"]:
        raise ValueError("ORB pose hash mismatch")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["frame_count"] != 1000:
        raise ValueError("Unexpected frame count")
    views = sorted({frame for t in protocol["transitions"] for frame in t["future_holdouts"]})
    rows = {int(row["frame_id"]): row for row in manifest["rows"]}
    for transition in protocol["transitions"]:
        if transition["after"] <= transition["before"]:
            raise ValueError("Checkpoints not time ordered")
        for frame in transition["future_holdouts"]:
            if frame != transition["after"] + 1 or frame >= protocol["num_frames"]:
                raise ValueError(f"Frame {frame} is not the registered next unseen frame")
    for frame in views:
        if sha256(Path(rows[frame]["depth"])) != rows[frame]["hashes"]["depth"]:
            raise ValueError(f"Depth hash mismatch at {frame}")
    maps = {}
    for variant, snapshots in protocol["variants"].items():
        maps[variant] = {}
        for snap, expected in snapshots.items():
            path = root / "runs" / variant / f"params{snap}.npz"
            if sha256(path) != expected:
                raise ValueError(f"Map hash mismatch: {variant} {snap}")
            maps[variant][snap] = path

    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/evaluation_only/" in path or "/poses/" in path:
                raise RuntimeError("GT path blocked in support probe")
    sys.addaudithook(guard)

    import torch
    import diff_gaussian_rasterization as rasterizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA rasterizer required")
    torch.set_num_threads(4)
    source = root / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.recon_helpers import setup_camera
    from external_tracking import render, set_external
    core = load_module("m15_g09_core", source / "scripts/splatam_external_tm.py")

    pose_rows = np.loadtxt(pose_file, dtype=np.float64).reshape(-1, 3, 4)
    if len(pose_rows) != manifest["frame_count"]:
        raise ValueError("ORB pose count mismatch")
    c2w = np.repeat(np.eye(4)[None], len(pose_rows), axis=0)
    c2w[:, :3, :] = pose_rows
    w2c = np.linalg.inv(c2w)
    settings = protocol["support"]
    stride = int(settings["pixel_stride"])
    yy, xx = np.indices((manifest["H"], manifest["W"]))
    subsample = (yy % stride == 0) & (xx % stride == 0)
    depth_valid = {}
    for frame in views:
        depth = np.load(rows[frame]["depth"], allow_pickle=False).astype(np.float32).squeeze()
        if depth.shape != subsample.shape:
            raise ValueError("Depth shape mismatch")
        depth_valid[frame] = (np.isfinite(depth)
                              & (depth >= settings["depth_min_m"])
                              & (depth <= settings["depth_max_m"]))
    data = {"cam": setup_camera(manifest["W"], manifest["H"], np.asarray(manifest["K"]), np.eye(4)),
            "w2c": torch.eye(4, device="cuda")}
    required = {snap: set() for snap in next(iter(protocol["variants"].values()))}
    for transition in protocol["transitions"]:
        for snap in (str(transition["before"]), str(transition["after"])):
            required[snap].update(transition["future_holdouts"])
    results = {}
    started = time.monotonic()
    for variant, snapshots in maps.items():
        results[variant] = {}
        for snap, path in snapshots.items():
            with np.load(path, allow_pickle=False) as archive:
                params = {key: torch.as_tensor(archive[key], device="cuda") for key in MAP_KEYS}
                params["cam_unnorm_rots"] = torch.as_tensor(archive["cam_unnorm_rots"], device="cuda").clone()
                params["cam_trans"] = torch.as_tensor(archive["cam_trans"], device="cuda").clone()
            for frame in sorted(required[snap]):
                set_external(params, torch.as_tensor(w2c[frame], device="cuda", dtype=torch.float32), frame, core)
                rgb, ds = render(params, data, frame, core)
                valid = (np.isfinite(rgb).all(axis=0) & np.isfinite(ds).all(axis=0)
                         & (ds[1] > settings["alpha_threshold"])
                         & (ds[0] >= settings["depth_min_m"])
                         & (ds[0] <= settings["depth_max_m"]))
                results[variant][(snap, frame)] = {
                    "valid": valid,
                    "coverage099": float((ds[1] > settings["alpha_threshold"]).mean()),
                    "valid_depth_stride_pixels": int((valid & depth_valid[frame] & subsample).sum()),
                    "gaussians": len(params["means3D"]),
                }
                print(json.dumps({"variant": variant, "snapshot": snap, "frame": frame,
                                  "coverage099": results[variant][(snap, frame)]["coverage099"]}), flush=True)
            del params
            torch.cuda.empty_cache()

    args.out.mkdir(parents=True)
    output_file = args.out / "support.jsonl"
    with output_file.open("w", encoding="utf-8") as stream:
        for variant in protocol["variants"]:
            for transition in protocol["transitions"]:
                before, after = str(transition["before"]), str(transition["after"])
                for frame in transition["future_holdouts"]:
                    old = results[variant][(before, frame)]
                    new = results[variant][(after, frame)]
                    common = int((old["valid"] & new["valid"]
                                  & depth_valid[frame] & subsample).sum())
                    row = {"variant": variant, "before": int(before), "after": int(after),
                           "frame": frame, "source_frame_id": rows[frame]["source_frame_id"],
                           "before_gaussians": old["gaussians"], "after_gaussians": new["gaussians"],
                           "before_coverage099": old["coverage099"],
                           "after_coverage099": new["coverage099"],
                           "before_valid_depth_stride_pixels": old["valid_depth_stride_pixels"],
                           "after_valid_depth_stride_pixels": new["valid_depth_stride_pixels"],
                           "common_fixed_support_pixels": common,
                           "pose_probe_eligible": common >= settings["min_common_pixels_for_pose_probe"]}
                    stream.write(json.dumps(row) + "\n")
    receipt = {"status": "COMPLETED", "scope": protocol["scope"],
               "protocol_sha256": sha256(args.protocol), "runner_sha256": sha256(Path(__file__)),
               "input_manifest_sha256": sha256(manifest_file), "orb_pose_sha256": sha256(pose_file),
               "rasterizer_sha256": sha256(Path(rasterizer._C.__file__)),
               "map_sha256": protocol["variants"],
               "support_sha256": sha256(output_file),
               "rows": sum(1 for _ in output_file.open(encoding="utf-8")),
               "wall_seconds": time.monotonic() - started,
               "GT_access": "audit_hook_blocks_GT_paths"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"],
                      "seconds": receipt["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
