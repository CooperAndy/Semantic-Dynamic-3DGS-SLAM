"""Frozen-map paired camera recovery on G11 evaluation-only views; no GT access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def delta_w2c(condition: dict) -> np.ndarray:
    kind, value = condition["kind"], float(condition["value"])
    t = np.zeros(3)
    omega = np.zeros(3)
    if kind == "tx_m":
        t[0] = value
    elif kind == "tz_m":
        t[2] = value
    elif kind == "yaw_deg":
        omega[1] = math.radians(value)
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    theta = np.linalg.norm(omega)
    cross = np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]], dtype=np.float64)
    rotation = (np.eye(3) - cross + 0.5 * cross @ cross if theta < 1e-7
                else np.eye(3) - np.sin(theta) / theta * cross
                + (1 - np.cos(theta)) / theta**2 * cross @ cross)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -t
    return transform


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--g11-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one frame, one perturbation, two maps, five steps")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.out}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest_path = args.g11_root / "inputs/runtime/input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = ("A0", "A1")
    map_paths = {arm: args.g11_root / "runs" / protocol["maps"][arm]["variant"] / "params.npz"
                 for arm in arms}
    for arm in arms:
        actual = sha256(map_paths[arm])
        if actual != protocol["maps"][arm]["sha256"]:
            raise ValueError(f"{arm} map hash mismatch: {actual}")
    frames = protocol["frames"][:1] if args.smoke else protocol["frames"]
    conditions = [next(c for c in protocol["perturbations"] if c["id"] == "yaw_p1")] if args.smoke else protocol["perturbations"]
    rows = {int(row["frame_id"]): row for row in manifest["rows"]}
    for frame in frames:
        if frame not in manifest["mapping_holdout_frames"]:
            raise ValueError(f"Frame {frame} was not held out from mapping")
        for key in ("rgb", "depth"):
            source = Path(rows[frame][key])
            if sha256(source) != rows[frame]["hashes"][key]:
                raise ValueError(f"Input hash mismatch: frame={frame}, key={key}")

    # Match G11 prediction process: no GT/pose-truth data can be opened here.
    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/poses/" in path or "/evaluation_only/" in path:
                raise RuntimeError("GT path blocked in M1-2 prediction worker")
    sys.addaudithook(guard)

    import torch
    import torch.nn.functional as F
    source = args.g11_root / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.recon_helpers import setup_camera
    core = load_module("m12_g11_core", source / "scripts/splatam_external_tm.py")
    from external_tracking import render, set_external

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.manual_seed(protocol["optimization"]["seed"])
    if protocol["optimization"]["loss"] != "sum_of_absolute_RGB_and_E6a_depth_residual_on_common_fixed_support":
        raise ValueError("Unexpected protocol loss")
    cfg = protocol["optimization"]
    steps = 5 if args.smoke else cfg["steps"]
    with np.load(map_paths["A0"], allow_pickle=False) as a, np.load(map_paths["A1"], allow_pickle=False) as b:
        if not (np.array_equal(a["external_w2c_all_frames"], b["external_w2c_all_frames"])
                and np.array_equal(a["intrinsics"], b["intrinsics"])):
            raise ValueError("A0/A1 camera or intrinsics mismatch")
        anchor_poses = a["external_w2c_all_frames"][frames].copy()
        k = a["intrinsics"].copy()
        if not np.allclose(k, np.asarray(manifest["K"]), atol=1e-3, rtol=0):
            raise ValueError("Intrinsics differ from G11 manifest")
        params = {}
        for arm, archive in (("A0", a), ("A1", b)):
            if not set(archive["eval_only_frames"]) >= set(frames):
                raise ValueError(f"{arm} missing held-out frame")
            params[arm] = {key: torch.as_tensor(archive[key], device="cuda") for key in MAP_KEYS}
            params[arm]["cam_unnorm_rots"] = torch.nn.Parameter(torch.as_tensor(
                archive["cam_unnorm_rots"], device="cuda").clone())
            params[arm]["cam_trans"] = torch.nn.Parameter(torch.as_tensor(
                archive["cam_trans"], device="cuda").clone())
            if any(params[arm][key].requires_grad for key in MAP_KEYS):
                raise RuntimeError("Map must remain frozen")
    data = {"cam": setup_camera(manifest["W"], manifest["H"], k, np.eye(4)),
            "w2c": torch.eye(4, device="cuda")}
    args.out.mkdir(parents=True)
    receipt = {
        "status": "RUNNING", "scope": protocol["scope"], "smoke": args.smoke,
        "protocol_sha256": sha256(args.protocol), "input_manifest_sha256": sha256(manifest_path),
        "map_sha256": {arm: sha256(path) for arm, path in map_paths.items()},
        "frames": frames, "condition_ids": [x["id"] for x in conditions],
        "expected_trials": len(frames) * len(conditions) * len(arms),
        "steps": steps, "gaussians": {arm: int(len(params[arm]["means3D"])) for arm in arms},
        "GT_access": "audit_hook_blocks_GT_paths", "source": str(source),
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    trial_file = args.out / "trials.jsonl"
    started = time.monotonic()

    def current_pose(p: dict, index: int) -> np.ndarray:
        with torch.no_grad():
            q = F.normalize(p["cam_unnorm_rots"][..., index].detach())
            rotation = core.build_rotation(q)[0]
            pose = torch.eye(4, device="cuda")
            pose[:3, :3] = rotation
            pose[:3, 3] = p["cam_trans"][..., index].detach()
            return pose.cpu().numpy().astype(np.float64)

    for frame, anchor in zip(frames, anchor_poses):
        row = rows[frame]
        with Image.open(row["rgb"]) as image:
            observed = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
        target_depth = np.load(row["depth"], allow_pickle=False).astype(np.float32)
        if target_depth.ndim == 3:
            target_depth = target_depth.squeeze()
        if target_depth.shape != (manifest["H"], manifest["W"]):
            raise ValueError("Unexpected target depth shape")
        baseline = {}
        for arm in arms:
            set_external(params[arm], torch.as_tensor(anchor, device="cuda", dtype=torch.float32), frame, core)
            baseline[arm] = render(params[arm], data, frame, core)
        rgb_valid = np.isfinite(baseline["A0"][0]).all(axis=0) & np.isfinite(baseline["A1"][0]).all(axis=0)
        ds0, ds1 = baseline["A0"][1], baseline["A1"][1]
        mask = (np.isfinite(target_depth) & (target_depth >= 3) & (target_depth <= 60)
                & np.isfinite(ds0).all(axis=0) & np.isfinite(ds1).all(axis=0)
                & (ds0[1] > .99) & (ds1[1] > .99) & rgb_valid)
        yy, xx = np.indices(mask.shape)
        mask &= (yy % cfg["pixel_stride"] == 0) & (xx % cfg["pixel_stride"] == 0)
        mask_count = int(mask.sum())
        support_record = {"frame": frame, "source_frame_id": row["source_frame_id"],
                          "common_fixed_support_pixels": mask_count,
                          "baseline_coverage099": {arm: float((baseline[arm][1][1] > .99).mean()) for arm in arms},
                          "depth_valid_pixels": int(np.count_nonzero((target_depth >= 3) & (target_depth <= 60))),
                          "stride": cfg["pixel_stride"]}
        append_jsonl(args.out / "support.jsonl", support_record)
        if mask_count < cfg["min_common_pixels"]:
            for condition in conditions:
                for arm in arms:
                    append_jsonl(trial_file, {"status": "SKIPPED_LOW_COMMON_SUPPORT", "frame": frame,
                                              "source_frame_id": row["source_frame_id"],
                                              "condition": condition, "arm": arm,
                                              "common_fixed_support_pixels": mask_count})
            continue
        fixed_mask = torch.as_tensor(mask, device="cuda")
        target_im = torch.as_tensor(observed, device="cuda")
        target_dep = torch.as_tensor(target_depth, device="cuda")
        for condition_number, condition in enumerate(conditions):
            init_pose = delta_w2c(condition) @ anchor
            arm_order = arms if condition_number % 2 == 0 else tuple(reversed(arms))
            for arm in arm_order:
                p = params[arm]
                set_external(p, torch.as_tensor(init_pose, device="cuda", dtype=torch.float32), frame, core)
                optimizer = torch.optim.Adam([
                    {"params": [p["cam_unnorm_rots"]], "lr": cfg["lr_camera_quaternion"]},
                    {"params": [p["cam_trans"]], "lr": cfg["lr_camera_translation"]},
                ])
                best_loss, best_pose, best_step = float("inf"), None, None
                losses = []
                torch.cuda.synchronize()
                trial_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                status = "COMPLETED"
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    transformed = core.transform_to_frame(p, frame, gaussians_grad=False, camera_grad=True)
                    rgb_var = core.transformed_params2rendervar(p, transformed)
                    depth_var = core.transformed_params2depthplussilhouette(p, data["w2c"], transformed)
                    rgb_pred, _, _ = core.Renderer(raster_settings=data["cam"])(**rgb_var)
                    depth_sil, _, _ = core.Renderer(raster_settings=data["cam"])(**depth_var)
                    loss_rgb = torch.abs(target_im - rgb_pred)[:, fixed_mask].sum()
                    loss_depth = torch.abs(target_dep - depth_sil[0])[fixed_mask].sum()
                    loss = cfg["rgb_weight"] * loss_rgb + cfg["depth_weight"] * loss_depth
                    if not torch.isfinite(loss).item():
                        status = "NONFINITE_LOSS"
                        break
                    value = float(loss.detach())
                    losses.append(value)
                    if value < best_loss:
                        best_loss, best_pose, best_step = value, current_pose(p, frame), step
                    loss.backward()
                    gradient_ok = all(x.grad is not None and torch.isfinite(x.grad[..., frame]).all().item()
                                      for x in (p["cam_unnorm_rots"], p["cam_trans"]))
                    if not gradient_ok:
                        status = "NONFINITE_OR_MISSING_CAMERA_GRADIENT"
                        break
                    optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - trial_started
                append_jsonl(trial_file, {
                    "status": status, "frame": frame, "source_frame_id": row["source_frame_id"],
                    "condition": condition, "arm": arm,
                    "common_fixed_support_pixels": mask_count,
                    "anchor_w2c": anchor.tolist(), "initial_w2c": init_pose.tolist(),
                    "selected_w2c": best_pose.tolist() if best_pose is not None else None,
                    "selected_pre_step": best_step, "steps_requested": steps,
                    "steps_completed": len(losses), "initial_loss": losses[0] if losses else None,
                    "selected_loss": best_loss if best_pose is not None else None,
                    "last_pre_step_loss": losses[-1] if losses else None,
                    "wall_seconds": elapsed,
                    "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                })
                del optimizer
        torch.cuda.empty_cache()
    rows_written = sum(1 for _ in trial_file.open(encoding="utf-8"))
    receipt.update(status="COMPLETED", trials_written=rows_written,
                   wall_seconds=time.monotonic() - started,
                   trials_sha256=sha256(trial_file),
                   support_sha256=sha256(args.out / "support.jsonl"))
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "trials": rows_written,
                      "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    main()
