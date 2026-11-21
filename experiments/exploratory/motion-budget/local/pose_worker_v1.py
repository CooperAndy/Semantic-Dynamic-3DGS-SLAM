"""GT-blind, paired future-view local 6DoF recovery on adjacent map snapshots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m12_worker import MAP_KEYS, append_jsonl, delta_w2c, load_module, sha256
from pose_update import compose_local_increment, insert_camera_at_frame


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
    if protocol["status"] != "FROZEN_BEFORE_POSE_RUN":
        raise ValueError("Protocol status differs")
    root = Path(protocol["root"])
    if sha256(root / "G00_S0/receipt.json") != protocol["replay_receipt_sha256"]:
        raise ValueError("Replay receipt mismatch")
    manifest_path = root / "inputs/runtime/input_manifest.json"
    pose_path = root / "inputs/runtime/external_orb_P2_c2w.txt"
    support_dir = root / "support_v1"
    if sha256(support_dir / "receipt.json") != protocol["support_receipt_sha256"]:
        raise ValueError("Support receipt mismatch")
    support_receipt = json.loads((support_dir / "receipt.json").read_text(encoding="utf-8"))
    if support_receipt["status"] != "COMPLETED" or sha256(support_dir / "support.jsonl") != support_receipt["support_sha256"]:
        raise ValueError("Support screen is not sealed")
    support_rows = [json.loads(line) for line in (support_dir / "support.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["pose_probe_eligible"]]
    if len(support_rows) != len(protocol["pairs"]):
        raise ValueError("Support row count mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != support_receipt["input_manifest_sha256"] or sha256(pose_path) != support_receipt["orb_pose_sha256"]:
        raise ValueError("Input version mismatch")
    rows = {int(row["frame_id"]): row for row in manifest["rows"]}
    map_paths = {}
    for pair, support in zip(protocol["pairs"], support_rows):
        if (pair["before"], pair["after"], pair["frame"], pair["support_pixels"]) != (
            support["before"], support["after"], support["frame"], support["common_fixed_support_pixels"]
        ) or not support["pose_probe_eligible"]:
            raise ValueError("Support pair mismatch or ineligible")
        if pair["frame"] not in manifest["mapping_holdout_frames"] or pair["frame"] <= pair["after"]:
            raise ValueError("Not a future mapping holdout")
        for snap in (pair["before"], pair["after"]):
            path = root / "G00_S0" / f"params{snap}.npz"
            if sha256(path) != support_receipt["map_sha256"]["G00_S0"][str(snap)]:
                raise ValueError(f"Map hash mismatch: {snap}")
            map_paths[snap] = path
        for key in ("rgb", "depth"):
            if sha256(Path(rows[pair["frame"]][key])) != rows[pair["frame"]]["hashes"][key]:
                raise ValueError(f"Frame {pair['frame']} {key} hash mismatch")

    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/poses/" in path or "/evaluation_only/" in path:
                raise RuntimeError("GT/evaluation path blocked in pose worker")
    sys.addaudithook(guard)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    cfg = protocol["optimization"]
    torch.manual_seed(int(cfg["seed"]))
    source = root / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.recon_helpers import setup_camera
    from external_tracking import render, set_external
    core = load_module("m15_dense_pose_core", source / "scripts/splatam_external_tm.py")
    pose_rows = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 3, 4)
    if len(pose_rows) != manifest["frame_count"]:
        raise ValueError("ORB pose count mismatch")
    c2w = np.repeat(np.eye(4)[None], len(pose_rows), axis=0)
    c2w[:, :3, :] = pose_rows
    w2c = np.linalg.inv(c2w)
    data = {"cam": setup_camera(manifest["W"], manifest["H"], np.asarray(manifest["K"]), np.eye(4)),
            "w2c": torch.eye(4, device="cuda")}
    args.out.mkdir(parents=True)
    receipt_path = args.out / "receipt.json"
    trials_path = args.out / "trials.jsonl"
    quality_path = args.out / "anchor_quality.jsonl"
    receipt = {
        "status": "RUNNING", "phase": protocol["phase"],
        "protocol_sha256": sha256(args.protocol), "worker_sha256": sha256(Path(__file__)),
        "m12_worker_sha256": sha256(HERE / "m12_worker.py"),
        "pose_update_sha256": sha256(HERE / "pose_update.py"),
        "manifest_sha256": sha256(manifest_path), "orb_pose_sha256": sha256(pose_path),
        "support_receipt_sha256": sha256(support_dir / "receipt.json"),
        "map_sha256": {str(s): sha256(path) for s, path in map_paths.items()},
        "expected_trials": len(protocol["pairs"]) * len(protocol["perturbations"]) * 2,
        "GT_access": "audit_hook_blocks_GT_paths",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    started = time.monotonic()

    def pose_from_qt(q, t):
        with torch.no_grad():
            pose = torch.eye(4, device="cuda")
            pose[:3, :3] = core.build_rotation(q.reshape(1, 4))[0]
            pose[:3, 3] = t
            return pose.cpu().numpy().astype(np.float64)

    try:
        for pair in protocol["pairs"]:
            frame = pair["frame"]
            anchor = w2c[frame]
            with Image.open(rows[frame]["rgb"]) as image:
                observed = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
            observed_depth = np.load(rows[frame]["depth"], allow_pickle=False).astype(np.float32).squeeze()
            params = {}
            baseline = {}
            for arm, snap in (("before", pair["before"]), ("after", pair["after"])):
                with np.load(map_paths[snap], allow_pickle=False) as archive:
                    params[arm] = {key: torch.as_tensor(archive[key], device="cuda") for key in MAP_KEYS}
                    params[arm]["cam_unnorm_rots"] = torch.as_tensor(archive["cam_unnorm_rots"], device="cuda").clone()
                    params[arm]["cam_trans"] = torch.as_tensor(archive["cam_trans"], device="cuda").clone()
                set_external(params[arm], torch.as_tensor(anchor, device="cuda", dtype=torch.float32), frame, core)
                baseline[arm] = render(params[arm], data, frame, core)
            mask = (np.isfinite(observed_depth) & (observed_depth >= cfg["depth_min_m"])
                    & (observed_depth <= cfg["depth_max_m"]))
            for arm in ("before", "after"):
                rgb, ds = baseline[arm]
                mask &= (np.isfinite(rgb).all(axis=0) & np.isfinite(ds).all(axis=0)
                         & (ds[1] > cfg["alpha_threshold"])
                         & (ds[0] >= cfg["depth_min_m"]) & (ds[0] <= cfg["depth_max_m"]))
            yy, xx = np.indices(mask.shape)
            mask &= (yy % int(cfg["pixel_stride"]) == 0) & (xx % int(cfg["pixel_stride"]) == 0)
            if int(mask.sum()) != pair["support_pixels"] or int(mask.sum()) < cfg["min_common_pixels"]:
                raise ValueError("Fixed-support mismatch")
            for arm in ("before", "after"):
                rgb, ds = baseline[arm]
                rgb_residual = observed[:, mask] - rgb[:, mask]
                depth_residual = observed_depth[mask] - ds[0, mask]
                mse = float(np.mean(rgb_residual ** 2))
                append_jsonl(quality_path, {
                    "frame": frame, "arm": arm, "snapshot": pair[arm],
                    "common_fixed_support_pixels": pair["support_pixels"],
                    "coverage099": float(np.mean(ds[1] > cfg["alpha_threshold"])),
                    "anchor_rgb_mae": float(np.mean(np.abs(rgb_residual))),
                    "anchor_psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
                    "anchor_depth_mae_m": float(np.mean(np.abs(depth_residual))),
                })
            fixed_mask = torch.as_tensor(mask, device="cuda")
            target_rgb = torch.as_tensor(observed, device="cuda")
            target_depth = torch.as_tensor(observed_depth, device="cuda")
            for condition_index, condition in enumerate(protocol["perturbations"]):
                initial = delta_w2c(condition) @ anchor
                arm_order = ("before", "after") if condition_index % 2 == 0 else ("after", "before")
                for arm in arm_order:
                    p = params[arm]
                    set_external(p, torch.as_tensor(initial, device="cuda", dtype=torch.float32), frame, core)
                    base_q = p["cam_unnorm_rots"][..., frame].reshape(4).detach().clone()
                    base_t = p["cam_trans"][..., frame].reshape(3).detach().clone()
                    omega = torch.nn.Parameter(torch.zeros(3, device="cuda"))
                    shift = torch.nn.Parameter(torch.zeros(3, device="cuda"))
                    optimizer = torch.optim.Adam([
                        {"params": [omega], "lr": cfg["lr_rotation_rad"]},
                        {"params": [shift], "lr": cfg["lr_translation_m"]},
                    ])

                    def evaluate():
                        q, t = compose_local_increment(base_q, base_t, omega, shift)
                        trial = insert_camera_at_frame(p, frame, q, t)
                        transformed = core.transform_to_frame(trial, frame, gaussians_grad=False, camera_grad=True)
                        rgb, _, _ = core.Renderer(raster_settings=data["cam"])(
                            **core.transformed_params2rendervar(trial, transformed))
                        depth, _, _ = core.Renderer(raster_settings=data["cam"])(
                            **core.transformed_params2depthplussilhouette(trial, data["w2c"], transformed))
                        loss = (cfg["rgb_weight"] * torch.abs(target_rgb - rgb)[:, fixed_mask].sum()
                                + cfg["depth_weight"] * torch.abs(target_depth - depth[0])[fixed_mask].sum())
                        return loss, q, t

                    best_loss, best_pose, best_step = float("inf"), None, None
                    first_loss, status = None, "COMPLETED"
                    torch.cuda.synchronize()
                    trial_started = time.monotonic()
                    for step in range(int(cfg["steps"]) + 1):
                        optimizer.zero_grad(set_to_none=True)
                        loss, q, t = evaluate()
                        if not torch.isfinite(loss).item():
                            status = "NONFINITE_LOSS"
                            break
                        value = float(loss.detach())
                        if first_loss is None:
                            first_loss = value
                        if value < best_loss:
                            best_loss, best_pose, best_step = value, pose_from_qt(q, t), step
                        if step == cfg["steps"]:
                            break
                        loss.backward()
                        if not (torch.isfinite(omega.grad).all() and torch.isfinite(shift.grad).all()):
                            status = "NONFINITE_6D_GRADIENT"
                            break
                        optimizer.step()
                    torch.cuda.synchronize()
                    append_jsonl(trials_path, {
                        "status": status, "frame": frame, "before": pair["before"], "after": pair["after"],
                        "source_frame_id": rows[frame]["source_frame_id"], "condition": condition,
                        "arm": arm, "common_fixed_support_pixels": pair["support_pixels"],
                        "anchor_w2c": anchor.tolist(), "initial_w2c": initial.tolist(),
                        "selected_w2c": best_pose.tolist() if best_pose is not None else None,
                        "selected_pre_step": best_step, "steps_requested": cfg["steps"],
                        "steps_completed": step, "initial_loss": first_loss,
                        "selected_loss": best_loss if best_pose is not None else None,
                        "wall_seconds": time.monotonic() - trial_started,
                    })
                    print(json.dumps({"frame": frame, "case": condition["id"], "arm": arm,
                                      "status": status, "selected_step": best_step,
                                      "selected_loss": best_loss}), flush=True)
                    del optimizer
            del params
            torch.cuda.empty_cache()
        receipt.update(status="COMPLETED", trials_written=sum(1 for _ in trials_path.open(encoding="utf-8")),
                       trials_sha256=sha256(trials_path), quality_sha256=sha256(quality_path),
                       wall_seconds=time.monotonic() - started)
    except BaseException as error:
        receipt.update(status="FAILED_PRESERVED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "trials": receipt["trials_written"]}), flush=True)


if __name__ == "__main__":
    main()
