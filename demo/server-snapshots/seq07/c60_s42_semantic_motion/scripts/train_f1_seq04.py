"""F1: appearance refinement of the complete F200 Gaussian map.

All 200 KITTI07 frames have already been used to build F200. Outputs from this
script are seen-view deployment experiments, never new held-out Test evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path("/root/autodl-tmp/seq07_000850_001049_tuning_20261115/c60_s42_semantic_motion/deployment")
SOURCE = Path("/root/autodl-tmp/seq07_000850_001049_tuning_20261115/c60_s42_semantic_motion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-frame-ids", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--color-lr", type=float, default=0.001)
    parser.add_argument("--opacity-lr", type=float, default=0.005)
    parser.add_argument("--protocol-root", type=Path, default=ROOT)
    parser.add_argument("--run-kind", choices=["smoke", "f1_all200"], required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing existing output: {args.out}")
    if not 1 <= args.steps <= 1000 or args.color_lr <= 0 or args.opacity_lr <= 0:
        raise ValueError("invalid pilot optimization budget")
    root = args.protocol_root.resolve()
    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "split.csv") != manifest["split_sha256"]:
        raise ValueError("frozen split changed")
    if len(set(args.train_frame_ids)) != len(args.train_frame_ids) or any(
            frame not in range(0, 200) for frame in args.train_frame_ids):
        raise ValueError("optimization views must be unique members of F200")
    if args.run_kind == "smoke" and (args.steps != 20 or len(args.train_frame_ids) != 10):
        raise ValueError("F1 smoke requires 10 unique frames and 20 steps")
    if args.run_kind == "f1_all200" and (
            args.train_frame_ids != list(range(0, 200)) or args.steps != 800
            or args.color_lr != 0.001 or args.opacity_lr != 0.005
            or manifest.get("status") != "ALL_200_SEEN_DEPLOYMENT_NO_HELDOUT"):
        raise ValueError("F1 requires four complete 200-frame rounds and frozen learning rates")
    source_sha = sha256(args.map)
    if source_sha != args.expected_map_sha256:
        raise ValueError("F200 input SHA256 mismatch")

    import sys
    sys.path.insert(0, "/root/autodl-tmp/SplaTAM")
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    from utils.recon_helpers import setup_camera

    tic = time.perf_counter()
    with np.load(args.map, allow_pickle=False) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    means = torch.from_numpy(data["means3D"]).cuda()
    colors = torch.nn.Parameter(torch.from_numpy(data["rgb_colors"]).cuda())
    logits = torch.nn.Parameter(torch.from_numpy(data["logit_opacities"]).cuda())
    rotations = F.normalize(torch.from_numpy(data["unnorm_rotations"]).cuda(), dim=1)
    log_scales = torch.from_numpy(data["log_scales"]).cuda()
    scales = torch.exp(log_scales.repeat(1, 3) if log_scales.shape[1] == 1 else log_scales)
    bg = torch.tensor([0.055, 0.065, 0.08], device="cuda", dtype=torch.float32)
    optimizer = torch.optim.Adam([
        {"params": [colors], "lr": args.color_lr},
        {"params": [logits], "lr": args.opacity_lr},
    ])
    poses = np.loadtxt(SOURCE / "poses_estimated_camera2.txt", dtype=np.float64).reshape(200, 3, 4)
    c2w = np.tile(np.eye(4, dtype=np.float64), (200, 1, 1))
    c2w[:, :3, :4] = poses
    first_inverse = np.linalg.inv(c2w[0])
    views = {}
    for frame in args.train_frame_ids:
        local = frame
        w2c = np.linalg.inv(first_inverse @ c2w[local]).astype(np.float32)
        cam = setup_camera(1226, 370, data["intrinsics"], w2c, near=0.01, far=150.0)
        common = dict(image_height=cam.image_height, image_width=cam.image_width,
                      tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
                      scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
                      projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
                      campos=cam.campos, prefiltered=cam.prefiltered)
        image_path = SOURCE / "sequences/07/image_2" / f"{local:06d}.png"
        mask_path = SOURCE / "reliable_roi" / f"{local:06d}_valid.npy"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError((image_path, mask_path))
        views[frame] = (Camera(bg=bg, **common), image_path, mask_path)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        from resource_guard import wait_for_headroom
        wait_for_headroom("F1", step)
        frame = args.train_frame_ids[step % len(args.train_frame_ids)]
        camera, image_path, mask_path = views[frame]
        target = torch.from_numpy(np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0).cuda().permute(2, 0, 1)
        mask = torch.from_numpy(np.load(mask_path, allow_pickle=False).astype(np.float32)).cuda()
        if not mask.any():
            raise ValueError(f"empty static ROI at frame {frame}")
        optimizer.zero_grad(set_to_none=True)
        rgb, _, _ = Renderer(raster_settings=camera)(
            means3D=means, means2D=torch.zeros_like(means),
            colors_precomp=colors, rotations=rotations,
            opacities=torch.sigmoid(logits), scales=scales)
        loss = ((rgb - target).abs() * mask[None]).sum() / (3 * mask.sum())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append({"step": step + 1, "train_frame": frame, "roi_l1": float(loss.detach().item())})
        if (step + 1) % 5 == 0 or step == 0:
            print(json.dumps(losses[-1]), flush=True)
        del rgb, loss, target, mask
    torch.cuda.synchronize()
    data["rgb_colors"] = colors.detach().cpu().numpy()
    data["logit_opacities"] = logits.detach().cpu().numpy()
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "params.npz"
    np.savez(checkpoint, **data)
    cycle_len = len(args.train_frame_ids)
    paired_cycle = (len(losses) >= 2 * cycle_len and len(losses) % cycle_len == 0)
    first_cycle_mean = float(np.mean([entry["roi_l1"] for entry in losses[:cycle_len]])) if paired_cycle else None
    last_cycle_mean = float(np.mean([entry["roi_l1"] for entry in losses[-cycle_len:]])) if paired_cycle else None
    summary = {
        "source_map": str(args.map), "source_map_sha256": source_sha,
        "output_map_sha256": sha256(checkpoint), "split_sha256": manifest["split_sha256"],
        "optimized_fields": ["rgb_colors", "logit_opacities"],
        "fixed_fields": ["means3D", "log_scales", "unnorm_rotations", "cameras"],
        "train_frame_ids": args.train_frame_ids, "steps": args.steps,
        "color_lr": args.color_lr, "opacity_lr": args.opacity_lr,
        "first_cycle_mean_train_roi_l1": first_cycle_mean,
        "last_cycle_mean_train_roi_l1": last_cycle_mean,
        "paired_cycle_comparison_valid": paired_cycle,
        "losses": losses,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "elapsed_seconds": time.perf_counter() - tic,
        "run_kind": args.run_kind,
        "gaussians": int(means.shape[0]),
        "status": ("F1_all_200_seen_deployment_not_heldout"
                   if args.run_kind == "f1_all200" else "F1_smoke_not_quality_evidence"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "losses"}), flush=True)


if __name__ == "__main__":
    main()
