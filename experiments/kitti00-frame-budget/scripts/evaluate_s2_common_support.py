"""Render a completed S1 map on one fixed RGB/depth evaluation support.

Unlike SplaTAM's legacy eval helper, this audit never masks RGB by the
threshold-specific hard-depth input.  Every map is compared on full-frame RGB
and on the same raw-RAFT 0.5--40 m depth support.  It also reports silhouette
coverage so a sparse map cannot improve metrics merely by shrinking support.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from datasets.gradslam_datasets import KITTIDataset, load_dataset_config
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_psnr
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2depthplussilhouette,
    transformed_params2rendervar,
)


DEFAULT_EVAL_FRAMES = tuple(range(19, 200, 20))
PARAM_KEYS = (
    "means3D",
    "rgb_colors",
    "unnorm_rotations",
    "logit_opacities",
    "log_scales",
    "cam_unnorm_rots",
    "cam_trans",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default=",".join(str(value) for value in DEFAULT_EVAL_FRAMES))
    parser.add_argument("--frame-role", choices=("heldout", "in_sample"), required=True)
    parser.add_argument("--sil-threshold", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=40.0)
    return parser.parse_args()


def load_params(path: Path) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(PARAM_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"missing params keys: {missing}")
        for key in PARAM_KEYS:
            params[key] = torch.from_numpy(archive[key]).cuda().float().contiguous()
    return params


def main() -> None:
    args = parse_args()
    eval_frames = tuple(int(value) for value in args.frames.split(",") if value.strip())
    if not eval_frames:
        raise ValueError("--frames must contain at least one frame index")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_dataset_config(str(args.data_config))
    dataset = KITTIDataset(
        config,
        basedir="/root/autodl-tmp/slam_project/data/kitti/dataset/sequences",
        sequence="00",
        start=0,
        end=-1,
        stride=1,
        desired_height=376,
        desired_width=1241,
        device="cuda:0",
        relative_pose=True,
        ignore_bad=False,
        use_train_split=True,
    )
    if len(dataset) < 200:
        raise ValueError(f"expected at least 200 frames, found {len(dataset)}")

    params = load_params(args.params)
    color0, _, intrinsics0, pose0 = dataset[0]
    first_w2c = torch.linalg.inv(pose0)
    camera = setup_camera(
        color0.shape[1],
        color0.shape[0],
        intrinsics0[:3, :3].cpu().numpy(),
        first_w2c.detach().cpu().numpy(),
    )
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
    rows: list[dict[str, int | float]] = []

    with torch.no_grad():
        for frame in eval_frames:
            color, depth, _, _ = dataset[frame]
            color = (color.permute(2, 0, 1) / 255.0).cuda().float()
            depth = depth.permute(2, 0, 1).cuda().float()
            common_valid = (depth[0] >= args.min_depth) & (depth[0] <= args.max_depth)

            transformed = transform_to_frame(
                params, frame, gaussians_grad=False, camera_grad=False
            )
            render_vars = transformed_params2rendervar(params, transformed)
            depth_vars = transformed_params2depthplussilhouette(
                params, first_w2c, transformed
            )
            image, _, _ = Renderer(raster_settings=camera)(**render_vars)
            depth_sil, _, _ = Renderer(raster_settings=camera)(**depth_vars)
            render_depth = depth_sil[0]
            silhouette = depth_sil[1]
            presence = silhouette > args.sil_threshold

            image = torch.clamp(image, 0.0, 1.0)
            psnr = float(calc_psnr(image, color).mean().item())
            ssim = float(
                ms_ssim(
                    image.unsqueeze(0).cpu(),
                    color.unsqueeze(0).cpu(),
                    data_range=1.0,
                    size_average=True,
                ).item()
            )
            lpips_value = float(
                lpips(image.unsqueeze(0), color.unsqueeze(0)).item()
            )

            supported = common_valid & presence
            abs_error = torch.abs(render_depth - depth[0])
            depth_l1_all = float(abs_error[common_valid].mean().item())
            depth_l1_supported = (
                float(abs_error[supported].mean().item()) if supported.any() else float("nan")
            )
            coverage_all = float(presence.float().mean().item())
            coverage_common = float(presence[common_valid].float().mean().item())
            rows.append(
                {
                    "threshold": args.threshold,
                    "frame": frame,
                    "full_rgb_psnr": psnr,
                    "full_rgb_ms_ssim": ssim,
                    "full_rgb_lpips": lpips_value,
                    "common_depth_l1_all_valid_m": depth_l1_all,
                    "common_depth_l1_supported_m": depth_l1_supported,
                    "silhouette_coverage_all": coverage_all,
                    "silhouette_coverage_common_depth": coverage_common,
                    "hole_ratio_common_depth": 1.0 - coverage_common,
                    "common_depth_valid_ratio": float(common_valid.float().mean().item()),
                }
            )
            print(
                f"T={args.threshold:.2f} frame={frame:03d} PSNR={psnr:.3f} "
                f"L1={depth_l1_all:.3f} coverage={coverage_common:.3f}",
                flush=True,
            )

    per_frame = args.output_dir / "common_support_per_frame.csv"
    with per_frame.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [key for key in rows[0] if key not in {"threshold", "frame"}]
    summary = {
        "threshold": args.threshold,
        "params_path": str(args.params),
        "data_config": str(args.data_config),
        "evaluation_frames": list(eval_frames),
        "evaluation_role": args.frame_role,
        "rgb_support": "full 376x1241 image for every mapping budget",
        "depth_support": "same raw RAFT depth pixels in [0.5,40] m for every mapping budget",
        "silhouette_threshold": args.sil_threshold,
        "metrics": {
            name: float(np.nanmean([float(row[name]) for row in rows]))
            for name in metric_names
        },
    }
    summary_path = args.output_dir / "common_support_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
