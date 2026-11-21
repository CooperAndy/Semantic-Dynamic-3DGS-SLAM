"""Fuse 2-D Mask2Former evidence into a static 3-D Gaussian map.

The script deliberately excludes the pre-registered evaluation-only frames.
Visibility is estimated from the SplaTAM rasterizer: a Gaussian must have a
positive rendered radius and its projected centre depth must agree with the
rendered Gaussian depth.  Evidence weight is opacity * depth agreement *
Mask2Former confidence.  Sky observations are never written.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASE_DIR))

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2depthplussilhouette,
)


DYNAMIC_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
SKY_ID = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--semantic-dir", required=True)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--semantic-confidence-threshold", type=float, default=0.50)
    p.add_argument("--silhouette-threshold", type=float, default=0.05)
    p.add_argument("--depth-relative-tolerance", type=float, default=0.08)
    p.add_argument("--depth-absolute-tolerance", type=float, default=0.35)
    p.add_argument("--representative-frames", default="19,39,59,79,99,119,139,159,179,199")
    return p.parse_args()


def to_gpu_params(npz):
    names = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities",
             "log_scales", "cam_unnorm_rots", "cam_trans"]
    return {name: torch.from_numpy(npz[name]).cuda().float().contiguous() for name in names}


def semantic_rendervar(params, transformed, colors, semantic_valid):
    log_scales = params["log_scales"]
    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    return {
        "means3D": transformed["means3D"],
        "colors_precomp": colors,
        "rotations": F.normalize(transformed["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]) * semantic_valid[:, None],
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda"),
    }


def main():
    args = parse_args()
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = Path(args.output_dir)
    render_dir = out / "semantic_rendered_views"
    out.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.params, allow_pickle=True)
    params = to_gpu_params(raw)
    intrinsics = torch.from_numpy(raw["intrinsics"]).cuda().float()
    first_w2c = raw["w2c"].astype(np.float32)
    width, height = int(raw["org_width"]), int(raw["org_height"])
    num_frames = params["cam_trans"].shape[-1]
    held_out = sorted(int(x) for x in raw["eval_only_frames"].tolist())
    train_frames = [i for i in range(num_frames) if i not in set(held_out)]
    timestep = torch.from_numpy(raw["timestep"]).cuda().float()
    opacity = torch.sigmoid(params["logit_opacities"][:, 0])
    n = params["means3D"].shape[0]

    sem_dir = Path(args.semantic_dir)
    with open(sem_dir / "id2label.json", "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    with open(sem_dir / "palette.json", "r", encoding="utf-8") as f:
        palette_json = json.load(f)
    num_classes = len(id2label)
    palette_np = np.array([palette_json[str(i)]["rgb"] for i in range(num_classes)], dtype=np.float32) / 255.0
    palette = torch.from_numpy(palette_np).cuda()

    cam = setup_camera(width, height, raw["intrinsics"], first_w2c)
    evidence = torch.zeros((n, num_classes), device="cuda", dtype=torch.float32)
    observations = torch.zeros(n, device="cuda", dtype=torch.int32)

    processed = 0
    accepted_observations = 0
    with torch.no_grad():
        for frame in train_frames:
            label_path = sem_dir / "labels" / f"{frame:06d}_semantic.npy"
            conf_path = sem_dir / "confidence" / f"{frame:06d}_confidence.npy"
            labels = torch.from_numpy(np.load(label_path, allow_pickle=False).astype(np.int64)).cuda()
            confidence = torch.from_numpy(np.load(conf_path, allow_pickle=False).astype(np.float32)).cuda()

            transformed = transform_to_frame(params, frame, gaussians_grad=False, camera_grad=False)
            depth_rv = transformed_params2depthplussilhouette(params, torch.from_numpy(first_w2c).cuda(), transformed)
            depth_sil, radii, _ = Renderer(raster_settings=cam)(**depth_rv)
            rendered_depth = depth_sil[0]
            silhouette = depth_sil[1]

            xyz = transformed["means3D"]
            z = xyz[:, 2]
            uv_h = (intrinsics @ xyz.T).T
            u = torch.round(uv_h[:, 0] / torch.clamp(uv_h[:, 2], min=1e-6)).long()
            v = torch.round(uv_h[:, 1] / torch.clamp(uv_h[:, 2], min=1e-6)).long()
            valid = ((timestep <= frame) & (z > 0.05) & (radii > 0) &
                     (u >= 0) & (u < width) & (v >= 0) & (v < height))
            idx = torch.nonzero(valid, as_tuple=False)[:, 0]
            if idx.numel() == 0:
                continue
            uu, vv = u[idx], v[idx]
            lab = labels[vv, uu]
            sem_conf = confidence[vv, uu]
            map_depth = rendered_depth[vv, uu]
            map_sil = silhouette[vv, uu]
            tol = args.depth_absolute_tolerance + args.depth_relative_tolerance * torch.clamp(map_depth, min=0)
            delta = torch.abs(z[idx] - map_depth)
            keep = ((lab != SKY_ID) & (sem_conf >= args.semantic_confidence_threshold) &
                    (map_depth > 0) & (map_sil >= args.silhouette_threshold) & (delta <= tol))
            idx, lab, sem_conf, delta, tol = idx[keep], lab[keep], sem_conf[keep], delta[keep], tol[keep]
            if idx.numel():
                visibility = opacity[idx] * torch.exp(-delta / torch.clamp(tol, min=1e-6))
                weight = visibility * sem_conf
                evidence.index_put_((idx, lab), weight, accumulate=True)
                observations.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
                accepted_observations += int(idx.numel())
            processed += 1
            if processed == 1 or processed % 20 == 0:
                print(f"fused {processed}/{len(train_frames)} training frames; accepted={accepted_observations}", flush=True)

    total_evidence = evidence.sum(dim=1)
    max_evidence, labels = evidence.max(dim=1)
    assigned = total_evidence > 0
    semantic_confidence = torch.zeros_like(total_evidence)
    semantic_confidence[assigned] = max_evidence[assigned] / total_evidence[assigned]
    labels_out = labels.to(torch.int16)
    labels_out[~assigned] = -1
    dynamic = torch.zeros(n, dtype=torch.bool, device="cuda")
    for class_id in DYNAMIC_IDS:
        dynamic |= labels_out == class_id

    labels_cpu = labels_out.cpu().numpy()
    conf_cpu = semantic_confidence.cpu().numpy().astype(np.float16)
    total_cpu = total_evidence.cpu().numpy().astype(np.float32)
    obs_cpu = observations.cpu().numpy().astype(np.uint16)
    dynamic_cpu = dynamic.cpu().numpy()
    np.savez_compressed(
        out / "semantic_gaussian_labels.npz",
        label=labels_cpu,
        confidence=conf_cpu,
        evidence_sum=total_cpu,
        observation_count=obs_cpu,
        potentially_dynamic=dynamic_cpu,
        class_names=np.array([id2label[i] for i in range(num_classes)]),
        source_gaussian_index=np.arange(n, dtype=np.int32),
        held_out_frames=np.array(held_out, dtype=np.int16),
    )

    rows = []
    assigned_count = int(assigned.sum().item())
    for class_id in range(num_classes):
        mask = labels_cpu == class_id
        count = int(mask.sum())
        rows.append({
            "class_id": class_id,
            "class_name": id2label[class_id],
            "potentially_dynamic": class_id in DYNAMIC_IDS,
            "gaussian_count": count,
            "fraction_all_gaussians": count / n,
            "fraction_assigned_gaussians": count / assigned_count if assigned_count else 0.0,
            "mean_fused_confidence": float(conf_cpu[mask].mean()) if count else 0.0,
            "evidence_sum": float(total_cpu[mask].sum()) if count else 0.0,
        })
    with open(out / "semantic_class_histogram.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    representative = [int(x) for x in args.representative_frames.split(",") if x.strip()]
    semantic_colors = torch.zeros((n, 3), device="cuda")
    if assigned.any():
        semantic_colors[assigned] = palette[labels[assigned].long()]
    semantic_valid = assigned.float()
    rgb_dir = Path(args.rgb_dir)
    with torch.no_grad():
        for frame in representative:
            transformed = transform_to_frame(params, frame, gaussians_grad=False, camera_grad=False)
            rv = semantic_rendervar(params, transformed, semantic_colors, semantic_valid)
            rendered, _, _ = Renderer(raster_settings=cam)(**rv)
            semantic_rgb = (torch.clamp(rendered, 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(render_dir / f"semantic_{frame:06d}.png"), cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR))
            source_bgr = cv2.imread(str(rgb_dir / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
            if source_bgr is not None:
                sem_bgr = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(source_bgr, 0.45, sem_bgr, 0.55, 0)
                cv2.imwrite(str(render_dir / f"overlay_{frame:06d}.png"), overlay)

    runtime = time.time() - started
    summary = {
        "status": "COMPLETE",
        "params": str(Path(args.params).resolve()),
        "semantic_source": str(sem_dir.resolve()),
        "num_gaussians": n,
        "assigned_gaussians": assigned_count,
        "assigned_fraction": assigned_count / n,
        "unassigned_gaussians": n - assigned_count,
        "potentially_dynamic_gaussians": int(dynamic.sum().item()),
        "sky_written": 0,
        "num_training_frames": len(train_frames),
        "training_frames": train_frames,
        "held_out_frames": held_out,
        "accepted_gaussian_frame_observations": accepted_observations,
        "semantic_confidence_threshold": args.semantic_confidence_threshold,
        "visibility": {
            "definition": "positive rasterizer radius and projected-centre depth agreement with rendered Gaussian depth",
            "weight": "sigmoid(opacity_logit) * exp(-abs(center_depth-rendered_depth)/tolerance) * Mask2Former_confidence",
            "silhouette_threshold": args.silhouette_threshold,
            "depth_tolerance_m": f"{args.depth_absolute_tolerance} + {args.depth_relative_tolerance} * rendered_depth",
        },
        "class_semantics": "Cityscapes-19; IDs 11-18 mean potentially dynamic class, not measured motion",
        "representative_frames": representative,
        "runtime_seconds": runtime,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    with open(out / "semantic_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ["status", "num_gaussians", "assigned_gaussians", "assigned_fraction", "runtime_seconds", "peak_cuda_memory_gib"]}, indent=2))


if __name__ == "__main__":
    main()
