#!/usr/bin/env python3
"""Compute RAFT optical flow, rigid flow, and instance-level motion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--pose-coordinate", choices=["p0", "p2"], required=True)
    parser.add_argument("--calib-file", type=Path, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2, help="Inclusive final frame; outputs end-start pairs")
    parser.add_argument("--fb-abs-threshold", type=float, default=1.0)
    parser.add_argument("--fb-relative-threshold", type=float, default=0.05)
    parser.add_argument("--occlusion-abs-m", type=float, default=0.50)
    parser.add_argument("--occlusion-relative", type=float, default=0.05)
    parser.add_argument("--probability-center", type=float, default=3.0)
    parser.add_argument("--probability-temperature", type=float, default=1.0)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        numbers = np.fromstring(values, sep=" ", dtype=np.float64)
        if numbers.size == 12:
            matrices[key] = numbers.reshape(3, 4)
    if "P2" not in matrices:
        raise KeyError(f"P2 not found in {path}")
    p2 = matrices["P2"]
    k = p2[:, :3].copy()
    t_c2_c0 = np.eye(4, dtype=np.float64)
    t_c2_c0[:3, 3] = np.linalg.solve(k, p2[:, 3])
    return k, t_c2_c0


def load_poses(path: Path, pose_coordinate: str, t_c2_c0: np.ndarray) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], raw.shape[0], axis=0)
    poses[:, :3, :4] = raw
    if pose_coordinate == "p0":
        t_c0_c2 = np.linalg.inv(t_c2_c0)
        poses = poses @ t_c0_c2[None]
    return poses


def compute_rigid_flow(
    depth: np.ndarray,
    k: np.ndarray,
    pose_t_c2w: np.ndarray,
    pose_next_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rigid flow, validity, and predicted next-camera Z."""
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float64)
    z = depth.astype(np.float64, copy=False)
    x_cam = (x - k[0, 2]) * z / k[0, 0]
    y_cam = (y - k[1, 2]) * z / k[1, 1]
    points = np.stack((x_cam, y_cam, z, np.ones_like(z)), axis=0).reshape(4, -1)

    t_next_t = np.linalg.inv(pose_next_c2w) @ pose_t_c2w
    points_next = t_next_t @ points
    z_next = points_next[2]
    safe_z = np.where(np.abs(z_next) > 1e-9, z_next, 1.0)
    u_next = k[0, 0] * points_next[0] / safe_z + k[0, 2]
    v_next = k[1, 1] * points_next[1] / safe_z + k[1, 2]

    valid = (
        np.isfinite(z.reshape(-1))
        & (z.reshape(-1) > 0)
        & np.isfinite(u_next)
        & np.isfinite(v_next)
        & (z_next > 0)
        & (u_next >= 0)
        & (u_next <= width - 1)
        & (v_next >= 0)
        & (v_next <= height - 1)
    )
    flow = np.stack((u_next - x.reshape(-1), v_next - y.reshape(-1)), axis=1)
    flow[~valid] = 0.0
    return (
        flow.reshape(height, width, 2).astype(np.float32),
        valid.reshape(height, width),
        z_next.reshape(height, width).astype(np.float32),
    )


def read_rgb_tensor(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not decode {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return bgr, tensor


class RaftFlowEstimator:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).eval().to(self.device)
        self.transforms = self.weights.transforms()

    @torch.inference_mode()
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[np.ndarray, float]:
        batch1, batch2 = self.transforms(image1[None], image2[None])
        height, width = batch1.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        batch1 = torch_f.pad(batch1, (0, pad_w, 0, pad_h)).to(self.device)
        batch2 = torch_f.pad(batch2, (0, pad_w, 0, pad_h)).to(self.device)
        started = time.perf_counter()
        flow = self.model(batch1, batch2)[-1][0, :, :height, :width]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return flow.permute(1, 2, 0).cpu().numpy().astype(np.float32), elapsed_ms

    def checkpoint_metadata(self) -> dict[str, str | None]:
        filename = Path(urlparse(self.weights.url).path).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        return {
            "weights": str(self.weights),
            "url": self.weights.url,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        }


def remap_vector(field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    map_x = coordinates[..., 0].astype(np.float32)
    map_y = coordinates[..., 1].astype(np.float32)
    channels = [
        cv2.remap(field[..., channel], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        for channel in range(field.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def forward_backward_valid(
    forward: np.ndarray,
    backward: np.ndarray,
    absolute_threshold: float,
    relative_threshold: float,
) -> np.ndarray:
    height, width = forward.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    destination = np.stack((x, y), axis=-1) + forward
    inside = (
        (destination[..., 0] >= 0)
        & (destination[..., 0] <= width - 1)
        & (destination[..., 1] >= 0)
        & (destination[..., 1] <= height - 1)
    )
    backward_warped = remap_vector(backward, destination)
    error = np.linalg.norm(forward + backward_warped, axis=-1)
    magnitude = np.linalg.norm(forward, axis=-1) + np.linalg.norm(backward_warped, axis=-1)
    return inside & (error <= absolute_threshold + relative_threshold * magnitude)


def rigid_occlusion_mask(
    depth_next: np.ndarray,
    rigid_flow: np.ndarray,
    z_predicted: np.ndarray,
    absolute_threshold: float,
    relative_threshold: float,
) -> np.ndarray:
    height, width = depth_next.shape
    y, x = np.indices((height, width), dtype=np.float32)
    destination = np.stack((x, y), axis=-1) + rigid_flow
    sampled = cv2.remap(
        depth_next.astype(np.float32),
        destination[..., 0].astype(np.float32),
        destination[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    threshold = absolute_threshold + relative_threshold * np.maximum(sampled, z_predicted)
    return (sampled > 0) & (z_predicted > sampled + threshold)


def load_instances(path: Path, shape: tuple[int, int]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    masks = np.load(path.parent / f"{path.name}_masks.npz", allow_pickle=False)["masks"]
    if masks.dtype != np.bool_ or masks.shape[1:] != shape:
        raise RuntimeError(f"Bad instance mask shape/dtype at {path}")
    union = masks.any(axis=0) if masks.shape[0] else np.zeros(shape, dtype=np.bool_)
    return metadata["instances"], masks, union


def robust_motion_probability(
    residual_raw: np.ndarray,
    valid: np.ndarray,
    candidate_union: np.ndarray,
    center: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    background = valid & ~candidate_union
    samples = residual_raw[background]
    if samples.size < 100:
        samples = residual_raw[valid]
    if samples.size == 0:
        raise RuntimeError("No valid residual samples")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    robust_scale = max(1.4826 * mad, 0.25)
    normalized = (residual_raw - median) / robust_scale
    normalized[~valid] = 0.0
    logits = np.clip((normalized - center) / max(temperature, 1e-6), -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    probability[~valid] = 0.0
    return normalized.astype(np.float32), probability.astype(np.float32), {
        "background_median_px": median,
        "background_mad_px": mad,
        "robust_scale_px": robust_scale,
    }


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    scale = max(float(np.percentile(magnitude, 95)), 1.0)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_preview(
    image: np.ndarray,
    observed: np.ndarray,
    rigid: np.ndarray,
    normalized: np.ndarray,
    instances: list[dict],
) -> np.ndarray:
    observed_bgr = flow_to_bgr(observed)
    rigid_bgr = flow_to_bgr(rigid)
    residual_u8 = np.clip(normalized / 10.0 * 255.0, 0, 255).astype(np.uint8)
    residual_bgr = cv2.applyColorMap(residual_u8, cv2.COLORMAP_TURBO)
    for instance in instances:
        x1, y1, x2, y2 = np.rint(instance["bbox"]).astype(int)
        cv2.rectangle(residual_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)
    panels = [image, observed_bgr, rigid_bgr, residual_bgr]
    labels = ["RGB", "Observed RAFT flow", "Rigid flow", "Normalized residual"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (260, 28), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.concatenate(panels, axis=1)


def safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(np.asarray(values, dtype=np.float64), 95)),
    }


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    k, t_c2_c0 = parse_calibration(args.calib_file)
    poses = load_poses(args.pose_file, args.pose_coordinate, t_c2_c0)
    if poses.shape[0] <= args.end:
        raise RuntimeError(f"Pose file has {poses.shape[0]} rows, need index {args.end}")
    raft = RaftFlowEstimator(args.device)

    flow_times: list[float] = []
    valid_ratios: list[float] = []
    pair_summaries: list[dict] = []
    for frame_index in range(args.start, args.end):
        next_index = frame_index + 1
        frame_id = f"{frame_index:06d}"
        next_id = f"{next_index:06d}"
        image, tensor = read_rgb_tensor(args.image_dir / f"{frame_id}.png")
        _, tensor_next = read_rgb_tensor(args.image_dir / f"{next_id}.png")
        depth = np.load(args.depth_dir / f"{frame_id}_depth.npy", allow_pickle=False).astype(
            np.float32, copy=False
        )
        depth_next = np.load(
            args.depth_dir / f"{next_id}_depth.npy", allow_pickle=False
        ).astype(np.float32, copy=False)
        if depth.shape != image.shape[:2] or depth_next.shape != image.shape[:2]:
            raise RuntimeError(f"Depth/RGB shape mismatch at pair {frame_id}->{next_id}")

        observed, forward_ms = raft(tensor, tensor_next)
        backward, backward_ms = raft(tensor_next, tensor)
        flow_times.extend([forward_ms, backward_ms])
        rigid, rigid_valid, z_predicted = compute_rigid_flow(
            depth, k, poses[frame_index], poses[next_index]
        )
        fb_valid = forward_backward_valid(
            observed, backward, args.fb_abs_threshold, args.fb_relative_threshold
        )
        occlusion = rigid_occlusion_mask(
            depth_next,
            rigid,
            z_predicted,
            args.occlusion_abs_m,
            args.occlusion_relative,
        )
        valid = rigid_valid & fb_valid & ~occlusion
        residual_raw = np.linalg.norm(observed - rigid, axis=-1).astype(np.float32)

        instance_path = args.instance_dir / frame_id
        instances, instance_masks, candidate_union = load_instances(instance_path, depth.shape)
        normalized, probability, normalization = robust_motion_probability(
            residual_raw,
            valid,
            candidate_union,
            args.probability_center,
            args.probability_temperature,
        )

        instance_results: list[dict] = []
        for index, instance in enumerate(instances):
            instance_valid = valid & instance_masks[index]
            valid_count = int(instance_valid.sum())
            area = int(instance_masks[index].sum())
            if valid_count:
                residual_values = normalized[instance_valid]
                probability_values = probability[instance_valid]
                median_residual = float(np.median(residual_values))
                p90_residual = float(np.percentile(residual_values, 90))
                dynamic_probability = float(np.median(probability_values))
            else:
                median_residual = None
                p90_residual = None
                dynamic_probability = None
            instance_results.append(
                {
                    "instance_id": instance["instance_id"],
                    "track_id": instance["track_id"],
                    "category": instance["category"],
                    "valid_ratio": valid_count / max(area, 1),
                    "valid_pixels": valid_count,
                    "median_normalized_residual": median_residual,
                    "p90_normalized_residual": p90_residual,
                    "dynamic_probability": dynamic_probability,
                }
            )

        output_stem = f"{frame_id}_{next_id}"
        np.savez_compressed(
            args.output_dir / f"{output_stem}.npz",
            observed_flow=observed.astype(np.float32),
            rigid_flow=rigid.astype(np.float32),
            residual=normalized.astype(np.float32),
            residual_raw=residual_raw.astype(np.float32),
            valid_mask=valid.astype(np.bool_),
            occlusion_mask=occlusion.astype(np.bool_),
            dynamic_prob=probability.astype(np.float32),
        )
        pair_summary = {
            "frame_t": frame_id,
            "frame_next": next_id,
            "pose_coordinate_input": args.pose_coordinate,
            "valid_ratio": float(valid.mean()),
            "rigid_valid_ratio": float(rigid_valid.mean()),
            "flow_fb_valid_ratio": float(fb_valid.mean()),
            "occlusion_ratio": float(occlusion.mean()),
            "normalization": normalization,
            "raft_forward_ms": forward_ms,
            "raft_backward_ms": backward_ms,
            "instances": instance_results,
        }
        (args.output_dir / f"{output_stem}.json").write_text(
            json.dumps(pair_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if args.preview_every > 0 and (frame_index - args.start) % args.preview_every == 0:
            cv2.imwrite(
                str(preview_dir / f"{output_stem}.png"),
                make_preview(image, observed, rigid, normalized, instances),
            )
        pair_summaries.append(pair_summary)
        valid_ratios.append(pair_summary["valid_ratio"])
        print(
            f"[{frame_index - args.start + 1:04d}/{args.end - args.start:04d}] "
            f"{output_stem}: valid={pair_summary['valid_ratio']:.3f} "
            f"raft={forward_ms:.1f}+{backward_ms:.1f}ms",
            flush=True,
        )

    summary = {
        "status": "GENERATED",
        "pairs": len(pair_summaries),
        "frame_range": [args.start, args.end],
        "pose_file": str(args.pose_file),
        "pose_coordinate_input": args.pose_coordinate,
        "pose_coordinate_internal": "p2_c2w",
        "depth_dir": str(args.depth_dir),
        "instance_dir": str(args.instance_dir),
        "calibration": {
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
            "t_c2_c0": t_c2_c0.tolist(),
        },
        "raft": raft.checkpoint_metadata(),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "raft_inference_ms_per_direction": safe_stats(flow_times),
        "valid_ratio": safe_stats(valid_ratios),
        "thresholds": {
            "fb_abs": args.fb_abs_threshold,
            "fb_relative": args.fb_relative_threshold,
            "occlusion_abs_m": args.occlusion_abs_m,
            "occlusion_relative": args.occlusion_relative,
            "probability_center": args.probability_center,
            "probability_temperature": args.probability_temperature,
        },
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
