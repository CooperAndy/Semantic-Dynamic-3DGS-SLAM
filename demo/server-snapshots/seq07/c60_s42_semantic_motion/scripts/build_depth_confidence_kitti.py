"""Build pure-stereo confidence maps for KITTI depth observations.

This script needs no external sensors or depth ground truth.  It combines:
  * valid-range and optional static-valid masking;
  * left/right round-trip disparity consistency;
  * left/right photometric reprojection consistency; and
  * adjacent-frame depth consistency using fixed external camera poses.

It intentionally refuses non-empty output folders and validates frame IDs,
array shapes, dtypes, finite values, calibration and external poses before
writing any result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-image-dir", type=Path, required=True)
    parser.add_argument("--right-image-dir", type=Path, required=True)
    parser.add_argument("--depth-left-dir", type=Path, required=True)
    parser.add_argument("--depth-right-dir", type=Path, required=True)
    parser.add_argument("--pose-path", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--static-valid-mask-dir",
        type=Path,
        help="Optional bool masks named <frame>_valid.npy; false pixels receive zero confidence.",
    )
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=40.0)
    parser.add_argument("--seed-threshold", type=float, default=0.80)
    parser.add_argument("--loss-threshold", type=float, default=0.20)
    parser.add_argument("--lr-cycle-sigma-px", type=float, default=1.0)
    parser.add_argument("--photo-sigma", type=float, default=0.10)
    parser.add_argument("--temporal-sigma-rel", type=float, default=0.10)
    parser.add_argument("--preview-indices", type=int, nargs="*", default=[0, 49, 99, 149, 199])
    return parser.parse_args()


def parse_calibration(path: Path) -> dict[str, float]:
    matrices: dict[str, np.ndarray] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 13 or parts[0].rstrip(":") not in {"P2", "P3"}:
            continue
        matrices[parts[0].rstrip(":")] = np.asarray(parts[1:], dtype=np.float64).reshape(3, 4)
    if set(matrices) != {"P2", "P3"}:
        raise ValueError("calib.txt must contain 3x4 P2 and P3 projection matrices")
    p2, p3 = matrices["P2"], matrices["P3"]
    center2 = -p2[0, 3] / p2[0, 0]
    center3 = -p3[0, 3] / p3[0, 0]
    baseline = abs(center3 - center2)
    if not 0.1 < baseline < 1.0:
        raise ValueError(f"unexpected P2/P3 baseline: {baseline:.6f} m")
    return {
        "fx": float(p2[0, 0]),
        "fy": float(p2[1, 1]),
        "cx": float(p2[0, 2]),
        "cy": float(p2[1, 2]),
        "baseline_m": float(baseline),
    }


def frame_paths(directory: Path, pattern: str, count: int) -> list[Path]:
    paths = sorted(directory.glob(pattern))[:count]
    if len(paths) != count:
        raise ValueError(f"expected {count} {pattern} files in {directory}, found {len(paths)}")
    return paths


def depth_frame_id(path: Path) -> str:
    return path.name.removesuffix("_depth.npy")


def load_poses(path: Path, count: int) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[0] < count or poses.shape[1] != 12 or not np.isfinite(poses[:count]).all():
        raise ValueError("pose file must have at least num-frames finite KITTI 12-column poses")
    transforms = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    transforms[:, :3, :4] = poses[:count].reshape(count, 3, 4)
    return transforms


def load_depth(path: Path, shape: tuple[int, int]) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if depth.dtype != np.float32 or depth.shape != shape or not np.isfinite(depth).all():
        raise ValueError(f"invalid depth file: {path}")
    return depth


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray, border_value: float = np.nan) -> np.ndarray:
    sampled = cv2.remap(
        image.astype(np.float32, copy=False),
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return sampled.astype(np.float32, copy=False)


def confidence_from_stereo(
    left: np.ndarray,
    right: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
    scale: float,
    valid: np.ndarray,
    cycle_sigma_px: float,
    photo_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LR confidence, round-trip error, photo error and correspondence-valid mask."""
    height, width = depth_left.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    disparity_left = np.zeros_like(depth_left)
    disparity_left[valid] = scale / depth_left[valid]
    x_right = grid_x - disparity_left
    inside = valid & (x_right >= 0.0) & (x_right <= width - 1.0)

    depth_right_sample = bilinear(depth_right, x_right, grid_y)
    disparity_right = np.zeros_like(depth_right_sample)
    right_valid = np.isfinite(depth_right_sample) & (depth_right_sample > 0.0)
    disparity_right[right_valid] = scale / depth_right_sample[right_valid]
    x_roundtrip = x_right + disparity_right
    cycle_error = np.abs(x_roundtrip - grid_x)

    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    right_warp = bilinear(right_gray, x_right, grid_y, border_value=0.0)
    photo_error = np.abs(left_gray - right_warp)

    correspond = inside & right_valid & np.isfinite(cycle_error)
    confidence = np.zeros_like(depth_left, dtype=np.float32)
    confidence[correspond] = np.exp(-cycle_error[correspond] / cycle_sigma_px) * np.exp(
        -photo_error[correspond] / photo_sigma
    )
    return confidence, cycle_error, photo_error, correspond


def temporal_confidence(
    depth: np.ndarray,
    next_depth: np.ndarray,
    pose_current: np.ndarray,
    pose_next: np.ndarray,
    calibration: dict[str, float],
    valid: np.ndarray,
    sigma_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project current-frame points into the next P2 image and compare depths."""
    height, width = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    z = depth.astype(np.float64)
    x = (grid_x - calibration["cx"]) * z / calibration["fx"]
    y = (grid_y - calibration["cy"]) * z / calibration["fy"]
    points_current = np.stack((x, y, z, np.ones_like(z)), axis=-1).reshape(-1, 4)
    points_world = (pose_current @ points_current.T).T
    points_next = (np.linalg.inv(pose_next) @ points_world.T).T
    z_next = points_next[:, 2].reshape(height, width)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = calibration["fx"] * points_next[:, 0].reshape(height, width) / z_next + calibration["cx"]
        v = calibration["fy"] * points_next[:, 1].reshape(height, width) / z_next + calibration["cy"]
    inside = valid & np.isfinite(u) & np.isfinite(v) & (z_next > 0.0)
    inside &= (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)
    observed_next = bilinear(next_depth, u, v)
    target_valid = np.isfinite(observed_next) & (observed_next > 0.0)
    correspond = inside & target_valid
    disagreement = np.full((height, width), np.nan, dtype=np.float32)
    disagreement[correspond] = (
        np.abs(z_next[correspond] - observed_next[correspond])
        / np.maximum(z_next[correspond], observed_next[correspond])
    ).astype(np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    confidence[correspond] = np.exp(-disagreement[correspond] / sigma_rel)
    return confidence, disagreement, correspond


def percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else math.nan


def save_preview(
    path: Path,
    rgb: np.ndarray,
    confidence: np.ndarray,
    seed: np.ndarray,
    temporal_error: np.ndarray,
    frame_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 7.5), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"KITTI seq00 frame {frame_id} RGB")
    axes[0, 1].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Pure-stereo confidence")
    axes[1, 0].imshow(seed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("High-confidence Gaussian seed mask")
    axes[1, 1].imshow(np.ma.masked_invalid(temporal_error), cmap="magma", vmin=0.0, vmax=0.3)
    axes[1, 1].set_title("Adjacent-frame relative depth disagreement")
    for axis in axes.flat:
        axis.axis("off")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.loss_threshold <= args.seed_threshold <= 1.0):
        raise ValueError("thresholds must satisfy 0 <= loss <= seed <= 1")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.output_root}")

    left_paths = frame_paths(args.left_image_dir, "*.png", args.num_frames)
    right_paths = frame_paths(args.right_image_dir, "*.png", args.num_frames)
    depth_left_paths = frame_paths(args.depth_left_dir, "*_depth.npy", args.num_frames)
    depth_right_paths = frame_paths(args.depth_right_dir, "*_depth.npy", args.num_frames)
    frame_ids = [path.stem for path in left_paths]
    if frame_ids != [path.stem for path in right_paths]:
        raise ValueError("left/right RGB frame IDs do not match")
    if frame_ids != [depth_frame_id(path) for path in depth_left_paths]:
        raise ValueError("left RGB and left depth frame IDs do not match")
    if frame_ids != [depth_frame_id(path) for path in depth_right_paths]:
        raise ValueError("left RGB and right depth frame IDs do not match")

    first_rgb = cv2.cvtColor(cv2.imread(str(left_paths[0]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if first_rgb is None:
        raise ValueError(f"cannot read {left_paths[0]}")
    shape = first_rgb.shape[:2]
    calibration = parse_calibration(args.calib)
    poses = load_poses(args.pose_path, args.num_frames)
    scale = calibration["fx"] * calibration["baseline_m"]
    mask_paths: list[Path] | None = None
    if args.static_valid_mask_dir is not None:
        mask_paths = frame_paths(args.static_valid_mask_dir, "*_valid.npy", args.num_frames)
        if frame_ids != [path.name.removesuffix("_valid.npy") for path in mask_paths]:
            raise ValueError("left RGB and static-valid mask frame IDs do not match")

    args.output_root.mkdir(parents=True, exist_ok=False)
    confidence_dir = args.output_root / "confidence"
    seed_dir = args.output_root / "seed_mask"
    loss_dir = args.output_root / "loss_mask"
    hard_depth_dir = args.output_root / "depth_confhard"
    previews_dir = args.output_root / "previews"
    for directory in (confidence_dir, seed_dir, loss_dir, hard_depth_dir, previews_dir):
        directory.mkdir()

    depths_left = [load_depth(path, shape) for path in depth_left_paths]
    depths_right = [load_depth(path, shape) for path in depth_right_paths]
    preview_indices = set(args.preview_indices)
    rows: list[dict[str, float | int | str]] = []

    for index, frame_id in enumerate(frame_ids):
        left_bgr = cv2.imread(str(left_paths[index]), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(right_paths[index]), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None or left_bgr.shape[:2] != shape or right_bgr.shape[:2] != shape:
            raise ValueError(f"invalid RGB image at frame {frame_id}")
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        depth = depths_left[index]
        valid = (depth >= args.min_depth) & (depth <= args.max_depth)
        if mask_paths is not None:
            static_valid = np.load(mask_paths[index], allow_pickle=False)
            if static_valid.dtype != np.bool_ or static_valid.shape != shape:
                raise ValueError(f"invalid static-valid mask: {mask_paths[index]}")
            valid &= static_valid

        lr_conf, cycle_error, photo_error, lr_correspond = confidence_from_stereo(
            left, right, depth, depths_right[index], scale, valid, args.lr_cycle_sigma_px, args.photo_sigma
        )
        neighbour_indices = [candidate for candidate in (index - 1, index + 1) if 0 <= candidate < args.num_frames]
        temporal_confs: list[np.ndarray] = []
        temporal_errors: list[np.ndarray] = []
        for neighbour in neighbour_indices:
            temporal_conf, temporal_error, _ = temporal_confidence(
                depth,
                depths_left[neighbour],
                poses[index],
                poses[neighbour],
                calibration,
                valid,
                args.temporal_sigma_rel,
            )
            temporal_confs.append(temporal_conf)
            temporal_errors.append(temporal_error)
        if temporal_confs:
            temporal_conf = np.maximum.reduce(temporal_confs)
            temporal_error = np.fmin.reduce(temporal_errors)
        else:
            temporal_conf = np.ones(shape, dtype=np.float32)
            temporal_error = np.full(shape, np.nan, dtype=np.float32)

        confidence = (lr_conf * temporal_conf).astype(np.float32)
        confidence[~valid] = 0.0
        seed = confidence >= args.seed_threshold
        loss = confidence >= args.loss_threshold
        hard_depth = depth.copy()
        hard_depth[~seed] = 0.0

        np.save(confidence_dir / f"{frame_id}_confidence.npy", confidence, allow_pickle=False)
        np.save(seed_dir / f"{frame_id}_seed_valid.npy", seed, allow_pickle=False)
        np.save(loss_dir / f"{frame_id}_loss_valid.npy", loss, allow_pickle=False)
        np.save(hard_depth_dir / f"{frame_id}_depth.npy", hard_depth.astype(np.float32), allow_pickle=False)

        rows.append(
            {
                "index": index,
                "frame_id": frame_id,
                "raw_valid_ratio": float(((depth >= args.min_depth) & (depth <= args.max_depth)).mean()),
                "mapping_valid_ratio": float(valid.mean()),
                "lr_correspond_ratio": float(lr_correspond.mean()),
                "mean_confidence_all": float(confidence.mean()),
                "mean_confidence_valid": float(confidence[valid].mean()) if valid.any() else math.nan,
                "seed_ratio": float(seed.mean()),
                "loss_ratio": float(loss.mean()),
                "cycle_error_median_px": percentile(cycle_error[lr_correspond], 50),
                "cycle_error_p95_px": percentile(cycle_error[lr_correspond], 95),
                "photo_error_median": percentile(photo_error[lr_correspond], 50),
                "photo_error_p95": percentile(photo_error[lr_correspond], 95),
                "temporal_error_median_rel": percentile(temporal_error, 50),
                "temporal_error_p95_rel": percentile(temporal_error, 95),
            }
        )
        if index in preview_indices:
            save_preview(previews_dir / f"{frame_id}_confidence_preview.png", left, confidence, seed, temporal_error, frame_id)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.num_frames:
            print(f"[{index + 1:03d}/{args.num_frames}] {frame_id} seed={seed.mean():.3f} meanC={confidence.mean():.3f}", flush=True)

    with (args.output_root / "per_frame_confidence_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": f"KITTI seq00 frames 000000-{args.num_frames - 1:06d}",
        "sensor_scope": "pure stereo RGB plus fixed external ORB poses; no external sensors or depth ground truth",
        "calibration": calibration,
        "confidence_formula": "valid_range * optional_static_mask * LR_cycle_photo * temporal_depth_consistency",
        "thresholds": {"seed": args.seed_threshold, "loss": args.loss_threshold},
        "aggregate": {
            field: float(np.nanmean([float(row[field]) for row in rows]))
            for field in rows[0]
            if field not in {"index", "frame_id"}
        },
    }
    with (args.output_root / "confidence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
