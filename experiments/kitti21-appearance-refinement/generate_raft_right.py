"""Generate metric KITTI depth maps with the official RAFT-Stereo model.

The saved depth convention matches the existing SplaTAM KITTI adapter:
float32 metres, HxW, with invalid pixels encoded as 0.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raft-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument(
        "--preview-indices",
        type=int,
        nargs="*",
        default=[0, 25, 50, 75, 100, 125, 150, 175, 199],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_calibration(path: Path) -> dict[str, float]:
    matrices: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            key, *values = raw_line.split()
            matrices[key.rstrip(":")] = np.asarray(values, dtype=np.float64).reshape(3, 4)

    p2, p3 = matrices["P2"], matrices["P3"]
    left_center_x = -p2[0, 3] / p2[0, 0]
    right_center_x = -p3[0, 3] / p3[0, 0]
    return {
        "fx": float(p2[0, 0]),
        "fy": float(p2[1, 1]),
        "cx_left": float(p2[0, 2]),
        "cx_right": float(p3[0, 2]),
        "cy": float(p2[1, 2]),
        "baseline_m": float(abs(right_center_x - left_center_x)),
    }


def load_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda()


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def save_preview(
    path: Path,
    rgb_path: Path,
    disparity: np.ndarray,
    depth: np.ndarray,
    frame_id: str,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    valid_disp = disparity[np.isfinite(disparity) & (disparity > 0)]
    disp_max = percentile_or_nan(valid_disp, 99.0)
    if not np.isfinite(disp_max) or disp_max <= 0:
        disp_max = 1.0

    masked_depth = np.ma.masked_where(depth <= 0, depth)
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"KITTI 00 frame {frame_id} - left RGB")
    axes[1].imshow(disparity, cmap="turbo", vmin=0, vmax=disp_max)
    axes[1].set_title("RAFT-Stereo disparity (pixels; clipped at valid p99)")
    axes[2].imshow(masked_depth, cmap="turbo_r", vmin=0.5, vmax=80.0)
    axes[2].set_title("Metric depth (m; invalid pixels transparent)")
    for axis in axes:
        axis.axis("off")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def model_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation="alt",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    left_paths = sorted((args.sequence_dir / "image_2").glob("*.png"))[: args.num_frames]
    right_paths = sorted((args.sequence_dir / "image_3").glob("*.png"))[: args.num_frames]
    if len(left_paths) != args.num_frames or len(right_paths) != args.num_frames:
        raise RuntimeError(
            f"Expected {args.num_frames} stereo pairs, got {len(left_paths)} left and "
            f"{len(right_paths)} right"
        )
    if [p.stem for p in left_paths] != [p.stem for p in right_paths]:
        raise RuntimeError("Left/right frame IDs do not match")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir()
    calibration = parse_calibration(args.sequence_dir / "calib.txt")
    if abs(calibration["cx_right"] - calibration["cx_left"]) > 1e-6:
        raise RuntimeError("This script expects rectified stereo images with equal principal points")

    sys.path.insert(0, str(args.raft_root))
    sys.path.insert(0, str(args.raft_root / "core"))
    from raft_stereo import RAFTStereo  # pylint: disable=import-error,import-outside-toplevel
    from utils.utils import InputPadder  # pylint: disable=import-error,import-outside-toplevel

    model = torch.nn.DataParallel(RAFTStereo(model_arguments()), device_ids=[0])
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.module.cuda().eval()

    def infer(left_path: Path, right_path: Path) -> tuple[np.ndarray, float]:
        image_left = torch.flip(load_image(right_path), dims=[3])
        image_right = torch.flip(load_image(left_path), dims=[3])
        padder = InputPadder(image_left.shape, divis_by=32)
        image_left, image_right = padder.pad(image_left, image_right)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            _, flow_up = model(
                image_left, image_right, iters=args.valid_iters, test_mode=True
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        disparity = -padder.unpad(flow_up).squeeze().float().cpu().numpy()
        return np.fliplr(disparity).astype(np.float32, copy=False), elapsed_ms

    # One unrecorded warm-up prevents CUDA initialization from biasing the timings.
    print("Running one warm-up inference...", flush=True)
    infer(left_paths[0], right_paths[0])

    rows: list[dict[str, float | int | str]] = []
    preview_indices = set(args.preview_indices)
    expected_shape = np.asarray(Image.open(left_paths[0])).shape[:2]
    scale = calibration["fx"] * calibration["baseline_m"]

    for index, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
        disparity, inference_ms = infer(left_path, right_path)
        if disparity.shape != expected_shape:
            raise RuntimeError(
                f"Frame {left_path.stem}: expected {expected_shape}, got {disparity.shape}"
            )

        depth = np.zeros(disparity.shape, dtype=np.float32)
        positive = np.isfinite(disparity) & (disparity > 0)
        depth[positive] = scale / disparity[positive]
        valid = positive & np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
        depth[~valid] = 0.0

        output_path = args.output_dir / f"{left_path.stem}_depth.npy"
        np.save(output_path, depth, allow_pickle=False)
        loaded = np.load(output_path, allow_pickle=False)
        if loaded.dtype != np.float32 or loaded.shape != expected_shape or not np.isfinite(loaded).all():
            raise RuntimeError(f"Saved output validation failed: {output_path}")

        valid_depth = depth[valid]
        valid_disparity = disparity[positive]
        rows.append(
            {
                "index": index,
                "frame_id": left_path.stem,
                "inference_ms": inference_ms,
                "valid_ratio": float(valid.mean()),
                "disparity_median_px": percentile_or_nan(valid_disparity, 50.0),
                "depth_median_m": percentile_or_nan(valid_depth, 50.0),
                "depth_p05_m": percentile_or_nan(valid_depth, 5.0),
                "depth_p95_m": percentile_or_nan(valid_depth, 95.0),
                "depth_min_m": float(valid_depth.min()) if valid_depth.size else float("nan"),
                "depth_max_m": float(valid_depth.max()) if valid_depth.size else float("nan"),
            }
        )
        if index in preview_indices:
            save_preview(
                preview_dir / f"{left_path.stem}_raft_depth_preview.png",
                left_path,
                disparity,
                depth,
                left_path.stem,
            )
        if (index + 1) % 10 == 0 or index == 0:
            print(
                f"[{index + 1:03d}/{args.num_frames}] {left_path.stem} "
                f"{inference_ms:.1f} ms valid={valid.mean():.3f}",
                flush=True,
            )

    csv_path = args.output_dir / "depth_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    timings = np.asarray([float(row["inference_ms"]) for row in rows])
    valid_ratios = np.asarray([float(row["valid_ratio"]) for row in rows])
    try:
        repo_commit = subprocess.check_output(
            ["git", "-C", str(args.raft_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        repo_commit = "unknown"

    summary = {
        "method": "RAFT-Stereo Middlebury model",
        "frames": len(rows),
        "frame_range": [0, args.num_frames - 1],
        "output_format": "float32 metres, invalid=0.0",
        "shape": list(expected_shape),
        "raw_depth_range_m": [args.min_depth, args.max_depth],
        "valid_iters": args.valid_iters,
        "corr_implementation": "alt",
        "mixed_precision": True,
        "calibration": calibration,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "raft_repo": str(args.raft_root),
        "raft_repo_commit": repo_commit,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "inference_ms": {
            "mean": float(timings.mean()),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "min": float(timings.min()),
            "max": float(timings.max()),
        },
        "valid_ratio": {
            "mean": float(valid_ratios.mean()),
            "median": float(np.median(valid_ratios)),
            "min": float(valid_ratios.min()),
            "max": float(valid_ratios.max()),
        },
        "preview_indices": sorted(preview_indices),
    }
    with (args.output_dir / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
