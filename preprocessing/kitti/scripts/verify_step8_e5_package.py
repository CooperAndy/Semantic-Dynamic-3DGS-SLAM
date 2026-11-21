"""Validate Step 8 E5 masks and masked depth without requiring a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--masked-depth-dir", type=Path, required=True)
    parser.add_argument("--raw-depth-dir", type=Path)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--width", type=int, default=1241)
    parser.add_argument("--max-depth", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_shape = (args.height, args.width)
    masks = sorted(args.mask_dir.glob("*_valid.npy"))
    depths = sorted(args.masked_depth_dir.glob("*_depth.npy"))
    if len(masks) != args.num_frames or len(depths) != args.num_frames:
        raise RuntimeError(
            f"Expected {args.num_frames} masks/depths, got {len(masks)}/{len(depths)}"
        )

    valid_ratios: list[float] = []
    total_bytes = 0
    for index in range(args.num_frames):
        frame_id = f"{index:06d}"
        mask_path = args.mask_dir / f"{frame_id}_valid.npy"
        depth_path = args.masked_depth_dir / f"{frame_id}_depth.npy"
        mask = np.load(mask_path, allow_pickle=False)
        depth = np.load(depth_path, allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != expected_shape:
            raise RuntimeError(f"Bad mask dtype/shape: {mask_path}: {mask.dtype} {mask.shape}")
        if depth.dtype != np.float32 or depth.shape != expected_shape:
            raise RuntimeError(f"Bad depth dtype/shape: {depth_path}: {depth.dtype} {depth.shape}")
        if not np.isfinite(depth).all() or (depth < 0).any() or (depth > args.max_depth).any():
            raise RuntimeError(f"Bad depth values: {depth_path}")
        if not np.array_equal(depth > 0, mask):
            raise RuntimeError(f"Mask and positive-depth pixels differ: {frame_id}")
        if args.raw_depth_dir:
            raw = np.load(args.raw_depth_dir / f"{frame_id}_depth.npy", allow_pickle=False)
            if raw.dtype != np.float32 or raw.shape != expected_shape:
                raise RuntimeError(f"Bad raw depth: {frame_id}")
            if not np.allclose(depth[mask], raw[mask], rtol=0.0, atol=0.0):
                raise RuntimeError(f"Masked depth changed valid raw values: {frame_id}")
        valid_ratios.append(float(mask.mean()))
        total_bytes += mask_path.stat().st_size + depth_path.stat().st_size

    ratios = np.asarray(valid_ratios, dtype=np.float64)
    result = {
        "status": "PASS",
        "frames": args.num_frames,
        "shape": list(expected_shape),
        "mask_dtype": "bool",
        "depth_dtype": "float32",
        "depth_range_m": [0.0, args.max_depth],
        "valid_ratio": {
            "mean": float(ratios.mean()),
            "median": float(np.median(ratios)),
            "min": float(ratios.min()),
            "max": float(ratios.max()),
        },
        "mask_and_depth_bytes": total_bytes,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
