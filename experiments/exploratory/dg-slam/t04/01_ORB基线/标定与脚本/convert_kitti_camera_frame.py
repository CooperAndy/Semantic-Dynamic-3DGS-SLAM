#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def load_calibration(path: Path):
    calibration = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue

        name, values = line.split(":", 1)
        calibration[name] = np.fromstring(values, sep=" ")

    return calibration


def projection_translation(values):
    projection = values.reshape(3, 4)
    intrinsic = projection[:, :3]
    return np.linalg.solve(intrinsic, projection[:, 3])


def main():
    parser = argparse.ArgumentParser(
        description="Convert KITTI poses between rectified camera frames."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--source-camera", default="P2")
    parser.add_argument("--target-camera", default="P0")
    args = parser.parse_args()

    poses = np.loadtxt(args.input, dtype=np.float64)

    if poses.ndim == 1:
        poses = poses.reshape(1, -1)

    if poses.shape[1] != 12:
        raise ValueError(
            f"Expected 12 columns, received shape {poses.shape}"
        )

    if not np.isfinite(poses).all():
        raise ValueError("Input trajectory contains NaN or Inf")

    calibration = load_calibration(args.calib)

    if args.source_camera not in calibration:
        raise KeyError(f"Missing {args.source_camera} in calibration")

    if args.target_camera not in calibration:
        raise KeyError(f"Missing {args.target_camera} in calibration")

    source_translation = projection_translation(
        calibration[args.source_camera]
    )
    target_translation = projection_translation(
        calibration[args.target_camera]
    )

    # T_source_target for KITTI rectified cameras.
    source_to_target = source_translation - target_translation

    matrices = np.zeros((len(poses), 4, 4), dtype=np.float64)
    matrices[:, 3, 3] = 1.0
    matrices[:, :3, :4] = poses.reshape(-1, 3, 4)

    # Tw_target = Tw_source @ T_source_target
    matrices[:, :3, 3] += (
        matrices[:, :3, :3] @ source_to_target
    )

    output = matrices[:, :3, :4].reshape(-1, 12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, output, fmt="%.9f")

    print(f"Input poses: {len(poses)}")
    print(f"Source camera: {args.source_camera}")
    print(f"Target camera: {args.target_camera}")
    print(f"Translation: {source_to_target}")
    print(f"Output: {args.output}")
    print("Validation: passed")


if __name__ == "__main__":
    main()