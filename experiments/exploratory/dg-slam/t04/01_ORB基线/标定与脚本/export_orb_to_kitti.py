#!/usr/bin/env python3

import argparse
import math
from pathlib import Path


def load_kitti_poses(path: Path):
    poses = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            fields = line.split()

            if len(fields) != 12:
                raise ValueError(
                    f"{path}: line {line_number} contains "
                    f"{len(fields)} values; expected 12"
                )

            values = [float(value) for value in fields]

            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}: line {line_number} contains NaN or Inf"
                )

            poses.append(values)

    return poses


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate an ORB-SLAM3 SaveTrajectoryKITTI trajectory "
            "and export it as a normalized KITTI 12-column pose file. "
            "The input is already Twc, so no pose inversion is applied."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    poses = load_kitti_poses(args.input)

    if args.expected_rows is not None and len(poses) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} poses, but found {len(poses)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for pose in poses:
            file.write(" ".join(f"{value:.9f}" for value in pose) + "\n")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Poses:  {len(poses)}")
    print("Format: KITTI 12-column [Rwc|twc]")
    print("Direction: Twc (camera-to-world)")
    print("Pose inversion applied: no")
    print("Validation: passed")


if __name__ == "__main__":
    main()