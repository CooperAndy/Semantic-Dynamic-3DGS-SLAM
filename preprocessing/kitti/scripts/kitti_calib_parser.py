#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def parse_projection_matrix(calib_path, key):
    prefix = f"{key}:"
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue

        values = [float(value) for value in line[len(prefix):].split()]
        if len(values) != 12:
            raise ValueError(f"{key} should contain 12 values, got {len(values)}")

        return [values[0:4], values[4:8], values[8:12]]

    raise KeyError(f"{key} not found in {calib_path}")


def camera_x_from_projection(projection):
    fx = projection[0][0]
    tx = projection[0][3]
    return -tx / fx


def main():
    parser = argparse.ArgumentParser(description="Parse KITTI calib.txt P2/P3 parameters.")
    parser.add_argument("calib_txt", type=Path, help="Path to KITTI calib.txt")
    args = parser.parse_args()

    if not args.calib_txt.is_file():
        print(f"calib.txt not found: {args.calib_txt}", file=sys.stderr)
        return 1

    try:
        p2 = parse_projection_matrix(args.calib_txt, "P2")
        p3 = parse_projection_matrix(args.calib_txt, "P3")
    except (KeyError, ValueError) as exc:
        print(f"Failed to parse calibration file: {exc}", file=sys.stderr)
        return 1

    fx = p2[0][0]
    fy = p2[1][1]
    cx = p2[0][2]
    cy = p2[1][2]

    # KITTI odometry P2 has a non-zero x offset, so the stereo baseline is the
    # relative camera-center distance between P2 and P3.
    p2_x = camera_x_from_projection(p2)
    p3_x = camera_x_from_projection(p3)
    baseline = abs(p3_x - p2_x)

    print(f"fx: {fx:.6f}")
    print(f"fy: {fy:.6f}")
    print(f"cx: {cx:.6f}")
    print(f"cy: {cy:.6f}")
    print(f"baseline: {baseline:.6f} m")

    if baseline < 0.1 or baseline > 1.0:
        print(
            f"Baseline sanity check failed: expected roughly 0.537 m, got {baseline:.6f} m",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
