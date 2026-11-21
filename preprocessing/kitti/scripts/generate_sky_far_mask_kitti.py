"""Generate conservative sky/far masks and masked RAFT depth for KITTI.

The method is deliberately deterministic and does not require a semantic model:

* invalid and far depth are removed geometrically;
* sky candidates must be in the upper image, blue-dominant, and connected to
  the top border, which avoids masking isolated blue vehicles/windows;
* the original RGB and RAFT depth files are never modified.

Outputs are directly compatible with the existing SplaTAM KITTI adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--far-depth", type=float, default=40.0)
    parser.add_argument("--sky-upper-fraction", type=float, default=0.58)
    parser.add_argument("--sky-hue-min", type=int, default=90)
    parser.add_argument("--sky-hue-max", type=int, default=120)
    parser.add_argument("--sky-saturation-min", type=int, default=95)
    parser.add_argument("--sky-value-min", type=int, default=150)
    parser.add_argument("--blue-red-margin", type=int, default=40)
    parser.add_argument("--blue-green-margin", type=int, default=12)
    parser.add_argument("--top-seed-fraction", type=float, default=0.08)
    parser.add_argument("--minimum-component-area", type=int, default=30)
    parser.add_argument(
        "--preview-indices",
        type=int,
        nargs="*",
        default=[0, 25, 50, 75, 100, 125, 150, 175, 199],
    )
    return parser.parse_args()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def top_connected_sky_mask(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)

    rows = np.arange(height)[:, None]
    upper = rows < int(round(height * args.sky_upper_fraction))
    candidate = (
        upper
        & (hue >= args.sky_hue_min)
        & (hue <= args.sky_hue_max)
        & (saturation >= args.sky_saturation_min)
        & (value >= args.sky_value_min)
        & (blue >= red + args.blue_red_margin)
        & (blue >= green + args.blue_green_margin)
    )

    candidate_u8 = candidate.astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_u8, connectivity=8
    )
    seed_limit = max(1, int(round(height * args.top_seed_fraction)))
    sky = np.zeros((height, width), dtype=bool)
    for label_id in range(1, component_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        component = labels == label_id
        touches_top_seed = bool(component[:seed_limit].any())
        if area >= args.minimum_component_area and touches_top_seed:
            sky |= component

    # Grow by one pixel only inside a relaxed blue candidate region, reducing
    # thin holes while protecting non-blue building/tree boundaries.
    relaxed_blue = (
        upper
        & (saturation >= 75)
        & (blue >= red + 28)
        & (blue >= green + 8)
        & (value >= 130)
    )
    grown = cv2.dilate(sky.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    return sky | (grown & relaxed_blue)


def save_preview(
    path: Path,
    rgb: np.ndarray,
    raw_depth: np.ndarray,
    masked_depth: np.ndarray,
    sky_mask: np.ndarray,
    far_mask: np.ndarray,
    frame_id: str,
) -> None:
    raw_plot = np.ma.masked_where(raw_depth <= 0, raw_depth)
    masked_plot = np.ma.masked_where(masked_depth <= 0, masked_depth)
    overlay = rgb.copy()
    overlay[sky_mask] = (0.35 * overlay[sky_mask] + 0.65 * np.array([255, 0, 255])).astype(
        np.uint8
    )
    overlay[far_mask] = (0.35 * overlay[far_mask] + 0.65 * np.array([255, 215, 0])).astype(
        np.uint8
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"KITTI 00 frame {frame_id} - left RGB")
    axes[0, 1].imshow(raw_plot, cmap="turbo_r", vmin=0.5, vmax=40.0)
    axes[0, 1].set_title("Raw RAFT depth (display clipped at 40 m)")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Removed pixels: sky=magenta, far=yellow")
    axes[1, 1].imshow(masked_plot, cmap="turbo_r", vmin=0.5, vmax=40.0)
    axes[1, 1].set_title("E5 masked RAFT depth")
    for axis in axes.flat:
        axis.axis("off")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {args.output_root}")

    image_paths = sorted(args.image_dir.glob("*.png"))[: args.num_frames]
    depth_paths = sorted(args.depth_dir.glob("*_depth.npy"))[: args.num_frames]
    if len(image_paths) != args.num_frames or len(depth_paths) != args.num_frames:
        raise RuntimeError(
            f"Expected {args.num_frames} inputs, got {len(image_paths)} images and "
            f"{len(depth_paths)} depth files"
        )
    if [p.stem for p in image_paths] != [p.name.removesuffix("_depth.npy") for p in depth_paths]:
        raise RuntimeError("RGB and depth frame IDs do not match")

    args.output_root.mkdir(parents=True, exist_ok=False)
    mask_dir = args.output_root / "masks_static_valid"
    masked_depth_dir = args.output_root / "depth_raft_skyfar_mask_200"
    preview_dir = args.output_root / "previews"
    for directory in (mask_dir, masked_depth_dir, preview_dir):
        directory.mkdir()

    rows: list[dict[str, float | int | str]] = []
    preview_indices = set(args.preview_indices)
    expected_shape: tuple[int, int] | None = None

    for index, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        frame_id = image_path.stem
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        raw_depth = np.load(depth_path, allow_pickle=False)
        if expected_shape is None:
            expected_shape = rgb.shape[:2]
        if rgb.shape[:2] != expected_shape or raw_depth.shape != expected_shape:
            raise RuntimeError(f"Shape mismatch at frame {frame_id}")
        if raw_depth.dtype != np.float32 or not np.isfinite(raw_depth).all():
            raise RuntimeError(f"Invalid RAFT depth at frame {frame_id}")

        raw_valid = (raw_depth >= args.min_depth) & (raw_depth <= 80.0)
        far_mask = raw_valid & (raw_depth > args.far_depth)
        sky_mask = top_connected_sky_mask(rgb, args) & raw_valid
        mapping_valid = raw_valid & ~far_mask & ~sky_mask
        masked_depth = raw_depth.copy()
        masked_depth[~mapping_valid] = 0.0

        np.save(mask_dir / f"{frame_id}_valid.npy", mapping_valid, allow_pickle=False)
        np.save(
            masked_depth_dir / f"{frame_id}_depth.npy",
            masked_depth.astype(np.float32, copy=False),
            allow_pickle=False,
        )

        valid_depth = masked_depth[mapping_valid]
        rows.append(
            {
                "index": index,
                "frame_id": frame_id,
                "raw_valid_ratio_0p5_80m": float(raw_valid.mean()),
                "e4_backend_valid_ratio_0p5_40m": float((raw_valid & ~far_mask).mean()),
                "e5_valid_ratio": float(mapping_valid.mean()),
                "far_removed_ratio": float(far_mask.mean()),
                "sky_removed_ratio": float(sky_mask.mean()),
                "total_removed_from_raw_ratio": float((raw_valid & ~mapping_valid).mean()),
                "e5_depth_median_m": percentile(valid_depth, 50.0),
                "e5_depth_p95_m": percentile(valid_depth, 95.0),
            }
        )
        if index in preview_indices:
            save_preview(
                preview_dir / f"{frame_id}_sky_far_mask_preview.png",
                rgb,
                raw_depth,
                masked_depth,
                sky_mask,
                far_mask,
                frame_id,
            )
        if (index + 1) % 25 == 0 or index == 0:
            print(
                f"[{index + 1:03d}/{args.num_frames}] {frame_id} "
                f"raw={raw_valid.mean():.3f} e5={mapping_valid.mean():.3f} "
                f"sky={sky_mask.mean():.3f} far={far_mask.mean():.3f}",
                flush=True,
            )

    stats_path = args.output_root / "mask_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, float]] = {}
    numeric_keys = list(rows[0].keys())[2:]
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }

    settings = {
        key: getattr(args, key)
        for key in (
            "num_frames",
            "min_depth",
            "far_depth",
            "sky_upper_fraction",
            "sky_hue_min",
            "sky_hue_max",
            "sky_saturation_min",
            "sky_value_min",
            "blue_red_margin",
            "blue_green_margin",
            "top_seed_fraction",
            "minimum_component_area",
        )
    }
    summary = {
        "scope": "KITTI odometry sequence 00, frames 000000-000199",
        "method": "conservative top-connected blue-sky mask + explicit far-depth cutoff",
        "mask_format": "bool HxW; true means valid for mapping",
        "masked_depth_format": "float32 metres HxW; invalid=0.0",
        "settings": settings,
        "aggregate": aggregate,
        "preview_indices": sorted(preview_indices),
        "limitations": [
            "This is a deterministic heuristic, not semantic segmentation.",
            "Only blue sky connected to the image top is removed; overcast sky may remain.",
            "Final benefit must be measured by the pending E5 SplaTAM GPU run.",
        ],
    }
    with (args.output_root / "mask_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
