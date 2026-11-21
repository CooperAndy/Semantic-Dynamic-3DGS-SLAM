"""Create Step 10 class statistics, time-series plots, and RGB/semantic triptychs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


KEY_CLASSES = ("sky", "car", "person")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument(
        "--sample-offsets",
        type=int,
        nargs="*",
        default=[0, 25, 50, 75, 100, 125, 150, 175, 199],
        help="Zero-based offsets within the selected frame range.",
    )
    parser.add_argument(
        "--no-refresh-manifest",
        action="store_true",
        help="Do not regenerate SHA256SUMS.txt after creating plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_sha256_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and path.name != "verification_report.json"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_metadata(root: Path) -> tuple[dict[int, str], np.ndarray]:
    id2label = {
        int(key): str(value).strip().lower()
        for key, value in load_json(root / "id2label.json").items()
    }
    if sorted(id2label) != list(range(19)):
        raise RuntimeError(f"Expected 19 contiguous class IDs, got {sorted(id2label)}")
    palette_json = load_json(root / "palette.json")
    palette = np.asarray([palette_json[str(index)]["rgb"] for index in range(19)], dtype=np.uint8)
    if palette.shape != (19, 3):
        raise RuntimeError(f"Bad palette shape: {palette.shape}")
    return id2label, palette


def plot_triptych(
    output_path: Path,
    rgb: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    frame_id: str,
) -> None:
    color = palette[labels]
    overlay = np.clip(0.55 * rgb.astype(np.float32) + 0.45 * color, 0, 255).astype(np.uint8)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (rgb, color, overlay),
        ("KITTI left RGB", "Cityscapes-19 semantic labels", "45% semantic overlay"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Step 10 Mask2Former — frame {frame_id}", fontsize=14)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.semantic_dir
    if not args.image_dir.is_dir() or not root.is_dir():
        raise FileNotFoundError("--image-dir and --semantic-dir must both exist")
    if args.start < 0 or args.num_frames <= 0:
        raise ValueError("--start must be >= 0 and --num-frames must be > 0")
    id2label, palette = validate_metadata(root)
    label_to_id = {label: index for index, label in id2label.items()}
    missing_key_classes = [label for label in KEY_CLASSES if label not in label_to_id]
    if missing_key_classes:
        raise RuntimeError(f"Missing key classes: {missing_key_classes}")

    statistics_dir = root / "statistics"
    previews_dir = root / "previews"
    statistics_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)

    frame_ids = [f"{index:06d}" for index in range(args.start, args.start + args.num_frames)]
    class_counts = np.zeros((args.num_frames, 19), dtype=np.int64)
    mean_confidence = np.zeros(args.num_frames, dtype=np.float64)
    dynamic_ratio = np.zeros(args.num_frames, dtype=np.float64)
    frame_pixels = np.zeros(args.num_frames, dtype=np.int64)

    sample_offsets = set(args.sample_offsets)
    invalid_offsets = sorted(offset for offset in sample_offsets if offset < 0 or offset >= args.num_frames)
    if invalid_offsets:
        raise ValueError(f"Sample offsets outside frame range: {invalid_offsets}")

    for offset, frame_id in enumerate(frame_ids):
        image_path = args.image_dir / f"{frame_id}.png"
        label_path = root / "labels" / f"{frame_id}_semantic.npy"
        confidence_path = root / "confidence" / f"{frame_id}_confidence.npy"
        dynamic_path = root / "dynamic_candidates" / f"{frame_id}_dynamic.npy"
        if not all(path.is_file() for path in (image_path, label_path, confidence_path, dynamic_path)):
            raise FileNotFoundError(f"Incomplete inputs for frame {frame_id}")
        labels = np.load(label_path, allow_pickle=False)
        confidence = np.load(confidence_path, allow_pickle=False)
        dynamic = np.load(dynamic_path, allow_pickle=False)
        if labels.ndim != 2 or confidence.shape != labels.shape or dynamic.shape != labels.shape:
            raise RuntimeError(f"Array shape mismatch at frame {frame_id}")
        if labels.dtype != np.uint8 or confidence.dtype != np.float16 or dynamic.dtype != np.bool_:
            raise RuntimeError(f"Array dtype mismatch at frame {frame_id}")
        class_counts[offset] = np.bincount(labels.ravel(), minlength=19)
        mean_confidence[offset] = float(confidence.astype(np.float32).mean())
        dynamic_ratio[offset] = float(dynamic.mean())
        frame_pixels[offset] = labels.size

        if offset in sample_offsets:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != labels.shape:
                raise RuntimeError(f"RGB/label alignment shape mismatch at {frame_id}")
            plot_triptych(
                previews_dir / f"{frame_id}_rgb_semantic_overlay_triptych.png",
                rgb,
                labels,
                palette,
                frame_id,
            )
        if (offset + 1) % 25 == 0 or offset == 0:
            print(f"[stats {offset + 1:03d}/{args.num_frames}] {frame_id}")

    per_frame_ratios = class_counts / frame_pixels[:, None]
    total_counts = class_counts.sum(axis=0)
    overall_ratios = total_counts / total_counts.sum()

    aggregate_path = statistics_dir / "aggregate_class_stats.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_id", "label", "pixel_count", "pixel_ratio", "mean_frame_ratio"),
        )
        writer.writeheader()
        for class_id in range(19):
            writer.writerow(
                {
                    "class_id": class_id,
                    "label": id2label[class_id],
                    "pixel_count": int(total_counts[class_id]),
                    "pixel_ratio": float(overall_ratios[class_id]),
                    "mean_frame_ratio": float(per_frame_ratios[:, class_id].mean()),
                }
            )

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    positions = np.arange(19)
    colors = palette.astype(np.float32) / 255.0
    axis.bar(positions, overall_ratios * 100.0, color=colors, edgecolor="black", linewidth=0.4)
    axis.set_xticks(positions, [id2label[index] for index in positions], rotation=55, ha="right")
    axis.set_ylabel("All-pixel ratio (%)")
    axis.set_title(f"Step 10 Cityscapes-19 distribution — {args.num_frames} KITTI frames")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(statistics_dir / "semantic_class_pixel_ratios.png", dpi=160)
    plt.close(figure)

    x = np.arange(args.start, args.start + args.num_frames)
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for label in KEY_CLASSES:
        axes[0].plot(x, per_frame_ratios[:, label_to_id[label]] * 100.0, label=label)
    axes[0].set_ylabel("Pixels (%)")
    axes[0].set_title("Key semantic classes per frame")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, mean_confidence, color="tab:green")
    axes[1].set_ylabel("Mean confidence")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Selected-class normalized confidence")
    axes[1].grid(alpha=0.25)
    axes[2].plot(x, dynamic_ratio * 100.0, color="tab:red")
    axes[2].set_ylabel("Pixels (%)")
    axes[2].set_xlabel("KITTI frame index")
    axes[2].set_title("Dynamic-category candidate ratio")
    axes[2].grid(alpha=0.25)
    figure.savefig(statistics_dir / "semantic_per_frame_timeseries.png", dpi=160)
    plt.close(figure)

    summary = {
        "frames": args.num_frames,
        "frame_range": [frame_ids[0], frame_ids[-1]],
        "overall_class_pixel_ratios": {
            id2label[index]: float(overall_ratios[index]) for index in range(19)
        },
        "mean_confidence": float(mean_confidence.mean()),
        "mean_dynamic_candidate_ratio": float(dynamic_ratio.mean()),
        "triptych_frames": [frame_ids[offset] for offset in sorted(sample_offsets)],
    }
    (statistics_dir / "semantic_stats_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_refresh_manifest:
        refresh_sha256_manifest(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
