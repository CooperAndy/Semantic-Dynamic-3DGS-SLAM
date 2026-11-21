"""Validate Step 10 KITTI Mask2Former outputs without requiring a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CITYSCAPES_LABELS = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
DYNAMIC_LABELS = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--width", type=int, default=1241)
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Audit RGB names, decode, dimensions, and SHA-256 without semantic output.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip SHA256SUMS verification (not acceptable for final delivery).",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def audit_rgb(
    image_dir: Path,
    start: int,
    num_frames: int,
    expected_shape: tuple[int, int],
) -> tuple[list[Path], dict[str, str]]:
    if start < 0 or num_frames <= 0:
        raise ValueError("--start must be >= 0 and --num-frames must be > 0")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    paths = [image_dir / f"{index:06d}.png" for index in range(start, start + num_frames)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} RGB inputs: {missing[:10]}")

    hashes: dict[str, str] = {}
    for offset, path in enumerate(paths):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            actual_shape = (image.height, image.width)
            image.convert("RGB").load()
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Bad RGB shape {path.name}: {actual_shape}, expected {expected_shape}"
            )
        hashes[path.name] = sha256_file(path)
        if (offset + 1) % 50 == 0 or offset == 0:
            print(f"[RGB {offset + 1:03d}/{num_frames}] {path.name}", file=sys.stderr)
    return paths, hashes


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing required metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_mapping(raw: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value).strip().lower() for key, value in raw.items()}


def verify_manifest(root: Path) -> dict[str, int]:
    manifest_path = root / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing SHA256 manifest: {manifest_path}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed manifest line {line_number}: {line!r}") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Bad SHA-256 at manifest line {line_number}")
        relative = relative.replace("\\", "/")
        if relative in expected:
            raise RuntimeError(f"Duplicate manifest entry: {relative}")
        expected[relative] = digest

    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and path.name != "verification_report.json"
    }
    if set(expected) != set(actual_files):
        missing = sorted(set(actual_files) - set(expected))
        stale = sorted(set(expected) - set(actual_files))
        raise RuntimeError(
            f"Manifest file set mismatch; unlisted={missing[:10]}, missing={stale[:10]}"
        )
    for offset, relative in enumerate(sorted(expected)):
        actual_digest = sha256_file(actual_files[relative])
        if actual_digest != expected[relative]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        if (offset + 1) % 100 == 0:
            print(
                f"[SHA256 {offset + 1:03d}/{len(expected)}] {relative}", file=sys.stderr
            )
    return {"files": len(expected), "bytes": sum(path.stat().st_size for path in actual_files.values())}


def verify_outputs(
    args: argparse.Namespace,
    paths: list[Path],
    rgb_hashes: dict[str, str],
) -> dict[str, Any]:
    if args.semantic_dir is None:
        raise ValueError("--semantic-dir is required unless --input-only is used")
    root = args.semantic_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Missing semantic output directory: {root}")
    expected_shape = (args.height, args.width)
    expected_ids = [path.stem for path in paths]
    expected_mapping = {index: label for index, label in enumerate(CITYSCAPES_LABELS)}

    id2label = normalized_mapping(load_json(root / "id2label.json"))
    if id2label != expected_mapping:
        raise RuntimeError(f"id2label.json is not Cityscapes-19: {id2label}")
    model_config = load_json(root / "model_config.json")
    config_mapping = normalized_mapping(model_config.get("id2label", {}))
    if config_mapping != id2label:
        raise RuntimeError("id2label.json differs from saved model_config.json")
    summary = load_json(root / "generation_summary.json")
    summary_mapping = normalized_mapping(summary.get("id2label", {}))
    if summary_mapping != id2label:
        raise RuntimeError("id2label.json differs from generation_summary.json")
    if summary.get("input", {}).get("num_frames") != args.num_frames:
        raise RuntimeError("generation_summary num_frames differs from verifier request")
    if summary.get("input", {}).get("start") != args.start:
        raise RuntimeError("generation_summary start differs from verifier request")

    rgb_baseline = load_json(root / "input_rgb_sha256.json")
    saved_hashes = rgb_baseline.get("rgb_sha256", {})
    if saved_hashes != rgb_hashes:
        changed = sorted(
            name
            for name in set(saved_hashes) | set(rgb_hashes)
            if saved_hashes.get(name) != rgb_hashes.get(name)
        )
        raise RuntimeError(f"Original RGB SHA-256 changed or frame set differs: {changed[:10]}")

    dirs_and_suffixes = {
        "labels": "_semantic.npy",
        "confidence": "_confidence.npy",
        "dynamic_candidates": "_dynamic.npy",
    }
    for directory_name, suffix in dirs_and_suffixes.items():
        directory = root / directory_name
        actual = sorted(path.name for path in directory.glob("*.npy")) if directory.is_dir() else []
        expected = [f"{frame_id}{suffix}" for frame_id in expected_ids]
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise RuntimeError(
                f"{directory_name} frame set mismatch: count={len(actual)}, "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    dynamic_ids = np.asarray(
        [index for index, label in id2label.items() if label in DYNAMIC_LABELS],
        dtype=np.uint8,
    )
    class_pixels = np.zeros(len(CITYSCAPES_LABELS), dtype=np.int64)
    confidence_sum = 0.0
    dynamic_pixels = 0
    total_pixels = 0
    total_array_bytes = 0
    frame_confidences: list[float] = []
    frame_dynamic_ratios: list[float] = []

    for offset, frame_id in enumerate(expected_ids):
        label_path = root / "labels" / f"{frame_id}_semantic.npy"
        confidence_path = root / "confidence" / f"{frame_id}_confidence.npy"
        dynamic_path = root / "dynamic_candidates" / f"{frame_id}_dynamic.npy"
        labels = np.load(label_path, allow_pickle=False)
        confidence = np.load(confidence_path, allow_pickle=False)
        dynamic = np.load(dynamic_path, allow_pickle=False)

        if labels.dtype != np.uint8 or labels.shape != expected_shape:
            raise RuntimeError(f"Bad labels dtype/shape: {label_path}: {labels.dtype} {labels.shape}")
        if confidence.dtype != np.float16 or confidence.shape != expected_shape:
            raise RuntimeError(
                f"Bad confidence dtype/shape: {confidence_path}: "
                f"{confidence.dtype} {confidence.shape}"
            )
        if dynamic.dtype != np.bool_ or dynamic.shape != expected_shape:
            raise RuntimeError(
                f"Bad dynamic dtype/shape: {dynamic_path}: {dynamic.dtype} {dynamic.shape}"
            )
        if labels.min() < 0 or labels.max() >= len(CITYSCAPES_LABELS):
            raise RuntimeError(f"Label ID outside [0,18]: {label_path}")
        if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
            raise RuntimeError(f"Confidence outside finite [0,1]: {confidence_path}")
        expected_dynamic = np.isin(labels, dynamic_ids)
        if not np.array_equal(dynamic, expected_dynamic):
            mismatch_count = int(np.count_nonzero(dynamic != expected_dynamic))
            raise RuntimeError(
                f"Dynamic mask differs from labels at {frame_id}: {mismatch_count} pixels"
            )

        counts = np.bincount(labels.ravel(), minlength=len(CITYSCAPES_LABELS))
        class_pixels += counts
        frame_mean_confidence = float(confidence.astype(np.float32).mean())
        frame_dynamic_ratio = float(dynamic.mean())
        frame_confidences.append(frame_mean_confidence)
        frame_dynamic_ratios.append(frame_dynamic_ratio)
        confidence_sum += float(confidence.astype(np.float64).sum())
        dynamic_pixels += int(dynamic.sum())
        total_pixels += labels.size
        total_array_bytes += sum(
            path.stat().st_size for path in (label_path, confidence_path, dynamic_path)
        )
        if (offset + 1) % 25 == 0 or offset == 0:
            print(f"[arrays {offset + 1:03d}/{args.num_frames}] {frame_id}", file=sys.stderr)

    stats_path = root / "per_frame_stats.csv"
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        raise RuntimeError("Missing or empty per_frame_stats.csv")
    if not (root / "palette.json").is_file():
        raise RuntimeError("Missing palette.json")
    previews = sorted((root / "previews").glob("*.png"))
    if not previews:
        raise RuntimeError("No PNG previews found")

    manifest = None if args.skip_manifest else verify_manifest(root)
    return {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames": args.num_frames,
        "frame_range": [expected_ids[0], expected_ids[-1]],
        "shape": list(expected_shape),
        "dtypes": {"labels": "uint8", "confidence": "float16", "dynamic": "bool"},
        "original_rgb_sha256": "PASS",
        "id2label_vs_saved_model_config": "PASS",
        "dynamic_mask_exact_match": "PASS",
        "confidence_finite_range_0_1": "PASS",
        "mean_confidence": confidence_sum / total_pixels,
        "per_frame_mean_confidence_range": [
            min(frame_confidences),
            max(frame_confidences),
        ],
        "dynamic_candidate_ratio": dynamic_pixels / total_pixels,
        "per_frame_dynamic_ratio_range": [
            min(frame_dynamic_ratios),
            max(frame_dynamic_ratios),
        ],
        "class_pixel_ratios": {
            id2label[index]: int(class_pixels[index]) / total_pixels
            for index in range(len(CITYSCAPES_LABELS))
        },
        "preview_png_count": len(previews),
        "semantic_array_bytes": total_array_bytes,
        "sha256_manifest": manifest if manifest is not None else "SKIPPED",
        "model_id": summary.get("model_id"),
        "resolved_revision": summary.get("resolved_revision"),
    }


def main() -> None:
    args = parse_args()
    expected_shape = (args.height, args.width)
    try:
        paths, hashes = audit_rgb(
            args.image_dir, args.start, args.num_frames, expected_shape
        )
        if args.input_only:
            report = {
                "status": "PASS",
                "mode": "RGB_INPUT_ONLY",
                "image_dir": str(args.image_dir.resolve()),
                "frames": args.num_frames,
                "first_frame": paths[0].name,
                "last_frame": paths[-1].name,
                "shape": list(expected_shape),
                "decoded_all_images": True,
                "sha256_all_images": True,
                "rgb_sha256": hashes,
            }
        else:
            report = verify_outputs(args, paths, hashes)
        write_report(args.report, report)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_report(args.report, report)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
