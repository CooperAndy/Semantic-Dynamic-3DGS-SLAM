"""Generate Mask2Former Cityscapes semantics for a contiguous KITTI frame range.

The script is safe by default: it validates every input before model loading and
refuses to write into a non-empty output directory.  It stores compact labels,
selected-class confidence, and dynamic-category candidates rather than the full
19-channel score tensor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = "facebook/mask2former-swin-small-cityscapes-semantic"
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
CITYSCAPES_PALETTE = (
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-source-id",
        default=DEFAULT_MODEL_ID,
        help="Original model repository ID recorded when --model-id is a local path.",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face revision/commit. Omit to use the repository default.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face hub cache directory (HF_HOME/hub). Prefer the AutoDL data disk.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and load the pinned model only from cache.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-every", type=int, default=25)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and hash RGB inputs without importing torch/transformers.",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Optional JSON destination used with --preflight-only.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_and_audit_images(
    image_dir: Path, start: int, num_frames: int
) -> tuple[list[Path], tuple[int, int], dict[str, str]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    if num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [image_dir / f"{index:06d}.png" for index in range(start, start + num_frames)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(f"Missing {len(missing)} required RGB files: {missing[:10]}{suffix}")

    expected_shape: tuple[int, int] | None = None
    hashes: dict[str, str] = {}
    for offset, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.convert("RGB").load()
        except Exception as exc:
            raise RuntimeError(f"Unreadable RGB file: {path}: {exc}") from exc
        shape = (height, width)
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            raise RuntimeError(
                f"Inconsistent RGB shape at {path.name}: {shape}, expected {expected_shape}"
            )
        hashes[path.name] = sha256_file(path)
        if (offset + 1) % 25 == 0 or offset == 0:
            print(f"[preflight {offset + 1:03d}/{num_frames}] {path.name} {width}x{height}")

    assert expected_shape is not None
    return paths, expected_shape, hashes


def preflight_payload(
    image_dir: Path,
    paths: list[Path],
    shape: tuple[int, int],
    hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "checked_at_utc": utc_now(),
        "image_dir": str(image_dir.resolve()),
        "frame_count": len(paths),
        "first_frame": paths[0].name,
        "last_frame": paths[-1].name,
        "height": shape[0],
        "width": shape[1],
        "rgb_sha256": hashes,
    }


def normalize_id2label(raw_mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value).strip().lower() for key, value in raw_mapping.items()}


def validate_cityscapes_mapping(id2label: dict[int, str]) -> None:
    expected = {index: label for index, label in enumerate(CITYSCAPES_LABELS)}
    if id2label != expected:
        raise RuntimeError(
            "Model id2label is not the required Cityscapes-19 mapping.\n"
            f"Expected: {expected}\nActual: {id2label}"
        )


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "numpy",
        "Pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_overlay(
    output_path: Path,
    rgb: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    frame_id: str,
) -> None:
    color = palette[labels]
    overlay = np.clip(0.55 * rgb.astype(np.float32) + 0.45 * color, 0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def refresh_sha256_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def semantic_output_with_scores(processor: Any, outputs: Any, target_size: tuple[int, int]):
    """Return labels and class evidence, supporting old and new Transformers."""
    import torch

    try:
        result = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[target_size],
            return_segmentation_scores=True,
        )[0]
        if hasattr(result, "segmentation"):
            return result.segmentation, result.segmentation_scores
        return result["segmentation"], result["segmentation_scores"]
    except TypeError:
        labels = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size]
        )[0]
        class_prob = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        processor_size = processor.size
        if not isinstance(processor_size, dict) or not {
            "height",
            "width",
        }.issubset(processor_size):
            raise RuntimeError(f"Unsupported Mask2Former processor size: {processor_size}")
        resized_mask_logits = torch.nn.functional.interpolate(
            outputs.masks_queries_logits,
            size=(int(processor_size["height"]), int(processor_size["width"])),
            mode="bilinear",
            align_corners=False,
        )
        mask_prob = resized_mask_logits.sigmoid()
        evidence = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
        evidence = torch.nn.functional.interpolate(
            evidence, size=target_size, mode="bilinear", align_corners=False
        )[0]
        if not torch.equal(labels, evidence.argmax(dim=0)):
            raise RuntimeError(
                "Installed Transformers post-processing differs from the confidence "
                "fallback. Upgrade Transformers so semantic scores are returned directly."
            )
        return labels, evidence


def main() -> None:
    args = parse_args()
    paths, shape, hashes = select_and_audit_images(
        args.image_dir, args.start, args.num_frames
    )
    audit = preflight_payload(args.image_dir, paths, shape, hashes)
    if args.preflight_only:
        if args.preflight_report:
            args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.preflight_report, audit)
        print(json.dumps({key: value for key, value in audit.items() if key != "rgb_sha256"}, indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --preflight-only is used")
    if args.preview_every <= 0:
        raise ValueError("--preview-every must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    try:
        import torch
        import transformers
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "Generation requires torch and transformers in the isolated semantic environment"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    load_kwargs: dict[str, Any] = {}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        load_kwargs["local_files_only"] = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(
        args.model_id, use_fast=False, **load_kwargs
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_id, use_safetensors=True, **load_kwargs
    )
    model.to(device).eval()
    load_seconds = time.perf_counter() - load_started

    id2label = normalize_id2label(model.config.id2label)
    validate_cityscapes_mapping(id2label)
    dynamic_ids = sorted(index for index, label in id2label.items() if label in DYNAMIC_LABELS)
    palette = np.asarray(CITYSCAPES_PALETTE, dtype=np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = args.output_dir / "labels"
    confidence_dir = args.output_dir / "confidence"
    dynamic_dir = args.output_dir / "dynamic_candidates"
    previews_dir = args.output_dir / "previews"
    for directory in (labels_dir, confidence_dir, dynamic_dir, previews_dir):
        directory.mkdir()

    write_json(args.output_dir / "id2label.json", {str(k): v for k, v in id2label.items()})
    write_json(
        args.output_dir / "palette.json",
        {
            str(index): {"label": id2label[index], "rgb": list(CITYSCAPES_PALETTE[index])}
            for index in range(len(CITYSCAPES_LABELS))
        },
    )
    write_json(args.output_dir / "input_rgb_sha256.json", audit)
    model.config.to_json_file(args.output_dir / "model_config.json", use_diff=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "NONE (CPU inference)"

    rows: list[dict[str, Any]] = []
    total_inference_seconds = 0.0
    run_started = time.perf_counter()
    for offset, image_path in enumerate(paths):
        from resource_guard import wait_for_headroom
        wait_for_headroom("semantic", offset)
        frame_id = image_path.stem
        with Image.open(image_path) as image_handle:
            pil_image = image_handle.convert("RGB")
            rgb = np.asarray(pil_image, dtype=np.uint8)
            inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        total_inference_seconds += inference_seconds

        label_tensor, evidence = semantic_output_with_scores(processor, outputs, shape)
        evidence_sum = evidence.sum(dim=0).clamp_min(torch.finfo(evidence.dtype).eps)
        normalized = evidence / evidence_sum.unsqueeze(0)
        selected = normalized.gather(0, label_tensor.unsqueeze(0)).squeeze(0)
        labels = label_tensor.detach().cpu().numpy().astype(np.uint8, copy=False)
        confidence = selected.detach().cpu().numpy().astype(np.float16, copy=False)
        dynamic = np.isin(labels, dynamic_ids)

        if labels.shape != shape or confidence.shape != shape:
            raise RuntimeError(f"Post-process shape mismatch at {frame_id}")
        if not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1:
            raise RuntimeError(f"Invalid normalized confidence at {frame_id}")

        np.save(labels_dir / f"{frame_id}_semantic.npy", labels, allow_pickle=False)
        np.save(
            confidence_dir / f"{frame_id}_confidence.npy", confidence, allow_pickle=False
        )
        np.save(dynamic_dir / f"{frame_id}_dynamic.npy", dynamic, allow_pickle=False)
        if offset % args.preview_every == 0 or offset == len(paths) - 1:
            save_overlay(
                previews_dir / f"{frame_id}_overlay.png", rgb, labels, palette, frame_id
            )

        row: dict[str, Any] = {
            "index": offset,
            "frame_id": frame_id,
            "inference_seconds": inference_seconds,
            "mean_confidence": float(confidence.astype(np.float32).mean()),
            "min_confidence": float(confidence.min()),
            "dynamic_candidate_ratio": float(dynamic.mean()),
        }
        for class_id, label in id2label.items():
            row[f"ratio_{class_id:02d}_{label.replace(' ', '_')}"] = float(
                (labels == class_id).mean()
            )
        rows.append(row)
        del outputs, evidence, normalized, selected, inputs
        if (offset + 1) % 10 == 0 or offset == 0:
            print(
                f"[{offset + 1:03d}/{len(paths)}] {frame_id} "
                f"infer={inference_seconds:.3f}s confidence={row['mean_confidence']:.3f} "
                f"dynamic={row['dynamic_candidate_ratio']:.3%}",
                flush=True,
            )

    total_seconds = time.perf_counter() - run_started
    with (args.output_dir / "per_frame_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision or "UNKNOWN"
    checkpoint_path: str | None
    model_id_path = Path(args.model_id)
    if model_id_path.is_dir():
        local_checkpoint = model_id_path / "model.safetensors"
        checkpoint_path = str(local_checkpoint) if local_checkpoint.is_file() else None
    else:
        cached_checkpoint = try_to_load_from_cache(
            args.model_id,
            "model.safetensors",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            revision=resolved_revision if resolved_revision != "UNKNOWN" else args.revision,
        )
        checkpoint_path = (
            str(cached_checkpoint)
            if isinstance(cached_checkpoint, str) and Path(cached_checkpoint).is_file()
            else None
        )
    checkpoint_metadata = {
        "path": checkpoint_path,
        "bytes": Path(checkpoint_path).stat().st_size if checkpoint_path else None,
        "sha256": sha256_file(Path(checkpoint_path)) if checkpoint_path else None,
    }
    summary = {
        "status": "COMPLETE_PENDING_VERIFICATION",
        "created_at_utc": utc_now(),
        "command": shlex.join(sys.argv),
        "model_id": args.model_source_id,
        "model_load_path": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
        "checkpoint": checkpoint_metadata,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "id2label": {str(k): v for k, v in id2label.items()},
        "dynamic_labels": sorted(DYNAMIC_LABELS),
        "dynamic_class_ids": dynamic_ids,
        "confidence_definition": (
            "selected class Mask2Former semantic evidence divided by the per-pixel "
            "sum of all 19 class evidence values"
        ),
        "input": {
            "image_dir": str(args.image_dir.resolve()),
            "start": args.start,
            "num_frames": args.num_frames,
            "first_frame": paths[0].name,
            "last_frame": paths[-1].name,
            "height": shape[0],
            "width": shape[1],
        },
        "runtime": {
            "device": str(device),
            "gpu": gpu_name,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "pid": os.getpid(),
            "seed": args.seed,
            "package_versions": package_versions(),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "generation_wall_seconds": total_seconds,
            "inference_total_seconds": total_inference_seconds,
            "mean_inference_seconds_per_frame": total_inference_seconds / len(paths),
            "inference_fps": len(paths) / total_inference_seconds,
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    write_json(args.output_dir / "generation_summary.json", summary)
    refresh_sha256_manifest(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
