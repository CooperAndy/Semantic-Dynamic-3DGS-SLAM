"""Generate right-camera metric depth for the 9R pure-stereo confidence audit.

RAFT-Stereo is directional: it expects its matching view to lie to the left
in image coordinates.  Therefore ``image_3`` and ``image_2`` are both
horizontally mirrored before inference, then the predicted disparity is
mirrored back.  The output is a right-view depth map named
``<frame>_depth.npy`` (float32 metres; zero means invalid).  It complements,
rather than replaces, the archived E4 left-view RAFT depth maps.

The script deliberately creates a new, empty output directory and stores its
provenance.  It requires the existing CUDA RAFT-Stereo environment.
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
    baseline = abs((-p3[0, 3] / p3[0, 0]) - (-p2[0, 3] / p2[0, 0]))
    return {"fx": float(p3[0, 0]), "baseline_m": float(baseline)}


def model_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dims=[128, 128, 128], corr_implementation="alt", shared_backbone=False,
        corr_levels=4, corr_radius=4, n_downsample=2, context_norm="batch",
        slow_fast_gru=False, n_gru_layers=3, mixed_precision=True,
    )


def load_image(path: Path, mirror: bool = False) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    if mirror:
        tensor = torch.flip(tensor, dims=[3])
    return tensor.cuda()


def signed_disparity(raw_disparity: np.ndarray) -> tuple[np.ndarray, int]:
    """Choose the sign that produces the most valid RAFT-Stereo disparity.

    RAFT checkpoints may use opposite disparity sign conventions when the reference
    and matching images are swapped.  This data-driven choice is recorded in
    the provenance file and avoids silently discarding a reversed pair.
    """
    if raw_disparity.ndim != 2:
        raise ValueError(f"expected a single HxW RAFT-Stereo disparity field, got {raw_disparity.shape}")
    positive = np.isfinite(raw_disparity) & (raw_disparity > 0)
    negative = np.isfinite(raw_disparity) & (raw_disparity < 0)
    sign = 1 if positive.sum() >= negative.sum() else -1
    return (sign * raw_disparity).astype(np.float32, copy=False), sign


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this in the existing RAFT-Stereo GPU environment.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    reference_paths = sorted((args.sequence_dir / "image_3").glob("*.png"))[: args.num_frames]
    matching_paths = sorted((args.sequence_dir / "image_2").glob("*.png"))[: args.num_frames]
    if len(reference_paths) != args.num_frames or len(matching_paths) != args.num_frames:
        raise RuntimeError("Expected the requested number of image_3/image_2 stereo pairs")
    if [path.stem for path in reference_paths] != [path.stem for path in matching_paths]:
        raise RuntimeError("image_3/image_2 frame IDs do not match")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration = parse_calibration(args.sequence_dir / "calib.txt")
    scale = calibration["fx"] * calibration["baseline_m"]

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

    def infer(reference: Path, matching: Path) -> tuple[np.ndarray, float]:
        # Mirroring turns a right-reference correspondence into the same
        # left-to-right geometry used by the checkpoint during inference.
        reference_image, matching_image = load_image(reference, mirror=True), load_image(matching, mirror=True)
        padder = InputPadder(reference_image.shape, divis_by=32)
        reference_image, matching_image = padder.pad(reference_image, matching_image)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            _, flow_up = model(reference_image, matching_image, iters=args.valid_iters, test_mode=True)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        disparity_field = padder.unpad(flow_up).squeeze().float().cpu().numpy()
        disparity_field = np.ascontiguousarray(disparity_field[:, ::-1])
        return disparity_field, elapsed_ms

    print("Running right-to-left RAFT warm-up...", flush=True)
    infer(reference_paths[0], matching_paths[0])
    rows: list[dict[str, float | int | str]] = []
    expected_shape = np.asarray(Image.open(reference_paths[0])).shape[:2]
    observed_signs: list[int] = []

    for index, (reference, matching) in enumerate(zip(reference_paths, matching_paths)):
        raw_disparity, inference_ms = infer(reference, matching)
        if raw_disparity.shape != expected_shape:
            raise RuntimeError(
                f"Frame {reference.stem}: expected {expected_shape}, got {raw_disparity.shape}"
            )
        disparity, sign = signed_disparity(raw_disparity)
        observed_signs.append(sign)
        depth = np.zeros(expected_shape, dtype=np.float32)
        valid_disp = np.isfinite(disparity) & (disparity > 0)
        depth[valid_disp] = scale / disparity[valid_disp]
        valid = valid_disp & np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
        depth[~valid] = 0.0
        np.save(args.output_dir / f"{reference.stem}_depth.npy", depth, allow_pickle=False)
        rows.append({"index": index, "frame_id": reference.stem, "inference_ms": inference_ms,
                     "flow_sign_to_positive_disparity": sign, "valid_ratio": float(valid.mean()),
                     "depth_median_m": float(np.median(depth[valid])) if valid.any() else float("nan")})
        if (index + 1) % 10 == 0 or index == 0:
            print(f"[{index + 1:03d}/{args.num_frames}] {reference.stem} {inference_ms:.1f} ms valid={valid.mean():.3f}", flush=True)

    with (args.output_dir / "depth_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        repo_commit = subprocess.check_output(["git", "-C", str(args.raft_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        repo_commit = "unknown"
    summary = {"purpose": "9R right-to-left depth for pure-stereo confidence audit", "reference_view": "image_3",
               "matching_view": "image_2", "mirrored_inputs_before_inference": True,
               "frames": len(rows), "shape": list(expected_shape),
               "output_format": "float32 metres, invalid=0.0", "calibration": calibration,
               "valid_iters": args.valid_iters, "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": sha256_file(args.checkpoint), "raft_repo_commit": repo_commit,
               "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "flow_sign_counts": {"positive": observed_signs.count(1), "negative": observed_signs.count(-1)}}
    with (args.output_dir / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
