"""Fuse all 200 seen Mask2Former frames into the deployment Gaussian map.

The CPU-only dry-run path intentionally avoids importing torch, OpenCV, the
CUDA rasterizer, or SplaTAM.  The GPU path preserves the 11R fusion algorithm
while adding source locks, smoke limiting, progress memory logs, and richer
provenance.
"""

import argparse
import csv
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


DYNAMIC_IDS = {11, 12, 13, 14, 15, 16, 17, 18}
SKY_ID = 10
EXPECTED_GAUSSIANS = -1  # caller must lock the actual deployment map count
EXPECTED_HELD_OUT = []
EXPECTED_PARAMS_SHA256 = ""  # caller must pass --expected-params-sha256

# Populated only after the CPU-only checks have passed.
cv2 = None
torch = None
F = None
Renderer = None
setup_camera = None
transform_to_frame = None
transformed_params2depthplussilhouette = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse Cityscapes-19 semantics into KITTI 09 frames 0300-0499"
    )
    parser.add_argument("--params", required=True)
    parser.add_argument("--semantic-dir", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splatam-root")
    parser.add_argument("--semantic-confidence-threshold", type=float, default=0.50)
    parser.add_argument("--silhouette-threshold", type=float, default=0.05)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.08)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--representative-frames",
        default="0,49,99,149,199",
    )
    parser.add_argument(
        "--max-train-frames",
        type=int,
        help="Process only the first N non-held-out frames. Use only for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--deep-check",
        action="store_true",
        help="Read and validate all 200 semantic label/confidence arrays.",
    )
    parser.add_argument("--dry-run-report")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--expected-gaussians", type=int, default=EXPECTED_GAUSSIANS)
    parser.add_argument("--expected-params-sha256", default=EXPECTED_PARAMS_SHA256)
    parser.add_argument(
        "--expected-held-out",
        default=",".join(str(x) for x in EXPECTED_HELD_OUT),
    )
    parser.add_argument("--expected-height", type=int, default=376)
    parser.add_argument("--expected-width", type=int, default=1241)
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_frame_list(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def png_shape(path):
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return height, width


def inspect_inputs(args):
    started = time.time()
    params_path = Path(args.params).resolve()
    semantic_dir = Path(args.semantic_dir).resolve()
    rgb_dir = Path(args.rgb_dir).resolve()
    expected_held_out = parse_frame_list(args.expected_held_out)
    issues = []

    for path, label in [
        (params_path, "params"),
        (semantic_dir / "id2label.json", "id2label"),
        (semantic_dir / "palette.json", "palette"),
    ]:
        if not path.is_file():
            issues.append(f"missing {label}: {path}")

    if issues:
        report = {"status": "FAIL", "issues": issues}
        write_dry_run_report(args, report)
        return report

    params_sha256 = sha256_file(params_path)
    if args.expected_params_sha256 and params_sha256.lower() != args.expected_params_sha256.lower():
        issues.append(
            f"params sha256 mismatch: expected {args.expected_params_sha256}, got {params_sha256}"
        )

    required_keys = {
        "means3D",
        "rgb_colors",
        "unnorm_rotations",
        "logit_opacities",
        "log_scales",
        "cam_unnorm_rots",
        "cam_trans",
        "intrinsics",
        "w2c",
        "org_width",
        "org_height",
        "timestep",
    }
    with np.load(params_path, allow_pickle=True) as raw:
        missing_keys = sorted(required_keys.difference(raw.files))
        if missing_keys:
            issues.append(f"params missing keys: {missing_keys}")
        if missing_keys:
            num_gaussians = -1
            num_frames = -1
            width = -1
            height = -1
            held_out = []
            param_shapes = {}
        else:
            num_gaussians = int(raw["means3D"].shape[0])
            num_frames = int(raw["cam_trans"].shape[-1])
            width = int(raw["org_width"])
            height = int(raw["org_height"])
            held_out = (
                sorted(int(x) for x in raw["eval_only_frames"].tolist())
                if "eval_only_frames" in raw.files
                else []
            )
            param_shapes = {
                name: list(raw[name].shape)
                for name in [
                    "means3D",
                    "rgb_colors",
                    "unnorm_rotations",
                    "logit_opacities",
                    "log_scales",
                    "cam_unnorm_rots",
                    "cam_trans",
                    "timestep",
                ]
            }

    if num_gaussians != args.expected_gaussians:
        issues.append(
            f"gaussian count mismatch: expected {args.expected_gaussians}, got {num_gaussians}"
        )
    if num_frames != 1000:
        issues.append(f"frame count mismatch: expected 1000, got {num_frames}")
    if held_out != expected_held_out:
        issues.append(f"held-out mismatch: expected {expected_held_out}, got {held_out}")
    if (height, width) != (args.expected_height, args.expected_width):
        issues.append(
            f"image size mismatch: expected {(args.expected_height, args.expected_width)}, "
            f"got {(height, width)}"
        )

    with open(semantic_dir / "id2label.json", "r", encoding="utf-8") as handle:
        id2label = {int(key): value for key, value in json.load(handle).items()}
    with open(semantic_dir / "palette.json", "r", encoding="utf-8") as handle:
        palette = json.load(handle)
    if sorted(id2label) != list(range(19)):
        issues.append(f"id2label must contain IDs 0..18, got {sorted(id2label)}")
    if sorted(int(key) for key in palette) != list(range(19)):
        issues.append("palette must contain IDs 0..18")

    label_files = []
    confidence_files = []
    rgb_files = []
    for frame in range(max(num_frames, 0)):
        label_path = semantic_dir / "labels" / f"{frame:06d}_semantic.npy"
        confidence_path = semantic_dir / "confidence" / f"{frame:06d}_confidence.npy"
        rgb_path = rgb_dir / f"{frame:06d}.png"
        label_files.append(label_path)
        confidence_files.append(confidence_path)
        rgb_files.append(rgb_path)
        if not label_path.is_file():
            issues.append(f"missing semantic label for frame {frame}: {label_path}")
        if not confidence_path.is_file():
            issues.append(f"missing semantic confidence for frame {frame}: {confidence_path}")
        if not rgb_path.is_file():
            issues.append(f"missing RGB for frame {frame}: {rgb_path}")

    check_frames = list(range(num_frames)) if args.deep_check else [0, 19, 79, 99, 199]
    checked_arrays = []
    for frame in check_frames:
        if frame >= num_frames:
            continue
        label_path = label_files[frame]
        confidence_path = confidence_files[frame]
        rgb_path = rgb_files[frame]
        if not (label_path.is_file() and confidence_path.is_file() and rgb_path.is_file()):
            continue
        labels = np.load(label_path, allow_pickle=False)
        confidence = np.load(confidence_path, allow_pickle=False)
        if labels.shape != (height, width):
            issues.append(f"frame {frame} label shape {labels.shape} != {(height, width)}")
        if confidence.shape != (height, width):
            issues.append(
                f"frame {frame} confidence shape {confidence.shape} != {(height, width)}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            issues.append(f"frame {frame} labels dtype is not integer: {labels.dtype}")
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) > 18):
            issues.append(
                f"frame {frame} labels outside 0..18: {int(labels.min())}..{int(labels.max())}"
            )
        finite = np.isfinite(confidence)
        if not bool(finite.all()):
            issues.append(f"frame {frame} confidence contains non-finite values")
        if confidence.size and (float(confidence.min()) < 0 or float(confidence.max()) > 1):
            issues.append(
                f"frame {frame} confidence outside 0..1: {float(confidence.min())}..{float(confidence.max())}"
            )
        try:
            current_png_shape = png_shape(rgb_path)
            if current_png_shape != (height, width):
                issues.append(
                    f"frame {frame} RGB shape {current_png_shape} != {(height, width)}"
                )
        except ValueError as error:
            issues.append(str(error))
        checked_arrays.append(
            {
                "frame": frame,
                "label_dtype": str(labels.dtype),
                "confidence_dtype": str(confidence.dtype),
                "label_range": [int(labels.min()), int(labels.max())],
                "confidence_range": [float(confidence.min()), float(confidence.max())],
            }
        )

    train_frames = [frame for frame in range(max(num_frames, 0)) if frame not in set(held_out)]
    selected_train_frames = train_frames
    if args.max_train_frames is not None:
        if args.max_train_frames <= 0:
            issues.append("--max-train-frames must be positive")
        else:
            selected_train_frames = train_frames[: args.max_train_frames]

    report = {
        "status": "PASS" if not issues else "FAIL",
        "mode": "DRY_RUN_DEEP" if args.deep_check else "DRY_RUN",
        "issues": issues,
        "params": str(params_path),
        "params_bytes": params_path.stat().st_size,
        "params_sha256": params_sha256,
        "expected_params_sha256": args.expected_params_sha256,
        "num_gaussians": num_gaussians,
        "num_frames": num_frames,
        "image_size_hw": [height, width],
        "held_out_frames": held_out,
        "full_training_frames": len(train_frames),
        "selected_training_frames": len(selected_train_frames),
        "selected_training_frame_ids": selected_train_frames,
        "semantic_dir": str(semantic_dir),
        "rgb_dir": str(rgb_dir),
        "semantic_label_files": sum(path.is_file() for path in label_files),
        "semantic_confidence_files": sum(path.is_file() for path in confidence_files),
        "rgb_files_required_and_found": sum(path.is_file() for path in rgb_files),
        "id2label": id2label,
        "checked_array_count": len(checked_arrays),
        "checked_arrays": checked_arrays,
        "param_shapes": param_shapes,
        "estimated_evidence_tensor_gib": (
            num_gaussians * max(len(id2label), 0) * 4 / (1024**3)
            if num_gaussians > 0
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_dry_run_report(args, report)
    return report


def write_dry_run_report(args, report):
    if not args.dry_run_report:
        return
    path = Path(args.dry_run_report).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def load_gpu_runtime(splatam_root):
    global cv2, torch, F, Renderer, setup_camera
    global transform_to_frame, transformed_params2depthplussilhouette

    if not splatam_root:
        raise RuntimeError("GPU mode requires --splatam-root")
    root = Path(splatam_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SplaTAM root does not exist: {root}")
    sys.path.insert(0, str(root))

    import cv2 as cv2_module
    import torch as torch_module
    import torch.nn.functional as functional
    from diff_gaussian_rasterization import GaussianRasterizer
    from utils.recon_helpers import setup_camera as setup_camera_function
    from utils.slam_helpers import (
        transform_to_frame as transform_to_frame_function,
        transformed_params2depthplussilhouette as depth_silhouette_function,
    )

    cv2 = cv2_module
    torch = torch_module
    F = functional
    Renderer = GaussianRasterizer
    setup_camera = setup_camera_function
    transform_to_frame = transform_to_frame_function
    transformed_params2depthplussilhouette = depth_silhouette_function


def to_gpu_params(npz):
    names = [
        "means3D",
        "rgb_colors",
        "unnorm_rotations",
        "logit_opacities",
        "log_scales",
        "cam_unnorm_rots",
        "cam_trans",
    ]
    return {
        name: torch.from_numpy(npz[name]).cuda().float().contiguous()
        for name in names
    }


def semantic_rendervar(params, transformed, colors, semantic_valid):
    log_scales = params["log_scales"]
    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    return {
        "means3D": transformed["means3D"],
        "colors_precomp": colors,
        "rotations": F.normalize(transformed["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]) * semantic_valid[:, None],
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(
            params["means3D"], requires_grad=True, device="cuda"
        ),
    }


def ensure_output_directory(path, allow_existing):
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"refusing to write into non-empty output directory: {path}; "
            "use a new directory or pass --allow-existing-output explicitly"
        )
    path.mkdir(parents=True, exist_ok=True)


def run_gpu_fusion(args, inspection):
    load_gpu_runtime(args.splatam_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_directory(output_dir, args.allow_existing_output)
    render_dir = output_dir / "semantic_rendered_views"
    render_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.params, allow_pickle=True)
    params = to_gpu_params(raw)
    intrinsics = torch.from_numpy(raw["intrinsics"]).cuda().float()
    first_w2c = raw["w2c"].astype(np.float32)
    width, height = int(raw["org_width"]), int(raw["org_height"])
    num_frames = int(params["cam_trans"].shape[-1])
    held_out = (
        sorted(int(x) for x in raw["eval_only_frames"].tolist())
        if "eval_only_frames" in raw.files
        else []
    )
    full_train_frames = [frame for frame in range(num_frames) if frame not in set(held_out)]
    train_frames = full_train_frames
    if args.max_train_frames is not None:
        train_frames = full_train_frames[: args.max_train_frames]
    timestep = torch.from_numpy(raw["timestep"]).cuda().float()
    opacity = torch.sigmoid(params["logit_opacities"][:, 0])
    num_gaussians = int(params["means3D"].shape[0])

    semantic_dir = Path(args.semantic_dir)
    with open(semantic_dir / "id2label.json", "r", encoding="utf-8") as handle:
        id2label = {int(key): value for key, value in json.load(handle).items()}
    with open(semantic_dir / "palette.json", "r", encoding="utf-8") as handle:
        palette_json = json.load(handle)
    num_classes = len(id2label)
    palette_np = (
        np.array(
            [palette_json[str(index)]["rgb"] for index in range(num_classes)],
            dtype=np.float32,
        )
        / 255.0
    )
    palette = torch.from_numpy(palette_np).cuda()

    camera = setup_camera(width, height, raw["intrinsics"], first_w2c)
    evidence = torch.zeros(
        (num_gaussians, num_classes), device="cuda", dtype=torch.float32
    )
    observations = torch.zeros(num_gaussians, device="cuda", dtype=torch.int32)

    processed = 0
    accepted_observations = 0
    with torch.no_grad():
        for frame in train_frames:
            from resource_guard import wait_for_headroom
            wait_for_headroom("semantic_fusion", frame)
            label_path = semantic_dir / "labels" / f"{frame:06d}_semantic.npy"
            confidence_path = (
                semantic_dir / "confidence" / f"{frame:06d}_confidence.npy"
            )
            labels = torch.from_numpy(
                np.load(label_path, allow_pickle=False).astype(np.int64)
            ).cuda()
            confidence = torch.from_numpy(
                np.load(confidence_path, allow_pickle=False).astype(np.float32)
            ).cuda()

            transformed = transform_to_frame(
                params, frame, gaussians_grad=False, camera_grad=False
            )
            depth_render_variables = transformed_params2depthplussilhouette(
                params, torch.from_numpy(first_w2c).cuda(), transformed
            )
            depth_silhouette, radii, _ = Renderer(raster_settings=camera)(
                **depth_render_variables
            )
            rendered_depth = depth_silhouette[0]
            silhouette = depth_silhouette[1]

            xyz = transformed["means3D"]
            z = xyz[:, 2]
            uv_h = (intrinsics @ xyz.T).T
            u = torch.round(uv_h[:, 0] / torch.clamp(uv_h[:, 2], min=1e-6)).long()
            v = torch.round(uv_h[:, 1] / torch.clamp(uv_h[:, 2], min=1e-6)).long()
            valid = (
                (z > 0.05)  # fixed final map offline fusion
                & (radii > 0)
                & (u >= 0)
                & (u < width)
                & (v >= 0)
                & (v < height)
            )
            indices = torch.nonzero(valid, as_tuple=False)[:, 0]
            if indices.numel() > 0:
                uu, vv = u[indices], v[indices]
                semantic_labels = labels[vv, uu]
                semantic_confidence = confidence[vv, uu]
                map_depth = rendered_depth[vv, uu]
                map_silhouette = silhouette[vv, uu]
                tolerance = args.depth_absolute_tolerance + (
                    args.depth_relative_tolerance * torch.clamp(map_depth, min=0)
                )
                depth_delta = torch.abs(z[indices] - map_depth)
                keep = (
                    (semantic_labels != SKY_ID)
                    & (semantic_confidence >= args.semantic_confidence_threshold)
                    & (map_depth > 0)
                    & (map_silhouette >= args.silhouette_threshold)
                    & (depth_delta <= tolerance)
                )
                indices = indices[keep]
                semantic_labels = semantic_labels[keep]
                semantic_confidence = semantic_confidence[keep]
                depth_delta = depth_delta[keep]
                tolerance = tolerance[keep]
                if indices.numel():
                    visibility = opacity[indices] * torch.exp(
                        -depth_delta / torch.clamp(tolerance, min=1e-6)
                    )
                    weight = visibility * semantic_confidence
                    evidence.index_put_((indices, semantic_labels), weight, accumulate=True)
                    observations.index_add_(
                        0, indices, torch.ones_like(indices, dtype=torch.int32)
                    )
                    accepted_observations += int(indices.numel())

            processed += 1
            if processed == 1 or processed % 20 == 0 or processed == len(train_frames):
                peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"fused {processed}/{len(train_frames)} training frames; "
                    f"accepted={accepted_observations}; peak_cuda_gib={peak_gib:.3f}",
                    flush=True,
                )

    total_evidence = evidence.sum(dim=1)
    max_evidence, labels = evidence.max(dim=1)
    assigned = total_evidence > 0
    fused_confidence = torch.zeros_like(total_evidence)
    fused_confidence[assigned] = max_evidence[assigned] / total_evidence[assigned]
    labels_out = labels.to(torch.int16)
    labels_out[~assigned] = -1
    dynamic = torch.zeros(num_gaussians, dtype=torch.bool, device="cuda")
    for class_id in DYNAMIC_IDS:
        dynamic |= labels_out == class_id

    labels_cpu = labels_out.cpu().numpy()
    confidence_cpu = fused_confidence.cpu().numpy().astype(np.float16)
    evidence_cpu = total_evidence.cpu().numpy().astype(np.float32)
    observations_cpu = observations.cpu().numpy().astype(np.uint16)
    dynamic_cpu = dynamic.cpu().numpy()
    np.savez_compressed(
        output_dir / "semantic_gaussian_labels.npz",
        label=labels_cpu,
        confidence=confidence_cpu,
        evidence_sum=evidence_cpu,
        observation_count=observations_cpu,
        potentially_dynamic=dynamic_cpu,
        class_names=np.array([id2label[index] for index in range(num_classes)]),
        source_gaussian_index=np.arange(num_gaussians, dtype=np.int32),
        held_out_frames=np.array(held_out, dtype=np.int16),
        fused_training_frames=np.array(train_frames, dtype=np.int16),
    )

    rows = []
    assigned_count = int(assigned.sum().item())
    for class_id in range(num_classes):
        mask = labels_cpu == class_id
        count = int(mask.sum())
        rows.append(
            {
                "class_id": class_id,
                "class_name": id2label[class_id],
                "potentially_dynamic": class_id in DYNAMIC_IDS,
                "gaussian_count": count,
                "fraction_all_gaussians": count / num_gaussians,
                "fraction_assigned_gaussians": (
                    count / assigned_count if assigned_count else 0.0
                ),
                "mean_fused_confidence": (
                    float(confidence_cpu[mask].mean()) if count else 0.0
                ),
                "evidence_sum": float(evidence_cpu[mask].sum()) if count else 0.0,
            }
        )
    with open(
        output_dir / "semantic_class_histogram.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    representative = parse_frame_list(args.representative_frames)
    invalid_representative = [
        frame for frame in representative if frame < 0 or frame >= num_frames
    ]
    if invalid_representative:
        raise ValueError(f"representative frames outside range: {invalid_representative}")
    semantic_colors = torch.zeros((num_gaussians, 3), device="cuda")
    if assigned.any():
        semantic_colors[assigned] = palette[labels[assigned].long()]
    semantic_valid = assigned.float()
    rgb_dir = Path(args.rgb_dir)
    with torch.no_grad():
        for frame in representative:
            transformed = transform_to_frame(
                params, frame, gaussians_grad=False, camera_grad=False
            )
            render_variables = semantic_rendervar(
                params, transformed, semantic_colors, semantic_valid
            )
            rendered, _, _ = Renderer(raster_settings=camera)(**render_variables)
            semantic_rgb = (
                torch.clamp(rendered, 0, 1).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            cv2.imwrite(
                str(render_dir / f"semantic_{frame:06d}.png"),
                cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR),
            )
            source_bgr = cv2.imread(
                str(rgb_dir / f"{frame:06d}.png"), cv2.IMREAD_COLOR
            )
            if source_bgr is not None:
                semantic_bgr = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(source_bgr, 0.45, semantic_bgr, 0.55, 0)
                cv2.imwrite(str(render_dir / f"overlay_{frame:06d}.png"), overlay)

    runtime = time.time() - started
    mode = "FORMAL" if len(train_frames) == len(full_train_frames) else "SMOKE"
    summary = {
        "status": "COMPLETE",
        "step": "12R",
        "mode": mode,
        "params": str(Path(args.params).resolve()),
        "params_sha256": inspection["params_sha256"],
        "fusion_script": str(Path(__file__).resolve()),
        "fusion_script_sha256": sha256_file(Path(__file__).resolve()),
        "semantic_source": str(semantic_dir.resolve()),
        "num_gaussians": num_gaussians,
        "assigned_gaussians": assigned_count,
        "assigned_fraction": assigned_count / num_gaussians,
        "unassigned_gaussians": num_gaussians - assigned_count,
        "potentially_dynamic_gaussians": int(dynamic.sum().item()),
        "sky_written": int((labels_cpu == SKY_ID).sum()),
        "full_training_frame_count": len(full_train_frames),
        "processed_training_frame_count": len(train_frames),
        "processed_training_frames": train_frames,
        "held_out_frames": held_out,
        "accepted_gaussian_frame_observations": accepted_observations,
        "semantic_confidence_threshold": args.semantic_confidence_threshold,
        "visibility": {
            "definition": "positive rasterizer radius and projected-centre depth agreement with rendered Gaussian depth",
            "weight": "sigmoid(opacity_logit) * exp(-abs(center_depth-rendered_depth)/tolerance) * Mask2Former_confidence",
            "silhouette_threshold": args.silhouette_threshold,
            "depth_tolerance_m": (
                f"{args.depth_absolute_tolerance} + "
                f"{args.depth_relative_tolerance} * rendered_depth"
            ),
        },
        "class_semantics": (
            "Cityscapes-19; IDs 11-18 mean potentially dynamic class, not measured motion"
        ),
        "representative_frames": representative,
        "runtime_seconds": runtime,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    with open(
        output_dir / "semantic_fusion_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "source_map.sha256", "w", encoding="utf-8") as handle:
        handle.write(f"{inspection['params_sha256']}  {Path(args.params).resolve()}\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "status",
                    "mode",
                    "num_gaussians",
                    "assigned_gaussians",
                    "assigned_fraction",
                    "runtime_seconds",
                    "peak_cuda_memory_gib",
                ]
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    if args.expected_gaussians <= 0 or not args.expected_params_sha256:
        raise ValueError("deployment semantic fusion requires the exact map count and SHA256")
    inspection = inspect_inputs(args)
    print(
        json.dumps(
            {
                key: inspection.get(key)
                for key in [
                    "status",
                    "mode",
                    "params_sha256",
                    "num_gaussians",
                    "num_frames",
                    "full_training_frames",
                    "selected_training_frames",
                    "checked_array_count",
                    "elapsed_seconds",
                    "issues",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if inspection["status"] != "PASS":
        raise SystemExit(2)
    if args.dry_run:
        return
    run_gpu_fusion(args, inspection)


if __name__ == "__main__":
    main()
