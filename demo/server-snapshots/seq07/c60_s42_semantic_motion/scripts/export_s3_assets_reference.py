from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


C0 = 0.28209479177387814
EXPECTED_CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
CITYSCAPES_COLORS = np.asarray(
    [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32),
    ],
    dtype=np.float32,
) / 255.0
UNKNOWN_COLOR = np.asarray((110, 110, 110), dtype=np.float32) / 255.0

PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the KITTI 08 S3 Gaussian map for SuperSplat.")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    # Match SplaTAM's official export_ply.py exactly. Values are deliberately
    # not clipped here; the renderer applies display-domain clamping later.
    return (rgb - 0.5) / C0


def prepare_records(
    means: np.ndarray,
    colors: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
    logit_opacities: np.ndarray,
) -> np.ndarray:
    count = len(means)
    if not (
        means.shape == (count, 3)
        and colors.shape == (count, 3)
        and rotations.shape == (count, 4)
        and logit_opacities.shape == (count, 1)
        and log_scales.shape in ((count, 1), (count, 3))
    ):
        raise ValueError("Gaussian attribute shapes do not match the Graphdeco PLY schema")
    scales = np.repeat(log_scales, 3, axis=1) if log_scales.shape[1] == 1 else log_scales
    if not all(np.isfinite(value).all() for value in (means, colors, scales, rotations, logit_opacities)):
        raise ValueError("non-finite Gaussian attribute detected")
    sh = rgb_to_sh(colors)
    records = np.empty(count, dtype=PLY_DTYPE)
    records["x"], records["y"], records["z"] = means.T
    records["nx"] = records["ny"] = records["nz"] = 0.0
    records["f_dc_0"], records["f_dc_1"], records["f_dc_2"] = sh.T
    records["opacity"] = logit_opacities[:, 0]
    records["scale_0"], records["scale_1"], records["scale_2"] = scales.T
    records["rot_0"], records["rot_1"], records["rot_2"], records["rot_3"] = rotations.T
    return records


def write_ply(path: Path, records: np.ndarray) -> None:
    properties = "\n".join(f"property float {name}" for name in records.dtype.names)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment KITTI 21 frames 1200-1399, exported from SplaTAM for S3 display\n"
        f"element vertex {len(records)}\n"
        f"{properties}\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_poses(params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    quaternions = np.asarray(params["cam_unnorm_rots"])[0].T
    translations = np.asarray(params["cam_trans"])[0].T
    centers = []
    c2w_rotations = []
    for quaternion, translation in zip(quaternions, translations):
        rotation_w2c = quaternion_wxyz_to_rotation(quaternion)
        c2w_rotation = rotation_w2c.T
        centers.append(-c2w_rotation @ translation)
        c2w_rotations.append(c2w_rotation)
    return np.asarray(centers, dtype=np.float32), np.asarray(c2w_rotations, dtype=np.float32)


def trajectory_records(centers: np.ndarray, keyframes: np.ndarray) -> np.ndarray:
    samples = []
    colors = []
    for index in range(len(centers) - 1):
        for alpha in np.linspace(0.0, 1.0, 8, endpoint=False):
            samples.append((1.0 - alpha) * centers[index] + alpha * centers[index + 1])
            progress = (index + alpha) / (len(centers) - 1)
            colors.append((1.0, 0.85 * (1.0 - progress), 0.05 + 0.75 * progress))
    samples.append(centers[-1])
    colors.append((1.0, 0.0, 0.8))
    samples = np.asarray(samples, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    scales = np.full((len(samples), 1), math.log(0.12), dtype=np.float32)
    rotations = np.zeros((len(samples), 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((len(samples), 1), 5.0, dtype=np.float32)
    path_records = prepare_records(samples, colors, scales, rotations, opacities)

    keyframe_centers = centers[keyframes]
    keyframe_colors = np.tile(np.asarray((0.0, 1.0, 1.0), dtype=np.float32), (len(keyframes), 1))
    keyframe_scales = np.full((len(keyframes), 1), math.log(0.24), dtype=np.float32)
    keyframe_rotations = np.zeros((len(keyframes), 4), dtype=np.float32)
    keyframe_rotations[:, 0] = 1.0
    keyframe_opacities = np.full((len(keyframes), 1), 6.0, dtype=np.float32)
    keyframe_records = prepare_records(
        keyframe_centers, keyframe_colors, keyframe_scales, keyframe_rotations, keyframe_opacities
    )
    return np.concatenate((path_records, keyframe_records))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def viewer_settings(means: np.ndarray) -> dict[str, object]:
    low, high = np.quantile(means, [0.01, 0.99], axis=0)
    target = ((low + high) / 2.0).astype(float)
    extent = high - low
    radius = float(max(extent[0], extent[2]) * 0.72)
    elevation = float(max(extent[1] * 2.5, 24.0))
    # Begin at the original KITTI camera so the road is immediately
    # recognizable, then pull out into oblique, side and top-down views.
    original_target = [0.0, 0.0, min(float(high[2]), 55.0)]
    positions = [
        [0.0, 0.0, 0.0],
        [target[0], target[1] - elevation, target[2] - radius],
        [target[0] - radius, target[1] - elevation * 0.8, target[2]],
        [target[0], target[1] - elevation * 2.2, target[2]],
        [target[0] + radius, target[1] - elevation * 0.8, target[2]],
        [0.0, 0.0, 0.0],
    ]
    targets = [original_target] + [target.tolist()] * 4 + [original_target]
    return {
        "version": 2,
        "tonemapping": "linear",
        "highPrecisionRendering": False,
        "background": {"color": [0.055, 0.065, 0.08]},
        "postEffectSettings": {
            "sharpness": {"enabled": True, "amount": 0.25},
            "bloom": {"enabled": False, "intensity": 0.1, "blurLevel": 2},
            "grading": {
                "enabled": False, "brightness": 1, "contrast": 1,
                "saturation": 1, "tint": [1, 1, 1],
            },
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [
            {
                "name": "S3 orbit",
                "duration": 24,
                "frameRate": 30,
                "loopMode": "repeat",
                "interpolation": "spline",
                "smoothness": 0.8,
                "keyframes": {
                    "times": [0, 4, 9, 14, 19, 24],
                    "values": {
                        "position": [value for position in positions for value in position],
                        "target": [value for item in targets for value in item],
                        "fov": [62, 58, 58, 58, 58, 62],
                    },
                },
            }
        ],
        "cameras": [{"initial": {"position": positions[0], "target": original_target, "fov": 62}}],
        "annotations": [],
        "startMode": "animTrack",
    }


def trajectory_viewer_settings(means: np.ndarray) -> dict[str, object]:
    settings = viewer_settings(means)
    low, high = np.quantile(means, [0.01, 0.99], axis=0)
    target = ((low + high) / 2.0).astype(float)
    extent = high - low
    radius = float(max(extent[0], extent[2]) * 0.72)
    elevation = float(max(extent[1] * 2.5, 24.0))
    positions = [
        [target[0], target[1] - elevation * 2.2, target[2]],
        [target[0] - radius, target[1] - elevation, target[2]],
        [target[0], target[1] - elevation, target[2] - radius],
        [target[0] + radius, target[1] - elevation, target[2]],
        [target[0], target[1] - elevation * 2.2, target[2]],
    ]
    targets = [target.tolist()] * len(positions)
    track = settings["animTracks"][0]
    track["name"] = "S3 trajectory overview"
    track["duration"] = 20
    track["keyframes"] = {
        "times": [0, 5, 10, 15, 20],
        "values": {
            "position": [value for position in positions for value in position],
            "target": [value for item in targets for value in item],
            "fov": [58, 58, 58, 58, 58],
        },
    }
    settings["cameras"] = [{"initial": {"position": positions[0], "target": target.tolist(), "fov": 58}}]
    return settings


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.params, allow_pickle=True) as archive:
        params = {key: archive[key] for key in archive.files}
    with np.load(args.semantic, allow_pickle=True) as archive:
        semantic = {key: archive[key] for key in archive.files}

    means = np.asarray(params["means3D"], dtype=np.float32)
    rgb = np.asarray(params["rgb_colors"], dtype=np.float32)
    rotations = np.asarray(params["unnorm_rotations"], dtype=np.float32)
    log_scales = np.asarray(params["log_scales"], dtype=np.float32)
    opacities = np.asarray(params["logit_opacities"], dtype=np.float32)
    labels = np.asarray(semantic["label"], dtype=np.int32)
    source_indices = np.asarray(semantic["source_gaussian_index"], dtype=np.int64)
    class_names = tuple(str(value) for value in semantic["class_names"].tolist())
    count = len(means)
    if class_names != EXPECTED_CLASS_NAMES:
        raise ValueError(f"unexpected class names: {class_names}")
    if len(labels) != count or not np.array_equal(source_indices, np.arange(count)):
        raise ValueError("semantic labels are not one-to-one with the selected Gaussian map")
    if labels.min() < -1 or labels.max() >= len(class_names):
        raise ValueError("semantic label outside the expected range")

    semantic_rgb = np.tile(UNKNOWN_COLOR, (count, 1))
    known = labels >= 0
    semantic_rgb[known] = CITYSCAPES_COLORS[labels[known]]
    rgb_records = prepare_records(means, rgb, log_scales, rotations, opacities)
    semantic_records = prepare_records(means, semantic_rgb, log_scales, rotations, opacities)
    centers, c2w_rotations = camera_poses(params)
    keyframes = np.asarray(params["keyframe_time_indices"], dtype=np.int64)
    path_records = trajectory_records(centers, keyframes)

    outputs = {
        "rgb": args.output_dir / "KITTI08_RGB.ply",
        "semantic": args.output_dir / "KITTI08_Semantic.ply",
        "trajectory": args.output_dir / "KITTI08_CameraTrajectory_200frames.ply",
        "rgb_trajectory": args.output_dir / "KITTI08_RGB_with_Trajectory.ply",
    }
    write_ply(outputs["rgb"], rgb_records)
    write_ply(outputs["semantic"], semantic_records)
    write_ply(outputs["trajectory"], path_records)
    write_ply(outputs["rgb_trajectory"], np.concatenate((rgb_records, path_records)))

    width, height = int(params["org_width"]), int(params["org_height"])
    fx, fy = float(params["intrinsics"][0, 0]), float(params["intrinsics"][1, 1])
    cameras = [
        {
            "id": index,
            "img_name": f"{index:06d}.png",
            "width": width,
            "height": height,
            "position": centers[index].astype(float).tolist(),
            "rotation": c2w_rotations[index].astype(float).tolist(),
            "fx": fx,
            "fy": fy,
        }
        for index in range(len(centers))
    ]
    (args.output_dir / "KITTI08_camera_poses_inria.json").write_text(
        json.dumps(cameras, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "viewer_settings.json").write_text(
        json.dumps(viewer_settings(means), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "viewer_settings_trajectory.json").write_text(
        json.dumps(trajectory_viewer_settings(means), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "semantic_legend.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label_id", "class_name", "color_rgb", "gaussian_count"))
        writer.writerow((-1, "unknown", "110,110,110", int(np.count_nonzero(labels == -1))))
        for label_id, (name, color) in enumerate(zip(class_names, CITYSCAPES_COLORS)):
            writer.writerow((label_id, name, ",".join(str(int(round(v * 255))) for v in color), int(np.count_nonzero(labels == label_id))))

    evidence_files = list(outputs.values()) + [
        args.output_dir / "KITTI08_camera_poses_inria.json",
        args.output_dir / "viewer_settings.json",
        args.output_dir / "viewer_settings_trajectory.json",
        args.output_dir / "semantic_legend.csv",
    ]
    hashes = {path.name: sha256(path) for path in evidence_files}
    manifest = {
        "status": "S3_KITTI08_EXPORT_COMPLETE",
        "dataset": "KITTI odometry sequence 08",
        "source_frame_range": [3000, 3199],
        "pose_protocol": "KITTI ground-truth camera poses (not ORB-SLAM3)",
        "depth_protocol": "RAFT-Stereo with LR, photometric, temporal and static-semantic confidence gating",
        "seed_confidence_threshold": 0.60,
        "source_params": str(args.params),
        "source_semantic": str(args.semantic),
        "gaussian_count": count,
        "semantic_count": len(labels),
        "source_gaussian_index_identity": True,
        "camera_pose_count": len(centers),
        "keyframe_count": len(keyframes),
        "rgb_range": [float(rgb.min()), float(rgb.max())],
        "map_quantiles_1_99": np.quantile(means, [0.01, 0.99], axis=0).astype(float).tolist(),
        "class_counts": {
            "unknown": int(np.count_nonzero(labels == -1)),
            **{name: int(np.count_nonzero(labels == index)) for index, name in enumerate(class_names)},
        },
        "outputs_sha256": hashes,
        "format_note": "Binary little-endian Graphdeco 3DGS PLY with log scales, logit opacity, wxyz rotation and SH-DC color.",
    }
    manifest_path = args.output_dir / "s3_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for filename, digest in sorted(hashes.items()):
            handle.write(f"{digest}  {filename}\n")
        handle.write(f"{sha256(manifest_path)}  {manifest_path.name}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
