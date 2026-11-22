"""Construct or relocate a registered S23 prediction bundle on a fresh host."""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from export_observations import export
from reproduction_common import STUDY, WINDOW_START, dump, read, sha256

PREPROCESSING = STUDY.parents[1] / "preprocessing"
STEREO = PREPROCESSING / "server-snapshots/causal-stereo"
STAGE2 = PREPROCESSING / "tsdd-s23"
FRAMES = 1000


def stage(name, script, arguments, bundle):
    command = [sys.executable, str(script), *map(str, arguments)]
    started = time.monotonic()
    with (bundle / f"{name}.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (bundle / "producer_timings.jsonl").open("a") as stream:
        import json

        stream.write(json.dumps({"step": name, "seconds": time.monotonic() - started}) + "\n")


def bundle_metadata(sequence, kitti, bundle, pose):
    source = kitti / "sequences" / sequence
    raw_poses = np.loadtxt(pose, dtype=np.float64)
    if raw_poses.shape != (FRAMES, 12) or not np.isfinite(raw_poses).all():
        raise ValueError(f"Invalid serialized causal ORB trajectory: {pose}")
    calib = source / "calib.txt"
    with calib.open() as stream:
        matrices = {row.split(":")[0]: np.fromstring(row.split(":", 1)[1], sep=" ")
                    for row in stream if row.startswith(("P2:", "P3:"))}
    K = matrices["P2"].reshape(3, 4)[:, :3]
    start = WINDOW_START[sequence]
    rows = []
    for frame in range(FRAMES):
        original = start + frame
        rgb = source / "image_2" / f"{original:06d}.png"
        right = source / "image_3" / f"{original:06d}.png"
        rows.append(dict(frame_id=frame, source_frame_id=original, rgb=str(rgb), right=str(right),
                         hashes=dict(rgb=sha256(rgb), right=sha256(right))))
    from PIL import Image

    with Image.open(rows[0]["rgb"]) as image:
        width, height = image.size
    manifest = dict(task_id="S23_FORMAL_PAST_ONLY_STEREO", pose_source="ORB_estimate",
                    pose_file=str(pose), pose_sha256=sha256(pose), frame_convention="P2",
                    frame_count=FRAMES, K=K.tolist(), H=height, W=width, depth_units="meters",
                    valid_depth_range_m=[.5, 80], confidence_threshold=.8,
                    GT_runtime=False, temporal_scope="current and previous only", rows=rows)
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(calib, bundle / "calib.txt")
    (bundle / "projection_beta.txt").write_text("380.34753\n", encoding="ascii")
    return manifest


def register_depths(manifest, cache):
    for row in manifest["rows"]:
        for key, folder, suffix in (("depth", "depth_confhard", "depth"),
                                    ("confidence", "confidence", "confidence")):
            path = cache / folder / f"{row['frame_id']:06d}_{suffix}.npy"
            row[key] = str(path)
            row["hashes"][key] = sha256(path)


def restore(args, bundle, manifest):
    metadata = STUDY / "input-metadata" / f"seq{args.sequence}"
    runtime_lock = args.runtime_lock or metadata / "input_manifest.json"
    original = read(runtime_lock)
    obs_file = args.observation_lock or metadata / "observations/manifest.json"
    observations = read(obs_file)
    register_depths(manifest, args.cache / "cache")
    if (original["pose_sha256"], original["K"], original["H"], original["W"]) != (
            manifest["pose_sha256"], manifest["K"], manifest["H"], manifest["W"]):
        raise ValueError("Restored input calibration or ORB identity differs from the lock")
    if len(original["rows"]) != FRAMES or any(
            (a["source_frame_id"], a["hashes"]) != (b["source_frame_id"], b["hashes"])
            for a, b in zip(original["rows"], manifest["rows"])):
        raise ValueError("Restored RGB, depth or confidence differs from the lock")
    if len(observations["rows"]) != FRAMES:
        raise ValueError("Observation lock must cover the full window")
    folder = bundle / "observations"
    folder.mkdir()
    for row in observations["rows"]:
        name = f"{row['frame_id']:06d}.npz"
        if row["path"] != name:
            raise ValueError(f"Unexpected observation name: {row['path']}")
        path = args.cache / "observations" / name
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Restored observation differs from the lock: {path}")
        (folder / name).symlink_to(path)
    dump(folder / "manifest.json", observations)
    timing = args.cache / "semantic_inference_timing.json"
    if timing.is_file():
        shutil.copyfile(timing, bundle / timing.name)


def create_view(bundle, manifest, kitti, sequence):
    view = bundle / "stereo_view"
    for directory in ("image_2", "image_3"):
        (view / directory).mkdir(parents=True)
    for row in manifest["rows"]:
        for key, directory in (("rgb", "image_2"), ("right", "image_3")):
            (view / directory / f"{row['frame_id']:06d}.png").symlink_to(row[key])
    shutil.copyfile(bundle / "calib.txt", view / "calib.txt")
    times = (kitti / "sequences" / sequence / "times.txt").read_text().splitlines()
    first = WINDOW_START[sequence]
    if len(times) < first + FRAMES:
        raise ValueError("Incomplete KITTI timestamps")
    (view / "times.txt").write_text("\n".join(times[first:first + FRAMES]) + "\n")
    return view


def load_causal_confidence():
    folder = PREPROCESSING / "server-snapshots/past-only-confidence"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("frozen_g19c", folder / "producer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def generate_depth_cache(bundle, manifest):
    import cv2

    g19c = load_causal_confidence()
    poses = np.tile(np.eye(4), (FRAMES, 1, 1))
    poses[:, :3] = np.loadtxt(manifest["pose_file"]).reshape(-1, 3, 4)
    K = np.asarray(manifest["K"])
    beta = float((bundle / "projection_beta.txt").read_text())
    calibration = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], baseline_m=beta / K[0, 0])
    for name in ("cache/depth_confhard", "cache/confidence", "motion_depth"):
        (bundle / name).mkdir(parents=True)
    previous_depth = previous_pose = None
    shape = (manifest["H"], manifest["W"])
    for row, pose in zip(manifest["rows"], poses):
        name = f"{row['frame_id']:06d}"
        raw_left = bundle / "raw_left" / f"{name}_depth.npy"
        raw_right = bundle / "raw_right" / f"{name}_depth.npy"
        left = cv2.cvtColor(cv2.imread(row["rgb"]), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(row["right"]), cv2.COLOR_BGR2RGB)
        dl = g19c.legacy.load_depth(raw_left, shape)
        dr = g19c.legacy.load_depth(raw_right, shape)
        result = g19c.compute(left, right, dl, dr, pose, previous_depth, previous_pose, calibration)
        np.save(bundle / "cache/depth_confhard" / f"{name}_depth.npy", result["depth_confhard"], allow_pickle=False)
        np.save(bundle / "cache/confidence" / f"{name}_confidence.npy", result["confidence"], allow_pickle=False)
        valid = np.isfinite(dl) & (dl >= .5) & (dl <= 80)
        motion_path = bundle / "motion_depth" / raw_left.name
        if np.all(valid | (dl == 0)):
            motion_path.symlink_to(raw_left)
        else:
            np.save(motion_path, np.where(valid, dl, 0).astype(np.float32), allow_pickle=False)
        previous_depth, previous_pose = dl, pose
    register_depths(manifest, bundle / "cache")


def verify_weights(args):
    frozen = read(STUDY / "stage2_frozen_sources.json")
    observed = {}
    for row in frozen["files"]:
        original = Path(row["path"])
        if "models" not in original.parts:
            continue
        model = args.semantic_model if "cityscapes-semantic" in original.parent.name else args.instance_model
        path = model / original.name
        observed[str(path)] = sha256(path)
        if observed[str(path)] != row["sha256"]:
            # torch.save containers may differ after conversion. The operator
            # must explicitly choose that alternative, which is recorded below.
            if not (args.converted_instance and model == args.instance_model
                    and original.name == "pytorch_model.bin"):
                raise ValueError(f"Mask2Former file differs from the registered SHA-256: {path}")
    stereo_lock = read(STEREO / "registration_cpu.json")["source_files"]
    observed[str(args.raft_checkpoint)] = sha256(args.raft_checkpoint)
    if observed[str(args.raft_checkpoint)] != stereo_lock["runtime/raftstereo-middlebury.pth"]:
        raise ValueError("RAFT-Stereo checkpoint differs from the registered SHA-256")
    return observed


def generate(args, bundle, manifest):
    weights = verify_weights(args)
    dump(bundle / "checkpoint_sha256.json", dict(files=weights, converted_instance=args.converted_instance))
    view = create_view(bundle, manifest, args.kitti_root, args.sequence)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    for camera in ("left", "right"):
        stage(camera, STEREO / "source" / f"{camera}.py",
              ["--raft-root", STEREO / "source/RAFT-Stereo", "--checkpoint", args.raft_checkpoint,
               "--sequence-dir", view, "--output-dir", bundle / f"raw_{camera}",
               "--num-frames", FRAMES, "--valid-iters", 32, "--min-depth", .5, "--max-depth", 80]
              + (["--preview-indices"] if camera == "left" else []), bundle)
    generate_depth_cache(bundle, manifest)
    dump(bundle / "runtime.json", manifest)
    image = ["--image-dir", view / "image_2"]
    stage("semantic", STAGE2 / "scripts/generate_mask2former_semantics_kitti.py",
          image + ["--model-id", args.semantic_model, "--local-files-only", "--output-dir", bundle / "semantic",
                   "--start", 0, "--num-frames", FRAMES, "--device", "cuda", "--preview-every", 100], bundle)
    stage("instances", STAGE2 / "scripts/run_mask2former_instance_tracking.py",
          image + ["--model-dir", args.instance_model, "--model-source-id",
                   "facebook/mask2former-swin-small-cityscapes-instance", "--model-revision", "4c0fb9b2",
                   "--dense-semantic-dir", bundle / "semantic", "--output-dir", bundle / "instances",
                   "--start", 0, "--end", FRAMES - 1, "--device", "cuda:0"], bundle)
    stage("motion", STAGE2 / "scripts/compute_motion_evidence.py",
          image + ["--depth-dir", bundle / "motion_depth", "--pose-file", manifest["pose_file"],
                   "--pose-coordinate", "p2", "--calib-file", bundle / "calib.txt",
                   "--instance-dir", bundle / "instances", "--output-dir", bundle / "motion",
                   "--start", 0, "--end", FRAMES - 1, "--device", "cuda:0"], bundle)
    stage("tsdd", STAGE2 / "scripts/run_tsdd.py",
          image + ["--instance-dir", bundle / "instances", "--evidence-dir", bundle / "motion",
                   "--config", STAGE2 / "configs/tsdd.yaml", "--output-dir", bundle / "states",
                   "--start", 0, "--end", FRAMES - 2, "--run-status", "ENGINEERING_SMOKE"], bundle)
    export(bundle, args.sequence)
    shutil.copyfile(bundle / "semantic/generation_summary.json", bundle / "semantic_inference_timing.json")


def prepare(args):
    if not args.cache and not (args.raft_checkpoint and args.semantic_model and args.instance_model):
        raise ValueError("Provide --cache or all three checkpoint arguments")
    bundle = args.workdir / "bundles" / args.sequence
    manifest = bundle_metadata(args.sequence, args.kitti_root, bundle, args.orb_poses)
    if args.cache:
        restore(args, bundle, manifest)
        dump(bundle / "runtime.json", manifest)
    else:
        generate(args, bundle, manifest)
    dump(bundle / "reproduction_bundle.json", dict(sequence=args.sequence, restored=bool(args.cache),
                                                   frames=FRAMES, GT_runtime=False))
    print(bundle)
