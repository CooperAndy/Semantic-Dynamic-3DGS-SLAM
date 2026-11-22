#!/usr/bin/env python3
"""Run the recorded S23 E2/E3 variants with an explicit local workspace."""
import argparse
import ast
import importlib.machinery
import os
import pprint
import shutil
import subprocess
import sys
from pathlib import Path

from experiment import LIMITS
from reproduction_common import STUDY, WINDOW_START, dump, read, sha256


def cell_directory(workdir, sequence, seed):
    stage = "E2" if sequence == "08" else "E3"
    return workdir / f"{stage}_K{sequence}_S{seed}"


def check_bundle(bundle, sequence, verify):
    sys.path.insert(0, str(STUDY / "backend/scripts"))
    try:
        from semantic_bridge import Bridge

        runtime = read(bundle / "runtime.json")
        if (runtime["frame_count"] != 1000 or runtime["pose_source"] != "ORB_estimate"
                or runtime["frame_convention"] != "P2" or runtime["GT_runtime"] is not False):
            raise ValueError("Expected a registered 1,000-frame causal P2 bundle")
        first = WINDOW_START[sequence]
        if [row["source_frame_id"] for row in runtime["rows"]] != list(range(first, first + 1000)):
            raise ValueError("Bundle source frames do not match the registered window")
        bridge = Bridge(bundle / "observations/manifest.json", runtime, "dynamic_filter")
        if bridge.identity["sequence"] != sequence:
            raise ValueError("Observation sequence mismatch")
        if sha256(runtime["pose_file"]) != runtime["pose_sha256"]:
            raise ValueError("ORB trajectory hash differs from the bundle")
        for row in runtime["rows"]:
            for key in ("rgb", "right", "depth", "confidence"):
                path = Path(row[key])
                if not path.is_file():
                    raise FileNotFoundError(path)
                if verify and sha256(path) != row["hashes"][key]:
                    raise ValueError(f"Input hash mismatch: {path}")
            if verify:
                bridge.read(row["frame_id"])
            elif not (bridge.path.parent / bridge.rows[row["frame_id"]]["path"]).is_file():
                raise FileNotFoundError(bridge.rows[row["frame_id"]]["path"])
        (bundle / "calib.txt").stat()
        return runtime
    finally:
        sys.path.pop(0)


def read_raster_lock(raster):
    lock = read(STUDY / "native-source-lock.json")["sources"]
    for name, digest in lock.items():
        path = raster / name
        if sha256(path) != digest:
            raise ValueError(f"Rasterizer source hash mismatch: {path}")
    package = raster / "diff_gaussian_rasterization"
    extension = next((package / f"_C{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES
                      if (package / f"_C{suffix}").is_file()), None)
    if extension is None:
        raise FileNotFoundError(f"Build the FP64 CUDA extension in place under {raster}")
    return {**lock, extension.relative_to(raster).as_posix(): sha256(extension)}


def configure_cell(cell, archive, bundle, runtime, raster, raster_lock):
    registration = read(archive / "registration.json")
    groups = registration["variants"]
    configs = {}
    for group in groups:
        source = archive / f"{group}.py"
        if sha256(source) != registration["config_sha256"][group]:
            raise ValueError(f"Registered configuration changed: {source}")
        configs[group] = ast.literal_eval(source.read_text().split("=", 1)[1].strip())

    cell.mkdir()
    for folder in ("source", "scripts"):
        shutil.copytree(STUDY / "backend" / folder, cell / folder,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for folder in ("runs", "receipts", "inputs/runtime"):
        (cell / folder).mkdir(parents=True)
    with (cell / "inputs/runtime/input_manifest.json").open("x", encoding="utf-8") as out:
        import json

        json.dump(runtime, out, indent=2)
    K = runtime["K"]
    config_path = cell / "source/configs/data/g20_manifest.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(dict(
        dataset_name="kitti", input_manifest=str(cell / "inputs/runtime/input_manifest.json"),
        camera_params=dict(image_height=runtime["H"], image_width=runtime["W"],
                           fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], png_depth_scale=1.0, crop_edge=0)
    )), encoding="utf-8")
    for group, config in configs.items():
        if (config["data"]["desired_image_height"], config["data"]["desired_image_width"]) != (
                runtime["H"], runtime["W"]):
            raise ValueError("Bundle resolution differs from the frozen configuration")
        if config["mapping"]["residual_fb"] != float((bundle / "projection_beta.txt").read_text()):
            raise ValueError("Stereo projection beta differs from the frozen configuration")
        config["workdir"] = str(cell / "runs")
        config["data"]["gradslam_data_cfg"] = str(config_path)
        config["data"]["basedir"] = str(bundle)
        config["semantic"]["manifest"] = str(bundle / "observations/manifest.json")
        target = cell / "source/configs/kitti" / f"{group}.py"
        target.write_text(f"config = {pprint.pformat(config, sort_dicts=False)}\n", encoding="utf-8")
    registration["archived_config_sha256"] = registration["config_sha256"]
    registration["config_sha256"] = {g: sha256(cell / f"source/configs/kitti/{g}.py") for g in groups}
    registration["native_numerical_revision"] = dict(
        raster=str(raster), raster_sha256=raster_lock, source="candidate_v4_fp64accum")
    registration["runtime_sha256"] = sha256(cell / "inputs/runtime/input_manifest.json")
    registration["observation_manifest_sha256"] = sha256(bundle / "observations/manifest.json")
    dump(cell / "registration.json", registration)
    dump(cell / "source_locks.json", {
        p.relative_to(cell).as_posix(): sha256(p)
        for folder in ("source", "scripts") for p in sorted((cell / folder).rglob("*")) if p.is_file()
    })


def launch(script, arguments, env, log, timeout, cwd):
    with log.open("x") as out:
        subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=cwd, env=env,
                       stdout=out, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def worker_env(cell, raster):
    return dict(os.environ, S23_CELL_ROOT=str(cell), S23_WORKDIR=str(cell.parent),
                S23_RASTER_ROOT=str(raster), PYTHONPATH=os.pathsep.join(map(str, [raster, cell / "scripts"])),
                PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                MPLBACKEND="Agg", WANDB_MODE="disabled")


def audit_existing(cell, bundle, raster, raster_lock):
    registration = read(cell / "registration.json")
    if registration["native_numerical_revision"] != dict(
            raster=str(raster), raster_sha256=raster_lock, source="candidate_v4_fp64accum"):
        raise ValueError("This cell was registered against another rasterizer build")
    if (registration["runtime_sha256"] != sha256(cell / "inputs/runtime/input_manifest.json")
            or read(cell / "inputs/runtime/input_manifest.json") != read(bundle / "runtime.json")
            or registration["observation_manifest_sha256"] != sha256(bundle / "observations/manifest.json")):
        raise ValueError("This cell was registered against another input bundle")
    for path, digest in read(cell / "source_locks.json").items():
        if sha256(cell / path) != digest:
            raise ValueError(f"Cell source hash changed: {path}")


def run_group(args, cell, group, env):
    if shutil.disk_usage(cell).free < LIMITS["startup_free_GiB"] * 2**30:
        raise RuntimeError("The registered startup budget requires 23 GiB free disk space")
    scripts = cell / "scripts"
    if group != "B":
        donor = cell / "receipts/donor.json"
        if not donor.is_file():
            raise FileNotFoundError("Run and seal B in this cell before any paired variant")
    launch(scripts / "worker.py", [group], env, cell / f"{group}.log",
           LIMITS["run_seconds"], scripts)
    out = cell / "runs" / group
    receipt = read(out / "receipt.json")
    if (receipt["status"] != "PASS_NATIVE_MAPPING_INTEGRATION" or receipt["frames"] != 1000
            or receipt["mapping_backward"] != 57000 or receipt["GT_blocked_opens"] != 0):
        raise RuntimeError(f"Unsealed native result: {out}")
    launch(STUDY / "audit_native.py", [out], env, cell / f"{group}_audit.log", 900, STUDY)
    if group == "B":
        files = [out / "target_schedule.npz", *sorted((out / "addition_masks").glob("*.npz"))]
        if len(files) != 950:
            raise RuntimeError("Full B donor schedule and 949 birth masks are required")
        dump(cell / "receipts/donor.json", dict(status="PASS_SEALED_DONOR_INPUTS",
             files=[dict(path=p.relative_to(cell).as_posix(), sha256=sha256(p)) for p in files]))


def evaluate_group(args, cell, group, env):
    if not args.kitti_root or not args.labels_root:
        raise ValueError("Evaluation needs --kitti-root and --labels-root")
    launch(STUDY / "matrix_evaluation.py",
           ["--root", cell, "--group", group, "--kitti-root", args.kitti_root, "--labels-root", args.labels_root],
           env, cell / f"{group}_evaluation.log", LIMITS["run_seconds"], STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inputs", "check", "run", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--workdir", type=lambda x: Path(x).resolve(), required=True)
        p.add_argument("--sequence", choices=WINDOW_START, required=True)
        if command == "inputs":
            p.add_argument("--kitti-root", type=Path, required=True)
            p.add_argument("--orb-poses", type=Path, required=True)
            for name in ("cache", "runtime-lock", "observation-lock", "raft-checkpoint", "semantic-model", "instance-model"):
                p.add_argument("--" + name, type=Path)
            p.add_argument("--converted-instance", action="store_true")
        if command == "check":
            p.add_argument("--verify-hashes", action="store_true")
        if command in ("run", "evaluate"):
            p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
            p.add_argument("--group", choices=("all", "B", "T", "M", "TM", "D", "TD", "MD", "TMD"), default="all")
            p.add_argument("--raster-root", type=Path, default=STUDY / "native")
            p.add_argument("--kitti-root", type=Path)
            p.add_argument("--labels-root", type=Path)
            if command == "run":
                p.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())

    if args.command == "inputs":
        from reproduction_inputs import prepare

        prepare(args)
        return
    bundle = args.workdir / "bundles" / args.sequence
    runtime = check_bundle(bundle, args.sequence, getattr(args, "verify_hashes", False))
    if args.command == "check":
        print(f"READY: {args.sequence}, {len(runtime['rows'])} frames, {bundle}")
        return
    if args.workdir.is_relative_to(STUDY.parents[1]):
        parser.error("--workdir must sit outside the source repository")
    if args.command == "run" and args.evaluate and not (args.kitti_root and args.labels_root):
        parser.error("--evaluate requires --kitti-root and --labels-root")
    cell = cell_directory(args.workdir, args.sequence, args.seed)
    archive = STUDY / "configurations" / (cell.name + "_R3_serial")
    if not archive.is_dir():
        parser.error("The requested sequence/seed was not registered")
    groups = read(archive / "registration.json")["variants"]
    if args.group != "all":
        if args.group not in groups:
            parser.error("The requested group was not registered for this sequence")
        groups = [args.group]
    raster_lock = read_raster_lock(args.raster_root)
    if cell.exists():
        audit_existing(cell, bundle, args.raster_root, raster_lock)
    elif args.command == "run":
        if groups[0] != "B":
            parser.error("The first run in each cell must be B")
        configure_cell(cell, archive, bundle, runtime, args.raster_root, raster_lock)
    else:
        parser.error(f"Run the requested cell before evaluating it: {cell}")
    env = worker_env(cell, args.raster_root)
    for group in groups:
        if args.command == "run":
            run_group(args, cell, group, env)
        if args.command == "evaluate" or args.evaluate:
            evaluate_group(args, cell, group, env)
    print(cell)


if __name__ == "__main__":
    main()
