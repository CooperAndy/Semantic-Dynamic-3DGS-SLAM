"""Isolated, GT-blind 45-frame replay saving selected adjacent map updates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    # The mapping core changes cwd to the source tree. Resolve both CLI paths
    # first so checkpoints and the final receipt stay in the isolated run root.
    args.protocol = args.protocol.resolve()
    args.out = args.out.resolve()
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["status"] != "FROZEN_BEFORE_RUN":
        raise ValueError("Protocol is not frozen")
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path(p["source_root"])
    source = root / "source"
    files = {
        "source_config_sha256": source / "configs/kitti/G00_S0.py",
        "source_core_sha256": source / "scripts/splatam_external_tm.py",
        "source_data_config_sha256": source / "configs/data/g00_manifest.yaml",
        "input_manifest_sha256": root / "inputs/runtime/input_manifest.json",
    }
    for key, path in files.items():
        if sha256(path) != p[key]:
            raise ValueError(f"Frozen input mismatch: {key}")
    if (p["num_frames"], p["save_snapshots"]) != (45, [17, 18, 37, 38]):
        raise ValueError("Unexpected pilot design")
    args.out.mkdir(parents=True)
    receipt_path = args.out / "receipt.json"
    receipt = {
        "status": "RUNNING",
        "protocol_sha256": sha256(args.protocol),
        "runner_sha256": sha256(Path(__file__)),
        "source_hashes": {key: sha256(path) for key, path in files.items()},
        "saved_snapshots": [],
        "GT_access": "audit_hook_blocks_GT_paths",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    started = time.monotonic()

    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/poses/" in path or "/evaluation_only/" in path:
                raise RuntimeError("GT/evaluation path blocked in replay")

    sys.addaudithook(guard)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        sys.path.insert(0, str(source))
        os.chdir(source)
        config = load("m15_dense_config", source / "configs/kitti/G00_S0.py").config
        config["workdir"] = str(args.out.parent)
        config["run_name"] = args.out.name
        config["data"]["num_frames"] = p["num_frames"]
        config["eval_only_frames"] = [x for x in config["eval_only_frames"] if x < p["num_frames"]]
        config["save_checkpoints"] = True
        config["checkpoint_interval"] = 1
        if config["tracking"]["camera_policy"] != "external_fixed" or config["map_every"] != 1:
            raise ValueError("Source config changed camera or mapping policy")
        if config["eval_only_frames"] != [19, 39]:
            raise ValueError("Unexpected heldout frames")
        core = load("m15_dense_core", source / "scripts/splatam_external_tm.py")
        core.seed_everything(config["seed"])
        original_save = core.save_params_ckpt
        selected = set(p["save_snapshots"])

        def save_selected(params, output_dir, time_idx):
            if time_idx in selected:
                original_save(params, output_dir, time_idx)
                receipt["saved_snapshots"].append(time_idx)

        core.save_params_ckpt = save_selected
        core.rgbd_slam(config)
        torch.cuda.synchronize()
        if receipt["saved_snapshots"] != p["save_snapshots"]:
            raise RuntimeError("Missing or repeated selected checkpoint")
        receipt["map_sha256"] = {
            str(index): sha256(args.out / f"params{index}.npz") for index in p["save_snapshots"]
        }
        receipt["final_map_sha256"] = sha256(args.out / "params.npz")
        receipt["status"] = "COMPLETED"
    except BaseException as error:
        receipt.update(status="FAILED_PRESERVED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
