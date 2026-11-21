"""GT-blind G08 capped-native replay with 20 selected adjacent checkpoints."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.protocol = args.protocol.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(args.out)
    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["status"] != "FROZEN_BEFORE_RUN" or p["num_frames"] != 200:
        raise ValueError("Unexpected protocol")
    root = Path(p["source_root"])
    source = root / "source"
    files = {
        "source_config_sha256": source / "configs/kitti/G00_S0.py",
        "source_core_sha256": source / "scripts/splatam_external_tm.py",
        "source_c04n_addition_sha256": source / "c04n_addition.py",
        "source_data_config_sha256": source / "configs/data/g00_manifest.yaml",
        "input_manifest_sha256": root / "inputs/runtime/input_manifest.json",
    }
    for key, path in files.items():
        if sha256(path) != p[key]:
            raise ValueError(f"Frozen input mismatch: {key}")
    frames = p["future_holdouts"]
    if frames != list(range(19, 200, 20)) or len(p["events"]) != 10:
        raise ValueError("Event registration mismatch")
    selected = [i for event in p["events"] for i in (event["before"], event["after"])]
    if selected != [i for event in range(10) for i in (20 * event + 17, 20 * event + 18)]:
        raise ValueError("Checkpoint registration mismatch")
    args.out.mkdir(parents=True)
    receipt_path = args.out / "receipt.json"
    receipt = {
        "status": "RUNNING", "protocol_sha256": sha256(args.protocol),
        "runner_sha256": sha256(Path(__file__)),
        "source_hashes": {key: sha256(path) for key, path in files.items()},
        "saved_snapshots": [], "GT_access": "audit_hook_blocks_GT_paths",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    started = time.monotonic()

    def guard(event, items):
        if event == "open" and items and isinstance(items[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(items[0]).replace("\\", "/").lower()
            if path.endswith("groundtruth.txt") or "/poses/" in path or "/evaluation_only/" in path:
                raise RuntimeError("GT/evaluation path blocked in capped replay")

    sys.addaudithook(guard)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        sys.path.insert(0, str(source))
        os.chdir(source)
        config = load("m15_capped_config", source / "configs/kitti/G00_S0.py").config
        config["workdir"] = str(args.out.parent)
        config["run_name"] = args.out.name
        config["data"]["num_frames"] = p["num_frames"]
        config["eval_only_frames"] = [i for i in config["eval_only_frames"] if i < p["num_frames"]]
        config["save_checkpoints"] = True
        config["checkpoint_interval"] = 1
        if config["eval_only_frames"] != frames or config["map_every"] != 1:
            raise ValueError("Config/holdout mismatch")
        if (config["mapping"]["candidate_rule"], config["mapping"]["new_point_cap"],
                config["mapping"]["sampling_seed"]) != ("native", 3000, 0):
            raise ValueError("G08 capped-native selection mismatch")
        core = load("m15_capped_core", source / "scripts/splatam_external_tm.py")
        core.seed_everything(config["seed"])
        from c04n_addition import add_new_gaussians as selected_addition
        core.add_new_gaussians = lambda *a, **k: selected_addition(
            *a, **k, core=core, mode=config["mapping"]["candidate_rule"],
            cap=config["mapping"]["new_point_cap"], seed=config["mapping"]["sampling_seed"],
            log_path=args.out / "addition_history.jsonl", fb=config["mapping"]["residual_fb"])
        original_save = core.save_params_ckpt
        selected_set = set(selected)

        def save_selected(params, output_dir, time_idx):
            if time_idx in selected_set:
                original_save(params, output_dir, time_idx)
                receipt["saved_snapshots"].append(time_idx)

        core.save_params_ckpt = save_selected
        core.rgbd_slam(config)
        torch.cuda.synchronize()
        if receipt["saved_snapshots"] != selected:
            raise RuntimeError("Missing or repeated registered checkpoint")
        with (args.out / "addition_history.jsonl").open(encoding="utf-8") as stream:
            adds = [json.loads(line) for line in stream]
        if len(adds) != 189 or any(x["selected_count"] > 3000 for x in adds):
            raise RuntimeError("Capped addition audit failed")
        receipt["addition_history_sha256"] = sha256(args.out / "addition_history.jsonl")
        receipt["addition_events"] = len(adds)
        receipt["map_sha256"] = {str(i): sha256(args.out / f"params{i}.npz") for i in selected}
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
