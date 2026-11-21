"""Export compact metrics and array metadata without copying a large params.npz."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--log")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    exp = Path(args.experiment_dir)
    params_path = exp / "params.npz"
    raw = np.load(params_path, allow_pickle=True)
    result = {
        "name": args.name,
        "experiment_dir": str(exp.resolve()),
        "metrics": {},
        "gaussian_count": int(raw["means3D"].shape[0]),
        "keyframe_count": int(raw["keyframe_time_indices"].shape[0]) if "keyframe_time_indices" in raw else None,
        "params_bytes": params_path.stat().st_size,
        "runtime": {},
    }
    for metric in ["psnr", "rmse", "l1", "ssim", "lpips"]:
        x = np.atleast_1d(np.loadtxt(exp / "eval" / f"{metric}.txt")).astype(float)
        result["metrics"][metric] = {"mean": float(x.mean()), "samples": len(x)}
    if args.log:
        text = Path(args.log).read_text(encoding="utf-8", errors="replace")
        patterns = {
            "tracking_frame_seconds": r"Average Tracking/Frame Time: ([0-9.eE+-]+) s",
            "mapping_iteration_ms": r"Average Mapping/Iteration Time: ([0-9.eE+-]+) ms",
            "mapping_frame_seconds": r"Average Mapping/Frame Time: ([0-9.eE+-]+) s",
        }
        for key, pattern in patterns.items():
            found = re.findall(pattern, text)
            result["runtime"][key] = float(found[-1]) if found else None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
