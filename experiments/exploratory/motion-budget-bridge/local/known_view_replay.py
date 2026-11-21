"""Replay one G09 historical render with the M1.5 camera adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/root/autodl-tmp/stage3_g07_g08_20261025_p54957/C1000")
G09 = Path("/root/autodl-tmp/stage3_g09_checkpoint_v2_20261025_p54957")
HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol_support_v1.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    import torch
    import diff_gaussian_rasterization as rasterizer
    source = ROOT / "source"
    sys.path.insert(0, str(source))
    os.chdir(source)
    from utils.recon_helpers import setup_camera
    from external_tracking import render, set_external
    spec = importlib.util.spec_from_file_location("m15_replay_core", source / "scripts/splatam_external_tm.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    variant, snap, frame = "G00_S0", "199", 19
    map_file = ROOT / "runs" / variant / f"params{snap}.npz"
    if sha256(map_file) != PROTOCOL["variants"][variant][snap]:
        raise ValueError("Map hash mismatch")
    manifest = json.loads((ROOT / "inputs/runtime/input_manifest.json").read_text(encoding="utf-8"))
    pose_rows = np.loadtxt(ROOT / "inputs/runtime/external_orb_P2_c2w.txt").reshape(-1, 3, 4)
    c2w = np.eye(4)
    c2w[:3, :] = pose_rows[frame]
    anchor = np.linalg.inv(c2w)
    with np.load(map_file, allow_pickle=False) as archive:
        params = {key: torch.as_tensor(archive[key], device="cuda") for key in
                  ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales")}
        params["cam_unnorm_rots"] = torch.as_tensor(archive["cam_unnorm_rots"], device="cuda").clone()
        params["cam_trans"] = torch.as_tensor(archive["cam_trans"], device="cuda").clone()
    data = {"cam": setup_camera(manifest["W"], manifest["H"], np.asarray(manifest["K"]), np.eye(4)),
            "w2c": torch.eye(4, device="cuda")}
    set_external(params, torch.as_tensor(anchor, device="cuda", dtype=torch.float32), frame, core)
    rgb, ds = render(params, data, frame, core)
    reference = G09 / "renders/G00_S0/199/000019.npz"
    with np.load(reference, allow_pickle=False) as old:
        rgb_max_abs = float(np.max(np.abs(rgb - old["rgb"])))
        ds_max_abs = float(np.max(np.abs(ds - old["depth_silhouette"])))
        rgb_mean_abs = float(np.mean(np.abs(rgb - old["rgb"])))
        ds_mean_abs = float(np.mean(np.abs(ds - old["depth_silhouette"])))
    result = {"status": "COMPLETED", "variant": variant, "snapshot": snap, "frame": frame,
              "map_sha256": sha256(map_file), "reference_sha256": sha256(reference),
              "rasterizer_sha256": sha256(Path(rasterizer._C.__file__)),
              "coverage099": float((ds[1] > .99).mean()),
              "rgb_max_abs": rgb_max_abs, "rgb_mean_abs": rgb_mean_abs,
              "depth_silhouette_max_abs": ds_max_abs,
              "depth_silhouette_mean_abs": ds_mean_abs}
    (HERE / "known_view_replay_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
