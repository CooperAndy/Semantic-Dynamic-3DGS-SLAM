"""Method-independent KITTI08 renderer/evaluator for a frozen split.

Dev is available for smoke and candidate selection. Test requires a separate
frozen-decision file and is never run by the stage-2 smoke workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/SplaTAM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mse = float(np.mean((pred[mask] - target[mask]) ** 2))
    return -10.0 * math.log10(mse) if mse else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["Dev", "Test"], default="Dev")
    parser.add_argument("--frame-ids", type=int, nargs="*", help="Original KITTI frame IDs; omit for entire split")
    parser.add_argument("--frozen-decision", type=Path, help="Required for one-shot Test; not used in smoke")
    parser.add_argument("--test-batch-name", help="Pre-registered method name in the one-shot Test batch")
    parser.add_argument("--run-id", help="Optional unique method/run label; does not affect rendering")
    args = parser.parse_args()
    root = args.protocol_root.resolve()
    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "split.csv") != manifest["split_sha256"]:
        raise ValueError("split hash mismatch")
    if sha256(root / "roi_mask_ledger.csv") != manifest["roi_mask_ledger_sha256"]:
        raise ValueError("ROI mask ledger hash mismatch")
    if args.out.exists():
        raise FileExistsError(f"refusing existing evaluation output: {args.out}")
    batch_progress_path = root / "test_batch_progress.json"
    batch_decision_sha = None
    batch_candidates = None
    if args.split == "Test":
        if args.frozen_decision is None or not args.frozen_decision.is_file():
            raise PermissionError("Test requires an explicit frozen candidate decision file")
        decision = json.loads(args.frozen_decision.read_text(encoding="utf-8"))
        batch_candidates = decision.get("candidate_order")
        if (decision.get("split_sha256") != manifest["split_sha256"]
                or decision.get("roi_mask_ledger_sha256") != manifest["roi_mask_ledger_sha256"]
                or decision.get("evaluator_sha256") != sha256(Path(__file__))
                or not isinstance(batch_candidates, list) or not batch_candidates
                or not args.test_batch_name):
            raise PermissionError("Test lock does not match the frozen protocol or batch")
        names = [candidate.get("name") for candidate in batch_candidates]
        if len(names) != len(set(names)) or args.test_batch_name not in names:
            raise PermissionError("unregistered or duplicate Test method name")
        # Check every registered map before opening even the first Test image.
        for candidate in batch_candidates:
            if sha256(Path(candidate["map_path"])) != candidate["map_sha256"]:
                raise PermissionError(f"Test map hash mismatch: {candidate['name']}")
        batch_decision_sha = sha256(args.frozen_decision)
        if (root / "test_consumed.json").exists():
            raise FileExistsError("Test was already consumed for this protocol")
        if batch_progress_path.exists():
            progress = json.loads(batch_progress_path.read_text(encoding="utf-8"))
            if progress.get("decision_sha256") != batch_decision_sha:
                raise PermissionError("Test batch decision changed after starting")
        else:
            progress = {"decision_sha256": batch_decision_sha, "completed": [], "summaries": {}}
        completed = progress["completed"]
        if completed != names[:len(completed)] or len(completed) >= len(names):
            raise PermissionError("invalid Test batch progress")
        if args.test_batch_name != names[len(completed)]:
            raise PermissionError("Test methods must run in the pre-registered order")
        registered = batch_candidates[len(completed)]
        if Path(registered["map_path"]).resolve() != args.map.resolve():
            raise PermissionError("Test map path does not match batch decision")

    with (root / "split.csv").open(encoding="utf-8", newline="") as handle:
        split = {int(row["original_frame_id"]): row["split"] for row in csv.DictReader(handle)}
    with (root / "roi_mask_ledger.csv").open(encoding="utf-8", newline="") as handle:
        mask_hashes = {int(row["original_frame_id"]): row["mask_sha256"] for row in csv.DictReader(handle)}
    frame_ids = args.frame_ids if args.frame_ids is not None else [frame for frame, role in split.items() if role == args.split]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)) or any(split.get(i) != args.split for i in frame_ids):
        raise ValueError("evaluation frames must be unique members of requested split")
    if args.split == "Test" and len(frame_ids) != 20:
        raise ValueError("formal Test must cover all 20 frozen frames exactly once")
    if args.split == "Test" and not batch_progress_path.exists():
        # This marker is deliberately written before any Test RGB is loaded.
        batch_progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    from utils.recon_helpers import setup_camera
    from pytorch_msssim import ms_ssim
    import lpips

    torch.cuda.init()
    lpips_model = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
    source = Path("/root/autodl-tmp/seq21_chy_20260926")
    original = source / "sequences/21_1200_1399"
    raw_poses = np.loadtxt(source / "tracking/poses_orb_p2_c2w_rebased.txt", dtype=np.float64).reshape(200, 3, 4)
    c2w = np.tile(np.eye(4, dtype=np.float64), (200, 1, 1))
    c2w[:, :3, :4] = raw_poses
    first_inverse = np.linalg.inv(c2w[0])

    with np.load(args.map, allow_pickle=False) as params:
        k = params["intrinsics"].astype(np.float32)
        width, height = int(params["org_width"]), int(params["org_height"])
        if (height, width) != (376, 1241):
            raise ValueError("unexpected map resolution")
        means = torch.from_numpy(params["means3D"].copy()).cuda()
        colors = torch.from_numpy(params["rgb_colors"].copy()).cuda()
        rotations = F.normalize(torch.from_numpy(params["unnorm_rotations"].copy()).cuda(), dim=1)
        opacities = torch.sigmoid(torch.from_numpy(params["logit_opacities"].copy()).cuda())
        log_scales = torch.from_numpy(params["log_scales"].copy()).cuda()
        if log_scales.shape[1] == 1:
            log_scales = log_scales.repeat(1, 3)
        scales = torch.exp(log_scales)
        gaussian_count = int(means.shape[0])
    map_hash = sha256(args.map)
    renderer_version = f"SplaTAM_native_cuda_torch_{torch.__version__}"
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for frame_id in frame_ids:
        index = frame_id - 1200
        relative_c2w = first_inverse @ c2w[index]
        w2c = np.linalg.inv(relative_c2w).astype(np.float32)
        cam = setup_camera(width, height, k, w2c, near=0.01, far=150.0)
        common = dict(image_height=cam.image_height, image_width=cam.image_width,
                      tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
                      scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
                      projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
                      campos=cam.campos, prefiltered=cam.prefiltered)
        means2d = torch.zeros_like(means)
        base_args = dict(means3D=means, means2D=means2d, rotations=rotations,
                         opacities=opacities, scales=scales)
        bg = torch.tensor([0.055, 0.065, 0.08], device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            rgb, _, _ = Renderer(raster_settings=Camera(bg=bg, **common))(
                colors_precomp=colors, **base_args)
            alpha_rgb, _, _ = Renderer(raster_settings=Camera(bg=torch.zeros_like(bg), **common))(
                colors_precomp=torch.ones_like(colors), **base_args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred = np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0, 1)
        alpha = np.clip(alpha_rgb[0].cpu().numpy(), 0, 1)
        gt_path = original / "image_2" / f"{index:06d}.png"
        target = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        mask_path = source / "static_valid_masks" / f"{index:06d}_valid.npy"
        if sha256(mask_path) != mask_hashes[frame_id]:
            raise ValueError(f"static ROI changed for frame {frame_id}")
        roi = np.load(mask_path, allow_pickle=False)
        if roi.dtype != np.bool_ or roi.shape != (height, width) or not roi.any():
            raise ValueError(f"empty or bad static ROI for frame {frame_id}")
        if pred.shape != target.shape or float(alpha.max()) <= 0:
            raise ValueError(f"empty or mismatched render for frame {frame_id}")
        pred_tensor = torch.from_numpy(pred.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            msssim_value = float(ms_ssim(pred_tensor, target_tensor, data_range=1.0).item())
            lpips_value = float(lpips_model(pred_tensor * 2 - 1, target_tensor * 2 - 1).mean().item())
        row = {
            "run_id": args.run_id or f"{args.map.parent.name}/{args.map.stem}", "original_frame_id": frame_id,
            "source_local_index": index, "split": args.split,
            "map_sha256": map_hash, "renderer_version": renderer_version,
            "mask_sha256": mask_hashes[frame_id], "roi_pixels": int(roi.sum()),
            "roi_fraction": float(roi.mean()),
            "psnr_db": psnr(pred, target, roi),
            "psnr_full_db": psnr(pred, target, np.ones_like(roi, dtype=np.bool_)),
            "ms_ssim": msssim_value, "lpips": lpips_value,
            "hole_rate": float(((alpha < 0.5) & roi).sum() / roi.sum()),
            "gaussian_count": gaussian_count, "render_seconds": elapsed,
            "alpha_max": float(alpha.max()),
        }
        rows.append(row)
        Image.fromarray(np.round(pred * 255).astype(np.uint8)).save(args.out / f"render_{frame_id}.png")
        Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(args.out / f"alpha_{frame_id}.png")
        Image.fromarray((roi.astype(np.uint8) * 255)).save(args.out / f"static_roi_{frame_id}.png")
        print(json.dumps(row), flush=True)
    with (args.out / "per_frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split, "frames": frame_ids, "count": len(rows),
        "mean_static_roi_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "mean_full_psnr_db": float(np.mean([row["psnr_full_db"] for row in rows])),
        "mean_ms_ssim": float(np.mean([row["ms_ssim"] for row in rows])),
        "mean_lpips": float(np.mean([row["lpips"] for row in rows])),
        "mean_hole_rate": float(np.mean([row["hole_rate"] for row in rows])),
        "ms_ssim_lpips_protocol": "full RGB, same resolution; MS-SSIM pytorch_msssim data_range=1; LPIPS AlexNet v0.1 with [-1,1] RGB",
        "split_sha256": manifest["split_sha256"], "map_sha256": map_hash,
        "evaluator_sha256": sha256(Path(__file__)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.split == "Test":
        progress["completed"].append(args.test_batch_name)
        progress["summaries"][args.test_batch_name] = summary
        batch_progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        if len(progress["completed"]) == len(batch_candidates):
            (root / "test_consumed.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
