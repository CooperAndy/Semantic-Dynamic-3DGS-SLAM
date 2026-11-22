"""Build the registered 20-view evaluation references from official local data."""
import shutil
from pathlib import Path

import numpy as np

from reproduction_common import dump, read, sha256


def prepare_references(root, kitti_root, labels_root):
    from automatic_gt import official_gt

    runtime = read(root / "inputs/runtime/input_manifest.json")
    sequence = read(root / "registration.json")["cell"]["sequence"]
    destination = root.parent / "evaluation_only" / f"K{sequence}"
    destination.mkdir(parents=True, exist_ok=True)
    sequence_dir = kitti_root / "sequences" / sequence
    sources = dict(velodyne=sequence_dir / "velodyne", labels=labels_root / "sequences" / sequence / "labels")
    calib = sequence_dir / "calib.txt"
    poses = kitti_root / "poses" / f"{sequence}.txt"
    calib_sha = sha256(calib)
    pose_sha = sha256(poses)
    receipt = destination / "reference_receipt.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["calib_sha256"] != calib_sha or saved["pose_sha256"] != pose_sha:
            raise ValueError("This evaluation workspace already contains a different calibration or GT trajectory")
    if not (destination / "calib.txt").exists():
        shutil.copyfile(calib, destination / "calib.txt")
    ids = np.asarray([row["source_frame_id"] for row in runtime["rows"]], dtype=int)
    p2 = next(line.split(":", 1)[1] for line in calib.read_text().splitlines() if line.startswith("P2:"))
    p2 = np.fromstring(p2, sep=" ").reshape(3, 4)
    if not np.allclose(p2[:, :3], runtime["K"], rtol=0, atol=5e-4):
        raise ValueError("Official and runtime P2 intrinsics differ")
    gt = np.tile(np.eye(4), (len(ids), 1, 1))
    gt[:, :3] = np.loadtxt(poses, dtype=np.float64)[ids].reshape(-1, 3, 4)
    p0_from_p2 = np.eye(4)
    p0_from_p2[:3, 3] = -np.linalg.solve(p2[:, :3], p2[:, 3])
    gt = gt @ p0_from_p2
    trajectory = destination / "trajectory_reference.npz"
    if not trajectory.exists():
        np.savez_compressed(trajectory, P2_c2w=gt, source_frame_ids=ids)
    frames = list(range(19, 200, 20)) + list(range(819, 1000, 20))
    source_files = {}
    for kind, directory in sources.items():
        (destination / kind).mkdir(exist_ok=True)
        records = []
        extension = "bin" if kind == "velodyne" else "label"
        for logical in frames:
            frame = int(ids[logical])
            name = f"{frame:06d}.{extension}"
            path = directory / name
            expected = sha256(path)
            target = destination / kind / name
            if not target.exists():
                shutil.copyfile(path, target)
            elif sha256(target) != expected:
                raise ValueError(f"Existing reference differs from the registered bytes: {target}")
            records.append(dict(frame=frame, path=str(target), sha256=expected))
        source_files[kind] = records
    for logical in frames:
        frame = int(ids[logical])
        target = destination / f"{frame:06d}_GT.npz"
        if not target.exists():
            arrays, _ = official_gt(destination / "velodyne" / f"{frame:06d}.bin",
                                    destination / "labels" / f"{frame:06d}.label", calib,
                                    (runtime["H"], runtime["W"]))
            np.savez_compressed(target, **arrays)
    if not receipt.exists():
        dump(receipt, dict(status="OFFICIAL20_VIEW_SPARSE_REFERENCES_READY", sequence=sequence,
                           GT_prediction=False, source_files=source_files, pose_sha256=pose_sha,
                           calib_sha256=calib_sha, dense_ROI=False))
    return destination, gt
