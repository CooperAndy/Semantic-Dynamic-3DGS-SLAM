"""Hash-checked restoration and the isolated vehicle-state intervention."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
S23 = ROOT.parent
MODEL = Path("/root/autodl-tmp/stage2_new_model_pi3mos")
DEMO = MODEL.parent / "stage2_kitti08_pi3mos_demo_20261106"
WEIGHT = "1a536b1bb6e9c6c16a77b7ed544649351d387e281a16f3d946e0786cdcf23ded"
VEHICLES = {"car", "truck", "bus"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def restore(sequence, identity):
    assert re.fullmatch("[0-9a-f]{64}", identity)
    stage = ROOT / "restored"
    stage.mkdir(exist_ok=True)
    receipt = stage / (identity + ".json")
    if receipt.exists():
        return
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as stream:
        head = stream.next()
        assert head.name == "manifest.json" and head.size < 2**20
        locks = json.load(stream.extractfile(head))
        seen = set()
        for member in stream:
            if member.name == "manifest.json":
                continue
            prefix = f"bundles/{sequence}/"
            assert member.name.startswith(tuple(prefix + p + "/" for p in ("cache", "observations", "instances")))
            path = stage / member.name
            assert path.resolve().is_relative_to(stage.resolve())
            expected = locks[member.name]
            assert member.isfile() and member.name not in seen and member.size == expected["bytes"]
            data = stream.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == expected["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert sha(path) == expected["sha256"]
            else:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            seen.add(member.name)
        assert seen == locks.keys()
    write(receipt, {"sha256": identity, "verified_members": len(seen), "sequence": sequence})


class Predictor:
    def __init__(self):
        import torch
        from pi3.models.pi3 import Pi3
        from safetensors.torch import load_file
        self.torch = torch
        weights = MODEL / "checkpoints/pi3mos.safetensors"
        assert torch.cuda.is_bf16_supported() and sha(weights) == WEIGHT
        self.model = Pi3().eval()
        self.model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model.cuda()
        self.cache = {}

    def predict(self, frame, paths, shape):
        from PIL import Image
        from torchvision.transforms import functional as TF, InterpolationMode
        torch = self.torch
        window = range(frame - 7, frame + 1)
        self.cache = {i: x for i, x in self.cache.items() if i in window}
        for i in window:
            if i not in self.cache:
                with Image.open(paths[i]) as im:
                    assert (im.height, im.width) == shape
                    self.cache[i] = TF.resize(TF.to_tensor(im.convert("RGB")), [280, 910],
                                              interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = torch.stack([self.cache[i] for i in window]).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = self.model(x)
        low = y["dynamic_masks"][0, -1].squeeze().float()
        up = torch.nn.functional.interpolate(low[None, None], size=shape, mode="bilinear",
                                             align_corners=False)[0, 0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert up.isfinite().all() and up.min() >= 0 and up.max() <= 1
        return low.cpu().numpy(), up.cpu().numpy(), elapsed


def sanity():
    import numpy as np
    images = Path("/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/08/image_2")
    original = Path("/root/autodl-tmp/slam_project/stage2_tsdd/outputs/instances/seq08_mask2former_instance_flowwarp_v2_3000_3599")
    predictor = Predictor()
    errors = {}
    for i in (3007, 3015, 3024):
        paths = {j: images / f"{j:06}.png" for j in range(i - 7, i + 1)}
        low, up, _ = predictor.predict(i, paths, (370, 1226))
        f = f"{i:06}"
        with np.load(DEMO / "inference" / f"{f}.npz") as source:
            error = float(np.max(np.abs(low.astype(np.float16).astype(np.float32) - source["prob"])))
        assert error <= 0.001, (i, error)
        with np.load(original / f"{f}_masks.npz") as source:
            for row in read(DEMO / "inference" / f"{f}.json")["objects"]:
                mask = source["masks"][row["mask_index"]].astype(bool)
                if row["score"] is not None:
                    score = float(np.median(up[mask]))
                    assert abs(score - row["score"]) <= 0.001
                    assert state(score) == row["raw_state"]
        errors[i] = error
    assert [state(v) for v in (None, .499, .5)] == ["unknown", "static", "dynamic"]
    write(ROOT / "parity.json", {"status": "PASS_THREE_REFERENCE_FRAMES", "max_absolute_error": errors,
                                 "checkpoint_sha256": WEIGHT})


def state(score):
    return "unknown" if score is None else "dynamic" if score >= 0.5 else "static"


def generate(sequence):
    import numpy as np
    old = S23 / "bundles" / sequence
    restored = ROOT / "restored/bundles" / sequence
    dest = ROOT / "inputs" / sequence
    out = dest / "observations"
    out.mkdir(parents=True, exist_ok=False)
    probs = dest / "probabilities"
    probs.mkdir()
    records = (dest / "vehicle_scores.jsonl").open("x", encoding="utf-8")
    runtime = read(old / "runtime.json")
    original = read(restored / "observations/manifest.json")
    assert original["sequence"] == sequence and original["GT_runtime"] is runtime["GT_runtime"] is False
    shape = (runtime["H"], runtime["W"])
    assert len(runtime["rows"]) == len(original["rows"]) == 1000
    assert sha(runtime["pose_file"]) == runtime["pose_sha256"]
    for i, (row, obs) in enumerate(zip(runtime["rows"], original["rows"])):
        assert i == row["frame_id"] == row["source_frame_id"] == obs["source_frame"] == obs["frame_id"]
        assert obs["rgb_sha256"] == row["hashes"]["rgb"]
        for key in ("rgb", "right", "depth", "confidence"):
            path = Path(row[key])
            if key in ("depth", "confidence"):
                path = restored / path.relative_to(old)
                row[key] = str(path)
            assert sha(path) == row["hashes"][key], (key, i)
    images = [row["rgb"] for row in runtime["rows"]]
    predictor = Predictor()
    rows, forward = [], 0.0
    for i, old_obs in enumerate(original["rows"]):
        f = f"{i:06}"
        probability = None
        if i >= 7:
            low, probability, elapsed = predictor.predict(i, images, shape)
            np.savez_compressed(probs / f"{f}.npz", probability=low.astype(np.float16))
            forward += elapsed
        original_file = restored / "observations" / old_obs["path"]
        assert sha(original_file) == old_obs["sha256"]
        with np.load(original_file, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        assert set(arrays) == {"category", "confidence", "instance_id", "state"}
        assert all(a.shape == shape for a in arrays.values())
        original_state = arrays["state"]
        updated = original_state.copy()
        changed_scope = np.zeros(shape, dtype=bool)
        metadata = read(restored / "instances" / f"{f}.json")
        assert metadata["frame_id"] == f
        frame_records = []
        seen = set()
        with np.load(restored / "instances" / f"{f}_masks.npz", allow_pickle=False) as archive:
            masks = archive["masks"]
            assert masks.shape[1:] == shape
            for obj in metadata["instances"]:
                category = obj["category"].lower()
                tid = int(obj["track_id"])
                assert tid not in seen
                seen.add(tid)
                if category not in VEHICLES:
                    continue
                mask = masks[obj["mask_index"]].astype(bool)
                owned = arrays["instance_id"] == tid
                assert not (owned & ~mask).any()
                score = float(np.median(probability[mask])) if probability is not None and mask.any() else None
                decision = state(score)
                updated[owned] = {"unknown": 1, "static": 0, "dynamic": 2}[decision]
                changed_scope |= owned
                frame_records.append(dict(track_id=tid, category=category, score=score, raw_state=decision,
                                          mask_index=obj["mask_index"], owned_pixels=int(owned.sum())))
        assert np.array_equal(updated[~changed_scope], original_state[~changed_scope])
        arrays["state"] = updated
        path = out / f"{f}.npz"
        np.savez_compressed(path, **arrays)
        rows.append(dict(old_obs, path=path.name, sha256=sha(path), evidence_last_frame=i))
        records.write(json.dumps(dict(frame=i, window=list(range(i - 7, i + 1)) if i >= 7 else [],
                                      vehicles=frame_records), allow_nan=False) + "\n")
        if i % 50 == 0:
            print(sequence, i + 1, "/1000", flush=True)
    records.close()
    write(dest / "runtime.json", runtime)
    shutil.copy2(old / "projection_beta.txt", dest / "projection_beta.txt")
    shutil.copy2(old / "calib.txt", dest / "calib.txt")
    write(out / "manifest.json", dict(original, rows=rows, producer=dict(original["producer"],
          vehicle_motion="Pi3MOS eight-frame causal; median>=0.5", weights_sha256=WEIGHT)))
    write(dest / "prediction_complete.json", dict(status="PASS_STATE_ONLY_INTERVENTION", sequence=sequence,
          frames=1000, predicted_windows=993, forward_seconds=forward,
          peak_cuda_bytes=predictor.torch.cuda.max_memory_allocated(), manifest_sha256=sha(out / "manifest.json")))


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "check":
        raise SystemExit(0 if (ROOT / "restored" / (args[1] + ".json")).is_file() else 1)
    {"restore": restore, "parity": sanity, "generate": generate}[action](*args)
