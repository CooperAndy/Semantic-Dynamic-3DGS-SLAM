"""Strict RGB-bound stage2 observation interface. No GT or category->motion rule."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Bridge:
    def __init__(self, manifest, runtime, mode="off"):
        if mode not in ("off", "semantic_only", "dynamic_filter"):
            raise ValueError(mode)
        self.mode = mode
        self.runtime = runtime
        self.cache = {}
        self.rows = {}
        if mode == "off":
            return
        self.path = Path(manifest)
        m = json.loads(self.path.read_text(encoding="utf-8"))
        if m["schema"] != "s23_observations_v1" or m["GT_runtime"] is not False:
            raise ValueError("Prediction sidecars only")
        if mode == "dynamic_filter" and m.get("motion_states_available") is False:
            raise ValueError("Semantic-only observations cannot enable dynamic filtering")
        if m["encoding"] != {"Static": 0, "Uncertain": 1, "Dynamic": 2, "Ignore": 255}:
            raise ValueError("Stage2 state encoding mismatch")
        for key in ("dataset", "sequence", "camera", "class_names", "producer"):
            if not m.get(key):
                raise ValueError("Missing identity: " + key)
        self.identity = m
        if m["camera"] != "P2" or m["dataset"] != "KITTI_Odometry":
            raise ValueError("This adapter targets KITTI P2; other cameras need separate registration")
        self.rows = {r["frame_id"]: r for r in m["rows"]}
        if len(self.rows) != len(m["rows"]) or set(self.rows) != set(range(runtime["frame_count"])):
            raise ValueError("Incomplete or duplicate frame coverage")
        if len(runtime["rows"]) != runtime["frame_count"]:
            raise ValueError("Runtime frame count mismatch")
        for row in runtime["rows"]:
            r = self.rows[row["frame_id"]]
            if r["rgb_sha256"] != row["hashes"]["rgb"] or r["shape"] != [runtime["H"], runtime["W"]]:
                raise ValueError("RGB identity or pixel-grid mismatch")
            if r["source_frame"] != row["source_frame_id"]:
                raise ValueError("Source frame mismatch")
            if r["sequence"] != m["sequence"] or r["camera"] != m["camera"]:
                raise ValueError("Sequence/camera mismatch")
            if r["evidence_last_frame"] > r["source_frame"]:
                raise ValueError("Future motion evidence must be delayed to its availability frame")

    def read(self, frame):
        if self.mode == "off":
            return None
        if frame in self.cache:
            return self.cache[frame]
        r = self.rows[frame]
        p = (self.path.parent / r["path"]).resolve()
        if sha(p) != r["sha256"]:
            raise ValueError("Observation hash mismatch")
        with np.load(p, allow_pickle=False) as z:
            data = {key: z[key].copy() for key in ("category", "instance_id", "confidence", "state")}
        if any(a.shape != tuple(r["shape"]) for a in data.values()):
            raise ValueError("No implicit mask resizing")
        c, ids, conf, state = (data[k] for k in ("category", "instance_id", "confidence", "state"))
        if c.dtype != np.int16 or ids.dtype != np.int32 or conf.dtype != np.float32 or state.dtype != np.uint8:
            raise ValueError("dtype mismatch")
        if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
            raise ValueError("Invalid confidence")
        if np.any((c < -1) | (c >= len(self.identity["class_names"]))) or np.any(ids < -1):
            raise ValueError("Invalid category/instance identity")
        if not np.isin(state, [0, 1, 2, 255]).all():
            raise ValueError("Invalid motion state")
        data["allow"] = state != 2 if self.mode == "dynamic_filter" else np.ones(state.shape, bool)
        # Bounded cache: keyframes may reread their sealed observations.
        if len(self.cache) >= 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[frame] = data
        return data


ACTIVE = None


def allowed(frame):
    import semantic_runtime
    if semantic_runtime.ACTIVE is not None:return semantic_runtime.ACTIVE.allow(frame)
    if ACTIVE is None or ACTIVE.mode != "dynamic_filter":
        return None
    return ACTIVE.read(int(frame))["allow"]


def tensor_mask(frame, like):
    import torch
    mask = allowed(frame)
    return None if mask is None else torch.as_tensor(mask, device=like.device).reshape(like.shape)
