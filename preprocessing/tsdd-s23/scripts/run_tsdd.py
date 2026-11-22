#!/usr/bin/env python3
"""Run the offline motion-only TSDD log-odds state machine.

The script deliberately separates an engineering smoke run from a formal run.
A formal run is rejected unless both the motion-evidence package and the TSDD
configuration explicitly declare that Gate B has passed and is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


STATE_NAMES = {0: "Static", 1: "Uncertain", 2: "Dynamic"}
STATE_COLORS = {0: (60, 180, 75), 1: (0, 210, 255), 2: (40, 40, 230)}
IGNORE_STATE = 255
FORMAL_EVIDENCE_STATUS = "GATE_B_PASS_FROZEN"
FORMAL_CONFIG_STATUS = "frozen_after_gate_b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class TSDDConfig:
    forgetting_factor: float
    theta_low: float
    theta_high: float
    ell_clip: float
    probability_min: float
    probability_max: float
    min_valid_ratio: float
    min_confirm_frames: int
    dynamic_to_static_confirm_frames: int
    max_missing_frames: int
    calibration_status: str

    @classmethod
    def from_yaml(cls, path: Path) -> "TSDDConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        log_odds = raw["log_odds"]
        gating = raw["gating"]
        probability_min, probability_max = log_odds["probability_clip"]
        config = cls(
            forgetting_factor=float(log_odds["forgetting_factor"]),
            theta_low=float(log_odds["theta_low"]),
            theta_high=float(log_odds["theta_high"]),
            ell_clip=float(log_odds["clip"]),
            probability_min=float(probability_min),
            probability_max=float(probability_max),
            min_valid_ratio=float(gating["min_valid_ratio"]),
            min_confirm_frames=int(gating["min_confirm_frames"]),
            dynamic_to_static_confirm_frames=int(
                gating["dynamic_to_static_confirm_frames"]
            ),
            max_missing_frames=int(gating["max_missing_frames"]),
            calibration_status=str(raw.get("calibration_status", "missing")),
        )
        if not 0 < config.forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if not config.theta_low < 0 < config.theta_high:
            raise ValueError("Expected theta_low < 0 < theta_high")
        if not 0 < config.probability_min < config.probability_max < 1:
            raise ValueError("probability_clip must lie inside (0, 1)")
        if not 0 <= config.min_valid_ratio <= 1:
            raise ValueError("min_valid_ratio must be in [0, 1]")
        if min(
            config.min_confirm_frames,
            config.dynamic_to_static_confirm_frames,
            config.max_missing_frames,
        ) < 1:
            raise ValueError("Confirmation and missing-frame counts must be positive")
        return config


@dataclass
class TrackMemory:
    track_id: int
    state: int = 1
    log_odds: float = 0.0
    high_count: int = 0
    low_count: int = 0
    missing_count: int = 0
    observations: int = 0
    valid_observations: int = 0
    first_frame: int = -1
    last_frame: int = -1
    state_changes: int = 0
    uncertain_since_frame: int = -1
    uncertain_origin_state: int = 1


class TSDDStateMachine:
    def __init__(self, config: TSDDConfig) -> None:
        self.config = config
        self.tracks: dict[int, TrackMemory] = {}
        self.transitions: list[dict] = []

    def get_or_create(self, track_id: int, frame_index: int) -> TrackMemory:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackMemory(
                track_id=track_id,
                first_frame=frame_index,
                last_frame=frame_index,
                uncertain_since_frame=frame_index,
            )
        return self.tracks[track_id]

    def mark_missing(self, track: TrackMemory, frame_index: int) -> None:
        """Decay stale belief without injecting artificial static evidence."""
        track.log_odds *= self.config.forgetting_factor
        track.missing_count += 1
        track.high_count = 0
        track.low_count = 0
        if track.missing_count > self.config.max_missing_frames:
            self._transition(track, 1, frame_index, "missing_timeout")

    def update(
        self,
        track_id: int,
        frame_index: int,
        probability: float | None,
        valid_ratio: float,
    ) -> TrackMemory:
        track = self.get_or_create(track_id, frame_index)
        track.observations += 1
        track.last_frame = frame_index
        if probability is None or valid_ratio < self.config.min_valid_ratio:
            self.mark_missing(track, frame_index)
            return track

        probability = float(
            np.clip(
                probability, self.config.probability_min, self.config.probability_max
            )
        )
        evidence = math.log(probability / (1.0 - probability))
        track.log_odds = float(
            np.clip(
                self.config.forgetting_factor * track.log_odds + evidence,
                -self.config.ell_clip,
                self.config.ell_clip,
            )
        )
        track.valid_observations += 1
        track.missing_count = 0
        if track.log_odds >= self.config.theta_high:
            track.high_count += 1
            track.low_count = 0
        elif track.log_odds <= self.config.theta_low:
            track.low_count += 1
            track.high_count = 0
        else:
            track.high_count = 0
            track.low_count = 0
            # The neutral band is explicitly Uncertain and provides the
            # conservative fallback required for conflicting evidence.
            self._transition(track, 1, frame_index, "neutral_evidence_band")

        if track.state in (0, 1) and track.high_count >= self.config.min_confirm_frames:
            self._transition(track, 2, frame_index, "sustained_high_evidence")
        elif track.state == 1:
            low_required = (
                self.config.dynamic_to_static_confirm_frames
                if track.uncertain_origin_state == 2
                else self.config.min_confirm_frames
            )
            if track.low_count >= low_required:
                reason = (
                    "sustained_low_after_dynamic"
                    if track.uncertain_origin_state == 2
                    else "sustained_low_evidence"
                )
                self._transition(track, 0, frame_index, reason)
        elif (
            track.state == 2
            and track.low_count >= self.config.dynamic_to_static_confirm_frames
        ):
            self._transition(track, 0, frame_index, "sustained_low_after_dynamic")
        return track

    def _transition(
        self, track: TrackMemory, new_state: int, frame_index: int, reason: str
    ) -> None:
        if track.state == new_state:
            return
        old_state = track.state
        confirmation_latency = None
        if old_state == 1 and new_state in (0, 2):
            confirmation_latency = frame_index - track.uncertain_since_frame + 1
        track.state = new_state
        if new_state == 1:
            track.uncertain_since_frame = frame_index
            track.uncertain_origin_state = old_state
        else:
            track.uncertain_origin_state = new_state
        track.state_changes += 1
        track.high_count = 0
        track.low_count = 0
        self.transitions.append(
            {
                "frame": frame_index,
                "track_id": track.track_id,
                "from": STATE_NAMES[old_state],
                "to": STATE_NAMES[new_state],
                "reason": reason,
                "log_odds": track.log_odds,
                "confirmation_latency_frames": confirmation_latency,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=199, help="Inclusive final frame")
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument(
        "--run-status",
        choices=("ENGINEERING_SMOKE", "FORMAL"),
        default="ENGINEERING_SMOKE",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_instances(instance_dir: Path, frame_id: str) -> tuple[dict, np.ndarray]:
    metadata = json.loads((instance_dir / f"{frame_id}.json").read_text(encoding="utf-8"))
    masks = np.load(
        instance_dir / f"{frame_id}_masks.npz", allow_pickle=False
    )["masks"]
    return metadata, masks


def make_preview(image: np.ndarray, masks: np.ndarray, instances: list[dict]) -> np.ndarray:
    overlay = image.copy()
    for state in (0, 1, 2):
        selected = [index for index, item in enumerate(instances) if item["state"] == state]
        if not selected:
            continue
        union = masks[selected].any(axis=0)
        color = np.asarray(STATE_COLORS[state], dtype=np.float32)
        overlay[union] = (0.55 * overlay[union] + 0.45 * color).astype(np.uint8)
    for item in instances:
        x1, y1, x2, y2 = np.rint(item["bbox"]).astype(int)
        color = STATE_COLORS[item["state"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        if (x2 - x1) * (y2 - y1) >= 1200:
            label = f"T{item['track_id']} {STATE_NAMES[item['state']][0]}"
            cv2.putText(
                overlay,
                label,
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
    return overlay


def write_transitions(output_dir: Path, transitions: list[dict]) -> None:
    (output_dir / "transitions.json").write_text(
        json.dumps(transitions, indent=2), encoding="utf-8"
    )
    fields = [
        "frame",
        "track_id",
        "from",
        "to",
        "reason",
        "log_odds",
        "confirmation_latency_frames",
    ]
    with (output_dir / "transitions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transitions)


def validate_formal_guard(
    run_status: str, config: TSDDConfig, evidence_summary: dict
) -> None:
    if run_status != "FORMAL":
        return
    evidence_status = str(evidence_summary.get("status", "missing"))
    if evidence_status != FORMAL_EVIDENCE_STATUS:
        raise RuntimeError(
            "FORMAL run rejected: evidence status is "
            f"{evidence_status!r}, expected {FORMAL_EVIDENCE_STATUS!r}"
        )
    if config.calibration_status != FORMAL_CONFIG_STATUS:
        raise RuntimeError(
            "FORMAL run rejected: config calibration_status is "
            f"{config.calibration_status!r}, expected {FORMAL_CONFIG_STATUS!r}"
        )


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be >= --start")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    config = TSDDConfig.from_yaml(args.config)
    evidence_summary_path = args.evidence_dir / "generation_summary.json"
    if not evidence_summary_path.is_file():
        raise FileNotFoundError(f"Missing evidence summary: {evidence_summary_path}")
    evidence_summary = json.loads(evidence_summary_path.read_text(encoding="utf-8"))
    validate_formal_guard(args.run_status, config, evidence_summary)

    machine = TSDDStateMachine(config)
    state_counts = {name: 0 for name in STATE_NAMES.values()}
    valid_evidence_count = 0
    total_instances = 0
    frame_summaries: list[dict] = []
    track_trajectories: dict[int, list[dict]] = {}

    for frame_index in range(args.start, args.end + 1):
        frame_id = f"{frame_index:06d}"
        metadata, masks = load_instances(args.instance_dir, frame_id)
        input_instances = metadata["instances"]
        active_track_ids = {int(item["track_id"]) for item in input_instances}
        for track_id, track in list(machine.tracks.items()):
            if track_id not in active_track_ids:
                machine.mark_missing(track, frame_index)

        evidence_by_instance: dict[int, dict] = {}
        evidence_path = args.evidence_dir / f"{frame_id}_{frame_index + 1:06d}.json"
        if evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if str(evidence.get("frame_t")) != frame_id:
                raise RuntimeError(f"Evidence frame mismatch at {evidence_path}")
            evidence_by_instance = {
                int(item["instance_id"]): item for item in evidence["instances"]
            }

        output_instances: list[dict] = []
        states: list[int] = []
        for item in input_instances:
            instance_id = int(item["instance_id"])
            track_id = int(item["track_id"])
            observation = evidence_by_instance.get(instance_id)
            if observation is None:
                probability = None
                valid_ratio = 0.0
            else:
                if int(observation["track_id"]) != track_id:
                    raise RuntimeError(f"Track mismatch in evidence for {frame_id}/{instance_id}")
                probability = observation.get("dynamic_probability")
                valid_ratio = float(observation["valid_ratio"])
            track = machine.update(track_id, frame_index, probability, valid_ratio)
            valid_evidence = bool(
                probability is not None and valid_ratio >= config.min_valid_ratio
            )
            if valid_evidence:
                valid_evidence_count += 1
            states.append(track.state)
            state_counts[STATE_NAMES[track.state]] += 1
            total_instances += 1
            output = {
                "instance_id": instance_id,
                "track_id": track_id,
                "category": item["category"],
                "bbox": item["bbox"],
                "state": track.state,
                "state_name": STATE_NAMES[track.state],
                "log_odds": track.log_odds,
                "dynamic_probability": probability,
                "valid_ratio": valid_ratio,
                "valid_evidence": valid_evidence,
                "track_age_observations": track.observations,
                "missing_count": track.missing_count,
            }
            output_instances.append(output)
            track_trajectories.setdefault(track_id, []).append(
                {
                    "frame": frame_index,
                    "instance_id": instance_id,
                    "state": track.state,
                    "state_name": STATE_NAMES[track.state],
                    "log_odds": track.log_odds,
                    "dynamic_probability": probability,
                    "valid_ratio": valid_ratio,
                    "valid_evidence": valid_evidence,
                }
            )

        states_array = np.asarray(states, dtype=np.uint8)
        shape = masks.shape[1:]
        unions = {}
        for state in (0, 1, 2):
            selected = states_array == state
            unions[state] = (
                masks[selected].any(axis=0) if selected.any() else np.zeros(shape, dtype=np.bool_)
            )
        dynamic_mask = unions[2]
        uncertain_mask = unions[1] & ~dynamic_mask
        static_mask = unions[0] & ~dynamic_mask & ~uncertain_mask
        state_map = np.full(shape, IGNORE_STATE, dtype=np.uint8)
        state_map[static_mask] = 0
        state_map[uncertain_mask] = 1
        state_map[dynamic_mask] = 2
        np.savez_compressed(
            args.output_dir / f"{frame_id}_states.npz",
            states=states_array,
            state_map=state_map,
            static_mask=static_mask.astype(np.bool_),
            uncertain_mask=uncertain_mask.astype(np.bool_),
            dynamic_mask=dynamic_mask.astype(np.bool_),
        )
        frame_summary = {
            "schema_version": 2,
            "frame_id": frame_id,
            "frame_index": frame_index,
            "instances": output_instances,
            "state_counts": {
                name: sum(item["state_name"] == name for item in output_instances)
                for name in STATE_NAMES.values()
            },
        }
        (args.output_dir / f"{frame_id}.json").write_text(
            json.dumps(frame_summary, indent=2), encoding="utf-8"
        )
        frame_summaries.append(frame_summary)
        if args.preview_every > 0 and (frame_index - args.start) % args.preview_every == 0:
            image = cv2.imread(str(args.image_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode image {frame_id}")
            cv2.imwrite(
                str(preview_dir / f"{frame_id}.png"),
                make_preview(image, masks, output_instances),
            )

    track_summaries = [asdict(track) for track in machine.tracks.values()]
    confirmation_latencies = [
        event["confirmation_latency_frames"]
        for event in machine.transitions
        if event["confirmation_latency_frames"] is not None
    ]
    transition_denominator = max(total_instances - len(track_summaries), 1)
    instance_summary_path = args.instance_dir / "generation_summary.json"
    evidence_manifest_path = args.evidence_dir / "SHA256SUMS.txt"
    summary = {
        "schema_version": 2,
        "status": args.run_status,
        "scientific_status": (
            "PARTIAL_WAIT_GATE_B_AND_SYNTHETIC_GT"
            if args.run_status == "ENGINEERING_SMOKE"
            else "FORMAL_INPUTS_ACCEPTED_PENDING_T2_3_GT_EVALUATION"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames": len(frame_summaries),
        "frame_range_inclusive": [args.start, args.end],
        "total_instance_observations": total_instances,
        "valid_evidence_observations": valid_evidence_count,
        "valid_evidence_ratio": valid_evidence_count / max(total_instances, 1),
        "state_counts": state_counts,
        "uncertain_rate": state_counts["Uncertain"] / max(total_instances, 1),
        "tracks": len(track_summaries),
        "transition_count": len(machine.transitions),
        "state_flip_rate_per_track_step": len(machine.transitions) / transition_denominator,
        "confirmation_latency_frames": {
            "count": len(confirmation_latencies),
            "mean": float(np.mean(confirmation_latencies)) if confirmation_latencies else None,
            "p95": float(np.percentile(confirmation_latencies, 95)) if confirmation_latencies else None,
        },
        "transitions": machine.transitions,
        "track_summaries": track_summaries,
        "config": asdict(config),
        "inputs": {
            "image_dir": str(args.image_dir.resolve()),
            "instance_dir": str(args.instance_dir.resolve()),
            "evidence_dir": str(args.evidence_dir.resolve()),
            "config": file_fingerprint(args.config),
            "instance_generation_summary": file_fingerprint(instance_summary_path),
            "evidence_generation_summary": file_fingerprint(evidence_summary_path),
            "evidence_sha256_manifest": file_fingerprint(evidence_manifest_path),
            "evidence_declared_status": evidence_summary.get("status"),
        },
        "output_contract": {
            "state_encoding": {**{str(k): v for k, v in STATE_NAMES.items()}, "255": "Ignore"},
            "raster_priority": ["Dynamic", "Uncertain", "Static"],
            "background_value": IGNORE_STATE,
        },
    }
    write_transitions(args.output_dir, machine.transitions)
    (args.output_dir / "track_trajectories.json").write_text(
        json.dumps({str(k): v for k, v in track_trajectories.items()}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "status", "scientific_status", "frames", "total_instance_observations",
        "valid_evidence_ratio", "state_counts", "uncertain_rate", "tracks",
        "transition_count", "state_flip_rate_per_track_step", "confirmation_latency_frames"
    )}, indent=2))


if __name__ == "__main__":
    main()
