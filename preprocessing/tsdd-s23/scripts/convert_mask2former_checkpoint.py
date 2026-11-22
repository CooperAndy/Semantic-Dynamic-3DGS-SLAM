#!/usr/bin/env python3
"""Convert the legacy HF Mask2Former Swin checkpoint key layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForUniversalSegmentation


OLD_ENCODER_PREFIX = "model.pixel_level_module.encoder.model."
NEW_ENCODER_PREFIX = "model.pixel_level_module.encoder."
LEGACY_STAGE_PREFIX = "model.pixel_level_module.encoder.hidden_states_norms."
OBSOLETE_KEYS = {
    "model.pixel_level_module.encoder.layernorm.weight",
    "model.pixel_level_module.encoder.layernorm.bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_key(key: str) -> str | None:
    if key.startswith(OLD_ENCODER_PREFIX):
        key = key.replace(OLD_ENCODER_PREFIX, NEW_ENCODER_PREFIX, 1)
    if key.startswith(LEGACY_STAGE_PREFIX):
        suffix = key[len(LEGACY_STAGE_PREFIX) :]
        index, separator, parameter = suffix.partition(".")
        if separator and index.isdigit():
            key = f"{LEGACY_STAGE_PREFIX}stage{int(index) + 1}.{parameter}"
    if key in OBSOLETE_KEYS:
        return None
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("destination_dir", type=Path)
    args = parser.parse_args()

    source_weight = args.source_dir / "pytorch_model.bin"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    if args.destination_dir.exists() and any(args.destination_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.destination_dir}")
    args.destination_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(source_weight, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    remapped = {}
    dropped = []
    changed = []
    for old_key, value in checkpoint.items():
        new_key = remap_key(old_key)
        if new_key is None:
            dropped.append(old_key)
            continue
        if new_key in remapped:
            raise RuntimeError(f"Key collision after remap: {new_key}")
        remapped[new_key] = value
        if new_key != old_key:
            changed.append([old_key, new_key])

    config = AutoConfig.from_pretrained(args.source_dir, local_files_only=True)
    model = AutoModelForUniversalSegmentation.from_config(config)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(remapped))
    unexpected = sorted(set(remapped) - set(expected))
    shape_mismatches = {
        key: [list(remapped[key].shape), list(expected[key].shape)]
        for key in sorted(set(remapped) & set(expected))
        if remapped[key].shape != expected[key].shape
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            json.dumps(
                {
                    "missing": missing,
                    "unexpected": unexpected,
                    "shape_mismatches": shape_mismatches,
                },
                indent=2,
            )
        )

    for filename in ("config.json", "preprocessor_config.json", "README.md"):
        source_file = args.source_dir / filename
        if source_file.is_file():
            shutil.copy2(source_file, args.destination_dir / filename)
    destination_weight = args.destination_dir / "pytorch_model.bin"
    torch.save(remapped, destination_weight)
    manifest = {
        "status": "CONVERTED_AND_KEYSET_VALIDATED",
        "source_dir": str(args.source_dir),
        "source_weight": str(source_weight),
        "source_sha256": sha256_file(source_weight),
        "destination_weight": str(destination_weight),
        "destination_sha256": sha256_file(destination_weight),
        "source_keys": len(checkpoint),
        "destination_keys": len(remapped),
        "expected_keys": len(expected),
        "renamed_keys": len(changed),
        "dropped_obsolete_keys": dropped,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
    }
    (args.destination_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
