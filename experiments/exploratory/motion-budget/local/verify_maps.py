"""Verify downloaded map snapshots against the GPU replay receipts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for tag, directory, receipt_path in (
    ("holdout", HERE / "holdout_maps_G00_S0", HERE / "replay_v1/receipt.json"),
    ("grid", HERE / "grid_maps_G00_S0", HERE / "grid_replay_v1/receipt.json"),
):
    if len(sys.argv) > 1 and tag not in sys.argv[1:]:
        continue
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETED":
        raise ValueError(f"{tag}: replay incomplete")
    checked = 0
    for index, expected in receipt["map_sha256"].items():
        path = directory / f"params{index}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"{tag}: hash mismatch {path}")
        checked += 1
    print(f"{tag}: {checked}/{len(receipt['map_sha256'])} map snapshots verified")
