"""Shared paths and file identities for the portable S23 commands."""
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
WINDOW_START = {"05": 0, "08": 3000, "10": 0}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
