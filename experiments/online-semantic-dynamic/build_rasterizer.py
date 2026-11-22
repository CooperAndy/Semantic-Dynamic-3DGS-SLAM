"""Build the S23 FP64 renderer with the verified headers from its upstream base."""
import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from reproduction_common import STUDY, read, sha256

UPSTREAM = "https://codeload.github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/zip/"
COMMIT = "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"


def stage_headers(raster, archive_path):
    lock = read(STUDY / "native-source-lock.json")["sources"]
    with ZipFile(archive_path) as archive:
        prefix = archive.namelist()[0].split("/", 1)[0]
        for name, expected in lock.items():
            destination = raster / name
            if name.startswith("third_party/glm/") and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(f"{prefix}/{name}") as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
            if sha256(destination) != expected:
                raise ValueError(f"Source differs from the historical build: {destination}")
    return len(lock)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-archive", type=Path, help="Local ZIP of the pinned upstream commit")
    p.add_argument("--source-only", action="store_true", help="Stage and verify sources without invoking CUDA")
    args = p.parse_args()
    raster = STUDY / "native"
    if args.upstream_archive is not None:
        count = stage_headers(raster, args.upstream_archive)
    else:
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "upstream.zip"
            with urllib.request.urlopen(UPSTREAM + COMMIT, timeout=60) as src, archive.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count = stage_headers(raster, archive)
    print(f"Verified {count} FP64 rasterizer source files", flush=True)
    if not args.source_only:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=raster, check=True)
        subprocess.run([sys.executable, "-c", "import torch; from diff_gaussian_rasterization import _C"],
                       cwd=raster, check=True)


if __name__ == "__main__":
    main()
