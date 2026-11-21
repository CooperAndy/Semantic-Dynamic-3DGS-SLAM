#!/usr/bin/env python
import argparse
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


READ_SIZE = 1024 * 1024


def human_size(n):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0


def request(url, method="GET", headers=None, timeout=60):
    req_headers = {
        "User-Agent": "range-downloader/1.0",
        "Accept-Encoding": "identity",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def probe(url):
    size = None
    final_url = url
    try:
        with request(url, method="HEAD") as resp:
            final_url = resp.geturl()
            length = resp.headers.get("Content-Length")
            if length:
                size = int(length)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pass

    range_headers = {"Range": "bytes=0-0"}
    with request(final_url, headers=range_headers) as resp:
        status = getattr(resp, "status", resp.getcode())
        content_range = resp.headers.get("Content-Range")
        if status != 206 or not content_range:
            length = resp.headers.get("Content-Length")
            return final_url, int(length) if length else size, False
        if size is None:
            size = int(content_range.rsplit("/", 1)[1])
        return resp.geturl(), size, True


def download_range(url, part_path, start, end, retries):
    expected = end - start + 1
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing == expected:
        return expected
    if existing > expected:
        existing = 0

    offset = start + existing
    mode = "ab" if existing else "wb"
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={offset}-{end}"}
            with request(url, headers=headers, timeout=120) as resp:
                status = getattr(resp, "status", resp.getcode())
                if status != 206:
                    raise RuntimeError(f"server returned HTTP {status} for range request")
                with part_path.open(mode) as fh:
                    while True:
                        block = resp.read(READ_SIZE)
                        if not block:
                            break
                        fh.write(block)

            actual = part_path.stat().st_size
            if actual == expected:
                return actual
            existing = actual if actual < expected else 0
            offset = start + existing
            mode = "ab" if existing else "wb"
            raise RuntimeError(f"incomplete part {part_path.name}: {actual}/{expected}")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    return part_path.stat().st_size


def single_download(url, output):
    tmp = output.with_suffix(output.suffix + ".tmp")
    with request(url, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            block = resp.read(READ_SIZE)
            if not block:
                break
            fh.write(block)
    os.replace(tmp, output)


def merge_parts(parts, output):
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                while True:
                    block = fh.read(READ_SIZE)
                    if not block:
                        break
                    out.write(block)
    os.replace(tmp, output)


def main():
    parser = argparse.ArgumentParser(description="Download a file with HTTP range workers.")
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Probing {args.url}", flush=True)
    final_url, size, supports_range = probe(args.url)
    if not size:
        raise SystemExit("Could not determine remote file size.")

    if output.exists() and output.stat().st_size == size:
        print(f"Already complete: {output} ({human_size(size)})", flush=True)
        return 0

    print(f"Remote size: {human_size(size)}", flush=True)
    if not supports_range:
        print("Server did not advertise range support; falling back to one stream.", flush=True)
        single_download(final_url, output)
        return 0

    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mib * 1024 * 1024
    chunk_count = math.ceil(size / chunk_size)
    ranges = []
    for i in range(chunk_count):
        start = i * chunk_size
        end = min(start + chunk_size - 1, size - 1)
        ranges.append((part_dir / f"part_{i:05d}", start, end))

    workers = max(1, min(args.workers, len(ranges)))
    print(f"Downloading {chunk_count} parts with {workers} workers", flush=True)
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_range, final_url, part, start, end, args.retries)
            for part, start, end in ranges
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            downloaded = sum(part.stat().st_size for part, _, _ in ranges if part.exists())
            elapsed = max(time.time() - started, 0.001)
            rate = downloaded / elapsed
            pct = downloaded * 100.0 / size
            print(
                f"{completed}/{chunk_count} parts, {pct:5.1f}%, "
                f"{human_size(downloaded)}/{human_size(size)}, {human_size(rate)}/s",
                flush=True,
            )

    print("Merging parts", flush=True)
    merge_parts([part for part, _, _ in ranges], output)
    actual = output.stat().st_size
    if actual != size:
        raise SystemExit(f"Merged file size mismatch: {actual} != {size}")
    print(f"Done: {output} ({human_size(actual)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
