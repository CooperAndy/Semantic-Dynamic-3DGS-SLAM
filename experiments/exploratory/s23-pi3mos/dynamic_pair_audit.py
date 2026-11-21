"""Compare archived 3D–2D proposal inputs with and without D in one cell."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile


CONTROLS = {"D": "B", "TD": "T", "MD": "M", "TMD": "TM"}


def input_pairs(proposal):
    inputs = proposal.get("pre_semantic")
    if inputs is None:
        if proposal["reason"] not in {"insufficient_descriptors", "no_mutual_ratio_matches"}:
            raise ValueError("Missing pre-semantic correspondence ledger")
        return Counter()

    pairs = Counter()
    for reference, image, point in zip(
        inputs["reference_pixels"], inputs["image_points"], inputs["object_points"], strict=True
    ):
        if (len(reference), len(image), len(point)) != (2, 2, 3):
            raise ValueError("Unexpected correspondence dimensions")
        identity = tuple(reference + image + point)
        if not all(math.isfinite(value) for value in identity):
            raise ValueError("Non-finite correspondence")
        pairs[identity] += 1
    return pairs


def archived_proposals(archive_dir, cell_name, group, through):
    label = f"{cell_name}_{group}_full_science"
    receipts = []
    for path in archive_dir.glob("*_local_receipt.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["label"] == label:
            receipts.append(receipt)
    if len(receipts) != 1:
        raise ValueError(f"Expected one verified archive for {label}, found {len(receipts)}")

    receipt = receipts[0]
    identity = receipt["sha256"]
    archive = archive_dir / f"{identity}.tar.gz"
    if (receipt["status"] != "PASS_LOCAL_ARCHIVE_AND_EVERY_MEMBER"
            or receipt["verified_files"] != len(receipt["files"])
            or archive.stat().st_size != receipt["bytes"]):
        raise ValueError(f"Archive receipt or size mismatch for {label}")

    prefix = f"{cell_name}/runs/{group}/proposals/"
    expected = {f"{prefix}{frame:06}.json": frame for frame in range(1, through + 1)}
    locks = {row["path"]: row["sha256"] for row in receipt["files"] if row["path"] in expected}
    if locks.keys() != expected.keys():
        raise ValueError(f"Missing verified proposal members for {label}")
    proposals = {}
    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            if member.name not in expected:
                continue
            frame = expected[member.name]
            if frame in proposals or not member.isfile():
                raise ValueError(f"Invalid proposal member {member.name}")
            with stream.extractfile(member) as source:
                data = source.read()
            if hashlib.sha256(data).hexdigest() != locks[member.name]:
                raise ValueError(f"Changed proposal member {member.name}")
            proposals[frame] = json.loads(data)
    if len(proposals) != through:
        raise ValueError(f"Incomplete proposal data for {label}")
    return proposals, identity


def compare(cell, group, through, archive_dir=None):
    control = CONTROLS[group]
    archive_identities = None
    if archive_dir is None:
        def read(run, frame):
            path = cell / "runs" / run / "proposals" / f"{frame:06}.json"
            return json.loads(path.read_text())
    else:
        control_proposals, control_id = archived_proposals(archive_dir, cell.name, control, through)
        group_proposals, group_id = archived_proposals(archive_dir, cell.name, group, through)
        archive_identities = {control: control_id, group: group_id}
        proposals = {control: control_proposals, group: group_proposals}

        def read(run, frame):
            return proposals[run][frame]

    rows = []
    for frame in range(1, through + 1):
        without_d = read(control, frame)
        with_d = read(group, frame)
        reference = without_d["causal_stereo_reference"]
        if reference != with_d["causal_stereo_reference"] or reference["current_frame"] != frame:
            raise ValueError(f"Different stereo reference identities at frame {frame}")
        before, after = input_pairs(without_d), input_pairs(with_d)
        rows.append({
            "frame": frame,
            "reference_frame": reference["frame"],
            "without_D_pairs": sum(before.values()),
            "with_D_pairs": sum(after.values()),
            "removed_pairs": sum((before - after).values()),
            "added_pairs": sum((after - before).values()),
        })

    totals = {key: sum(row[key] for row in rows) for key in (
        "without_D_pairs", "with_D_pairs", "removed_pairs", "added_pairs"
    )}
    if totals["with_D_pairs"] - totals["without_D_pairs"] != totals["added_pairs"] - totals["removed_pairs"]:
        raise ValueError("Correspondence counts do not reconcile")
    result = {
        "cell": cell.name,
        "group": group,
        "control": control,
        "tracking_steps": through,
        "complete_window": through == 999,
        "stage": "3D–2D proposal inputs before T filtering and PnP",
        "scope": "Exact multiset differences include changes from detection and rematching under the dynamic masks.",
        "changed_steps": sum(bool(row["removed_pairs"] or row["added_pairs"]) for row in rows),
        **totals,
        "rows": rows,
    }
    if archive_identities is not None:
        result["verified_archive_sha256"] = archive_identities
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("group", choices=CONTROLS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--through", type=int, choices=range(1, 1000), metavar="FRAME", default=999)
    parser.add_argument("--archive-dir", type=Path, help="Read exact proposal members from locally verified science archives")
    args = parser.parse_args()
    result = compare(args.cell, args.group, args.through, args.archive_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False))
