from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HELDOUT = tuple(range(19, 200, 20))
TRAINING = tuple(index for index in range(200) if index not in HELDOUT)


def nested_farthest_order(candidates: tuple[int, ...]) -> tuple[int, ...]:
    selected = {candidates[0], candidates[-1]}
    order = [candidates[0], candidates[-1]]
    while len(order) < len(candidates):
        remaining = (value for value in candidates if value not in selected)
        best = max(
            remaining,
            key=lambda value: (min(abs(value - chosen) for chosen in selected), -value),
        )
        selected.add(best)
        order.append(best)
    return tuple(order)


def main() -> None:
    output = Path(__file__).resolve().parent.parent / "results"
    output.mkdir(parents=True, exist_ok=True)
    order = nested_farthest_order(TRAINING)
    budgets = {str(size): sorted(order[:size]) for size in (25, 50, 100, 190)}
    for smaller, larger in ((25, 50), (50, 100), (100, 190)):
        if not set(budgets[str(smaller)]).issubset(budgets[str(larger)]):
            raise AssertionError(f"{smaller} is not nested in {larger}")
    if any(set(values).intersection(HELDOUT) for values in budgets.values()):
        raise AssertionError("held-out leakage in mapping budgets")

    manifest = {
        "status": "S2_PROTOCOL_FROZEN_BEFORE_RESULTS",
        "sequence": "KITTI 00 frames 0-199",
        "confidence_gate": 0.60,
        "heldout_frames": list(HELDOUT),
        "selection": "deterministic greedy farthest temporal coverage; tie selects lower frame index",
        "mapping_budgets": budgets,
        "noheldout_mapping_frames": list(range(200)),
        "evaluation_views": list(HELDOUT),
        "comparison_note": "The same ten frame indices are held out for budget curves and included in no-held-out mapping for matched-view in-sample evaluation.",
    }
    json_path = output / "s2_protocol_manifest.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(text, encoding="utf-8")

    with (output / "s2_mapping_frame_indices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["budget", "frame", "is_heldout"])
        for budget, frames in budgets.items():
            for frame in frames:
                writer.writerow([budget, frame, False])

    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (output / "s2_protocol_manifest.sha256").write_text(
        f"{digest}  {json_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(json_path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
