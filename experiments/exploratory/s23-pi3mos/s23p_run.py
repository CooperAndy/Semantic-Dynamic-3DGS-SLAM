"""Run the registered Pi³MOS-only vehicle-state replacement in isolation."""
import ast
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from s23p_inputs import ROOT, S23, MODEL, read, sha, write

sys.path.insert(0, str(S23))
from experiment import PYTHON, LIMITS, design
from transfer_endpoint import pack, await_ack
import resource_audit

SEMANTIC = "/root/venvs/semantic_seg/bin/python"
RASTER = S23 / "raster_fp64_accum_v1"
ENV = dict(os.environ, PYTHONPATH=str(RASTER), S23_RASTER_ROOT=str(RASTER),
           PYTORCH_CUDA_ALLOC_CONF="backend:native", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
           MPLBACKEND="Agg", WANDB_MODE="disabled")
INFERENCE = dict(ENV, PYTHONPATH=f"{MODEL}/Pi3MOS-SLAM:{MODEL}/vendor")
PROGRESS = ROOT / "progress.json"
PRODUCERS = ROOT / "producer_results"


def progress(status, **extra):
    state = read(PROGRESS) if PROGRESS.exists() else {}
    state.update(status=status, at=time.time(), **extra)
    write(PROGRESS, state)


def wait_gpu():
    while True:
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True, timeout=20)
        if not raw.strip():
            return
        time.sleep(15)


def launch(argv, log, timeout, env=ENV, cwd=ROOT):
    with log.open("x", encoding="utf-8") as out:
        child = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        start = time.monotonic()
        try:
            while child.poll() is None:
                assert shutil.disk_usage(ROOT).free > LIMITS["minimum_free_GiB"] * 2**30
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Registered run limit {timeout}s: {log.name}")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Exit {child.returncode}: {log.name}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def sample_resources(stop):
    path = ROOT / "resources"
    path.mkdir(exist_ok=True)
    with (path / "GPU_context_samples.jsonl").open("a", buffering=1) as stream:
        while not stop.is_set():
            try:
                s = read(PROGRESS)
                row = resource_audit.sample(S23)
                row.update(status=s["status"], cell=s.get("cell"), group=s.get("group"))
                stream.write(json.dumps(row) + "\n")
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(15)


def prepare(sequence, seed):
    original = S23 / f"E3_K{sequence}_S{seed}_R3_serial"
    cell = S23 / f"S23P_K{sequence}_S{seed}"
    bundle = ROOT / "inputs" / sequence
    assert not cell.exists(), cell
    reg = read(original / "registration.json")
    expected = next(x for x in design()["formal"] if x["sequence"] == sequence and x["seed"] == seed)
    assert reg["cell"] == expected and reg["candidate"] == "candidate_v4_fp64accum"
    for row in read(original / "source_locks.json"):
        assert sha(original / row["path"]) == row["sha256"], row["path"]
    cell.mkdir()
    for name in ("scripts", "source"):
        shutil.copytree(original / name, cell / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("inputs/runtime", "receipts", "runs"):
        (cell / name).mkdir(parents=True, exist_ok=True)
    for name in ("scripts/common.py", "source/configs/data/g20_manifest.yaml"):
        path = cell / name
        path.write_text(path.read_text().replace(str(original), str(cell)))
    shutil.copy2(bundle / "runtime.json", cell / "inputs/runtime/input_manifest.json")
    config_hashes = {}
    for group in ("D", "TMD"):
        rel = f"source/configs/kitti/{group}.py"
        old = original / rel
        assert sha(old) == reg["config_sha256"][group]
        new = old.read_text().replace(str(original), str(cell)).replace(
            str(S23 / "bundles" / sequence / "observations/manifest.json"),
            str(bundle / "observations/manifest.json"))
        cfg = ast.literal_eval(new.split("=", 1)[1].strip())
        assert cfg["semantic"]["manifest"] == str(bundle / "observations/manifest.json")
        assert cfg["workdir"] == str(cell / "runs")
        (cell / rel).write_text(new)
        config_hashes[group] = sha(cell / rel)
    donor = read(original / "receipts/donor.json")
    assert donor["status"] == "PASS_SEALED_DONOR_INPUTS" and len(donor["files"]) == 950
    for row in donor["files"]:
        old = original / row["path"]
        dest = cell / row["path"]
        assert sha(old) == row["sha256"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.hardlink_to(old)
    write(cell / "receipts/donor.json", donor)
    write(cell / "registration.json", dict(reg, variants=["D", "TMD"], config_sha256=config_hashes,
          original_cell=original.name, method="Pi3MOS vehicle state"))
    write(cell / "source_locks.json", [dict(path=p.relative_to(cell).as_posix(), sha256=sha(p))
                                       for p in sorted(cell.rglob("*.py"))])
    return cell


def check_result(cell, group):
    run = cell / "runs" / group
    n = read(run / "receipt.json")
    e = read(run / "evaluation/results.json")
    a = read(run / "independent_numeric_audit.json")
    assert n["status"] == "PASS_NATIVE_MAPPING_INTEGRATION"
    assert n["frames"] == 1000 and n["mapping_backward"] == 57000 and n["GT_blocked_opens"] == 0
    assert n["mapping_observation"]["history_slots"] == 2280
    assert a["status"] == "PASS_INDEPENDENT_BIRTH_POSE_AND_SEMANTIC_IDENTITIES"
    assert e["status"] == "COMPLETE60_FIXED_VIEWS_AND_FULL_TRAJECTORY" and e["GT_prediction"] is False
    assert len(e["rows"]) == 60 and {x["frame"] for x in e["rows"]} == set(design()["evaluation_frames"])
    for line in (run / "mapping_updates.jsonl").open():
        row = json.loads(line)
        assert row["frame"] % 20 != 19 and row["target"] % 20 != 19 and row["target"] <= row["frame"]
    return dict(native=sha(run / "receipt.json"), evaluation=sha(run / "evaluation/results.json"),
                dynamic_consumption=e["tracking_diagnostics"]["dynamic_consumption"])


def archive(files, label, purge=()):
    progress("WAITING_LOCAL_BACKUP", archive_label=label)
    job = pack(files, purge, label=label)
    await_ack(job, timeout=float("inf"))
    return job["sha256"]


def restored(sequence):
    path = ROOT / "restored/bundles" / sequence
    required = (("instances", "*_masks.npz"), ("observations", "[0-9]*.npz"),
                ("cache/depth_confhard", "*.npy"), ("cache/confidence", "*.npy"))
    return (path / "observations/manifest.json").is_file() and all(
        sum(1 for _ in (path / d).glob(g)) == 1000 for d, g in required
    )


def run():
    assert read(ROOT / "parity.json")["status"] == "PASS_THREE_REFERENCE_FRAMES"
    freeze = read(S23 / "formal_method_freeze.json")
    for name, digest in freeze["raster_sha256"].items():
        assert sha(RASTER / name) == digest, name
    assert read(S23 / "long_operator_progress.json")["status"] == "COMPLETE32_RUNS_AND_EVALUATIONS_LOCAL_VERIFIED"
    completed = []
    PRODUCERS.mkdir(exist_ok=True)
    for sequence in ("05", "10"):
        progress("WAITING_RESTORED_INPUTS", sequence=sequence, cell=None, group=None, completed=completed)
        while not restored(sequence):
            time.sleep(15)
        assert shutil.disk_usage(ROOT).free >= LIMITS["startup_free_GiB"] * 2**30
        wait_gpu()
        progress("RUNNING_PI3MOS")
        launch([SEMANTIC, "-u", str(ROOT / "s23p_inputs.py"), "generate", sequence],
               ROOT / f"prediction_{sequence}.log", 7200, env=INFERENCE)
        bundle = ROOT / "inputs" / sequence
        assert read(bundle / "prediction_complete.json")["status"] == "PASS_STATE_ONLY_INTERVENTION"
        # The evaluator runs separately; its GT directory never enters the worker.
        with (ROOT / f"recognition_{sequence}.log").open("x") as stream:
            subprocess.Popen([SEMANTIC, "-u", str(ROOT / "s23p_report.py"), "recognition", sequence],
                             cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        files = [p for p in bundle.rglob("*") if p.is_file()]
        files += [ROOT / n for n in ("parity.json", "s23p_inputs.py", "s23p_run.py", "s23p_report.py",
                                    "dynamic_pair_audit.py", f"prediction_{sequence}.log")]
        archive(files, f"S23P_K{sequence}_predictions")
        for seed in range(3):
            cell = prepare(sequence, seed)
            for group in ("D", "TMD"):
                assert shutil.disk_usage(ROOT).free >= LIMITS["startup_free_GiB"] * 2**30
                wait_gpu()
                progress("RUNNING_NATIVE_GROUP", cell=cell.name, group=group)
                launch([PYTHON, str(cell / "scripts/worker.py"), group], cell / f"{group}.log",
                       LIMITS["run_seconds"], cwd=cell / "scripts")
                from audit_native import audit_run
                audit_run(cell / "runs" / group)
                wait_gpu()
                progress("RUNNING_FROZEN_EVALUATION")
                launch([PYTHON, str(Path(__file__)), "evaluate", str(cell), group],
                       cell / f"{group}_evaluation.log", LIMITS["run_seconds"])
                verdict = check_result(cell, group)
                files = [p for p in (cell / "runs" / group).rglob("*") if p.is_file() and not p.is_symlink()]
                files += [cell / n for n in (f"{group}.log", f"{group}_evaluation.log",
                                             "registration.json", "source_locks.json", "receipts/donor.json")]
                purge = [p for p in files if p.suffix in {".npz", ".npy", ".png"} and p.name not in
                         {"typical_000019.png", "typical_000999.png"}]
                verdict["archive_sha256"] = archive(files, f"{cell.name}_{group}_full_science", purge)
                completed.append(dict(sequence=sequence, seed=seed, group=group, cell=cell.name, **verdict))
                progress("GROUP_COMPLETE_AND_LOCAL_VERIFIED", completed=completed)
    assert len(completed) == 12
    progress("WAITING_RECOGNITION_REPORTS", cell=None, group=None)
    while not all((PRODUCERS / f"{s}.json").is_file() for s in ("05", "10")):
        time.sleep(15)
    launch([SEMANTIC, str(ROOT / "s23p_report.py"), "summary"], ROOT / "summary.log", 1800)
    files = [p for p in (ROOT / "report").rglob("*") if p.is_file()]
    files += [PRODUCERS / f"{s}.json" for s in ("05", "10")]
    identity = archive(files, "S23P_final_comparison")
    progress("COMPLETE_S23P_LOCAL_VERIFIED", final_archive=identity)


if __name__ == "__main__":
    if sys.argv[1:2] == ["evaluate"]:
        original_snapshot = resource_audit.snapshot
        resource_audit.snapshot = lambda remote, cell, group, out: original_snapshot(ROOT, cell, group, out)
        from matrix_evaluation import evaluate
        evaluate(Path(sys.argv[2]), sys.argv[3])
    else:
        with (ROOT / "controller.lock").open("a") as lock, (S23 / "execution.lock").open("a") as gpu_lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not PROGRESS.exists(), "Inspect existing controller before retry"
            progress("STARTING", completed=[])
            stop = threading.Event()
            thread = threading.Thread(target=sample_resources, args=(stop,), daemon=True)
            thread.start()
            try:
                run()
            except Exception as error:
                progress("STOPPED_WITH_ORIGINALS_PRESERVED", error=repr(error), traceback=traceback.format_exc())
                raise
            finally:
                stop.set()
