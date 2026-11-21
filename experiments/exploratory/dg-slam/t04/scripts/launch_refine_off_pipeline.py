"""Server coordinator: smoke gate then independent 200-frame refine-off runs.

Runs A_off (Stage A source) and B_off (Stage A+B source) with --gs-refine off.
Nothing else changes, so each off arm is a single-factor counterpart to the
already completed on arm built from the same source tree.

Refuses to start unless the GPU is idle, free disk exceeds the reserve, and every
uploaded worker and native library matches the hashes recorded during
preparation. Any gate failure aborts the whole pipeline; it never continues to
the next arm, and it never overwrites an existing run directory.

Counterpart protocol:
    00_任务边界与协议/精修开关对照_A_off_B_off_预登记_20261013.md
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

R = Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
W = R / 'stage_a/cuda_sm120'
O = R / 'stage_refoff'

DATE = '20261013'
MIN_FREE_BYTES = 4 * 1024 ** 3
RUN_TIMEOUT_S = 1800

ARMS = [
    {'arm': 'A_off', 'worker': 'run_stage_a.py',
     'smoke': 'KITTI00_200_A_off_smoke40_%s_01' % DATE,
     'full': 'KITTI00_200_A_off_%s_01' % DATE},
    {'arm': 'B_off', 'worker': 'run_stage_b.py',
     'smoke': 'KITTI00_200_B_off_smoke40_%s_01' % DATE,
     'full': 'KITTI00_200_B_off_%s_01' % DATE},
]

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest(run_name):
    directory = R / '04_reproduction' / run_name
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    return directory, manifest


def completed_run(run_name, expected_worker_sha):
    """Return the manifest of an already finished run, or None.

    The worker never writes a commit marker of its own beyond the manifest, so a
    run is only reused when the manifest says COMPLETED and was produced by the
    exact worker hash recorded during preparation.
    """
    manifest_path = R / '04_reproduction' / run_name / 'run_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'COMPLETED':
        return None
    assert manifest.get('worker_sha256') == expected_worker_sha, \
        'existing run %s was produced by a different worker' % run_name
    return manifest


def gate(run_name, frames):
    """Full integrity gate for a refine-off arm. Any failure aborts the pipeline."""
    directory, manifest = load_manifest(run_name)
    assert manifest['status'] == 'COMPLETED', (run_name, manifest.get('error'))

    counters = manifest['counters']
    assert counters['input_frames'] == counters['input_frames_submitted'] == frames, counters
    assert counters['gs_pose_optimization_calls'] == 0, \
        'refine-off arm performed pose updates: %s' % counters['gs_pose_optimization_calls']
    assert counters['gs_mapping_calls'] > 0 and counters['gs_render_calls'] > 0, counters
    assert manifest['gt_access_audit']['blocked_gt_accesses'] == 0, manifest['gt_access_audit']

    history = [json.loads(x) for x in (directory / 'map_range_history.jsonl').read_text().splitlines()]
    assert history, 'map_range_history.jsonl is empty'
    for entry in history:
        assert entry['xyz_finite'] and entry['scale_finite'], entry
        assert entry['max_distance_current_camera_m'] < 1000, entry
        assert entry['max_activated_scale_m'] < 100, entry

    if frames == 40:
        assert manifest['mapped_keyframes'] > 2, manifest['mapped_keyframes']

    return {
        'run_name': run_name,
        'frames': frames,
        'keyframes': manifest['mapped_keyframes'],
        'gaussians': manifest.get('gaussian_count'),
        'wall_seconds': manifest['wall_seconds'],
        'counters': counters,
    }


def main():
    O.mkdir(parents=True, exist_ok=True)
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    free = shutil.disk_usage(R).free
    assert free > MIN_FREE_BYTES, 'free disk below reserve: %.2f GiB' % (free / 1024 ** 3)

    preparation = json.loads((O / 'preparation.json').read_text())
    assert preparation['status'] == 'PREPARED_NOT_RUN', preparation['status']
    for arm, spec in preparation['arms'].items():
        assert sha(R / 'scripts' / spec['worker']) == spec['worker_sha256'], \
            'uploaded worker differs from preparation: %s' % spec['worker']
    assert sha(R / 'scripts' / 'run_stage_a.py') == preparation['arms']['A_off']['worker_sha256']
    assert sha(R / 'scripts' / 'run_stage_b.py') == preparation['arms']['B_off']['worker_sha256']

    native = json.loads((W / 'receipt.json').read_text())
    for name, digest in native['extensions'].items():
        assert sha(W / 'lib' / name) == digest, 'native mismatch: %s' % name

    env = os.environ.copy()
    env.update(LD_PRELOAD=str(W / 'runtime_libs/libstdc++.so.6'),
               LD_LIBRARY_PATH=str(W / 'runtime_libs') + ':' + env.get('LD_LIBRARY_PATH', ''),
               PYTHONPATH=str(W / 'lib') + ':' + str(R / 'scripts'),
               CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='4', WANDB_MODE='disabled',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')

    receipt = {
        'status': 'PREPARED',
        'steps': [],
        'reused_runs': [],
        'scope': 'CHY KITTI00 0-199 refine-off control arms; single flag flip --gs-refine off',
        'preparation_sha256': sha(O / 'preparation.json'),
        'native_receipt_sha256': sha(W / 'receipt.json'),
        'free_bytes_at_start': free,
    }
    save = lambda: (O / 'pipeline_receipt.json').write_text(json.dumps(receipt, indent=2))
    save()

    try:
        for spec in ARMS:
            arm_dir = O / spec['arm']
            arm_dir.mkdir(parents=True, exist_ok=True)
            receipt['current_arm'] = spec['arm']
            for phase, frames, run_name in [('smoke40', 40, spec['smoke']), ('full200', 200, spec['full'])]:
                if phase == 'full200':
                    receipt['%s_smoke_gate' % spec['arm']] = gate(spec['smoke'], 40)
                receipt.update(status='%s_%s_RUNNING' % (spec['arm'], phase.upper()), run_name=run_name)
                save()
                # Reuse an already finished run rather than duplicating GPU work; the
                # worker hash is re-checked so a stale run cannot be silently adopted.
                prior = completed_run(run_name, preparation['arms'][spec['arm']]['worker_sha256'])
                if prior is not None:
                    receipt['reused_runs'].append({'arm': spec['arm'], 'phase': phase, 'run_name': run_name})
                    receipt['steps'].append('%s_%s_reused_existing_run' % (spec['arm'], phase))
                else:
                    with (arm_dir / (phase + '.log')).open('w') as log:
                        completed = subprocess.run(
                            [sys.executable, str(R / 'scripts' / spec['worker']),
                             '--run-name', run_name, '--frames', str(frames), '--gs-refine', 'off'],
                            cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
                    (arm_dir / (phase + '.exit')).write_text(str(completed.returncode) + '\n')
                    assert completed.returncode == 0, (spec['arm'], phase, completed.returncode)
                    receipt['steps'].append('%s_%s_completed' % (spec['arm'], phase))
                save()
            receipt['%s_full_gate' % spec['arm']] = gate(spec['full'], 200)
            save()
        receipt['status'] = 'COMPLETED'
        receipt.pop('current_arm', None)
        save()
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
