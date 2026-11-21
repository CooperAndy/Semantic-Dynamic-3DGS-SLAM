"""Server coordinator: run the per-step budget ladder on the frozen arm C source tree.

No new source tree is built. Every arm uses source_v5_fixB with the freeze and the
minimum coverage guard active; only the per-step translation/rotation budget changes.
The registered 5 mm / 0.05 deg point is the already evaluated arm C and is not rerun.

The clamp count is captured per arm because it is the covariate that explains the
result: an arm whose budget never binds cannot be distinguished from the no-cap arm.

Counterpart protocol:
    00_任务边界与协议/精修单步预算消融_D_20261013.md
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
O = R / 'stage_capablation'

MIN_FREE_BYTES = 4 * 1024 ** 3
RUN_TIMEOUT_S = 1800
WORKER = 'run_stage_d.py'

ARMS = [
    {'label': 'tight_1mm_0.01deg', 'smoke': 'KITTI00_200_D_tight_smoke40_20261013_01',
     'full': 'KITTI00_200_D_tight_20261013_01', 'cap_mode': 'on',
     'cap_translation_mm': 1.0, 'cap_rotation_deg': 0.01},
    {'label': 'loose_20mm_0.2deg', 'smoke': 'KITTI00_200_D_loose_smoke40_20261013_01',
     'full': 'KITTI00_200_D_loose_20261013_01', 'cap_mode': 'on',
     'cap_translation_mm': 20.0, 'cap_rotation_deg': 0.2},
    {'label': 'no_cap_freeze_only', 'smoke': 'KITTI00_200_D_nocap_smoke40_20261013_01',
     'full': 'KITTI00_200_D_nocap_20261013_01', 'cap_mode': 'off',
     'cap_translation_mm': 0.0, 'cap_rotation_deg': 0.0},
]

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def completed_run(run_name, expected_worker_sha):
    manifest_path = R / '04_reproduction' / run_name / 'run_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'COMPLETED':
        return None
    assert manifest.get('worker_sha256') == expected_worker_sha, \
        'existing run %s was produced by a different worker' % run_name
    return manifest


def gate(run_name, frames, arm, log_path):
    directory = R / '04_reproduction' / run_name
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED', (run_name, manifest.get('error'))

    counters = manifest['counters']
    assert counters['input_frames'] == counters['input_frames_submitted'] == frames, counters
    assert counters['gs_pose_optimization_calls'] > 0, 'arm performed no pose updates'
    assert counters['gs_mapping_calls'] > 0 and counters['gs_render_calls'] > 0, counters
    assert manifest['gt_access_audit']['blocked_gt_accesses'] == 0

    history = [json.loads(x) for x in (directory / 'selection_history.jsonl').read_text().splitlines()]
    refined = [h for h in history if 'pixel_support_min' in h]
    assert refined, 'no keyframe refined at all'
    unfrozen = [h for h in refined if h['pixel_support_min'] != h['pixel_support_max']]
    assert not unfrozen, 'frozen support not constant on keyframes %s' % [h['keyframe'] for h in unfrozen]

    gaussians = [json.loads(x) for x in (directory / 'map_range_history.jsonl').read_text().splitlines()]
    assert gaussians, 'map_range_history.jsonl is empty'
    for entry in gaussians:
        assert entry['xyz_finite'] and entry['scale_finite'], entry
        assert entry['max_distance_current_camera_m'] < 1000, entry
        assert entry['max_activated_scale_m'] < 100, entry

    if frames == 40:
        assert manifest['mapped_keyframes'] > 2, manifest['mapped_keyframes']

    clamps = 0
    if log_path.exists():
        clamps = log_path.read_text(errors='replace').count('"event": "step_clamped"')
    fallbacks = [h for h in history if h.get('fallback')]
    return {
        'label': arm['label'],
        'run_name': run_name,
        'frames': frames,
        'cap_mode': arm['cap_mode'],
        'cap_translation_mm': arm['cap_translation_mm'],
        'cap_rotation_deg': arm['cap_rotation_deg'],
        'keyframes': manifest['mapped_keyframes'],
        'gaussians': manifest.get('gaussian_count'),
        'wall_seconds': manifest['wall_seconds'],
        'counter_pose_updates': counters['gs_pose_optimization_calls'],
        'refined_keyframes': len(refined),
        'fallback_keyframes': len(fallbacks),
        'clamp_events': clamps,
        'clamp_rate_of_iterations': (clamps / counters['gs_pose_optimization_calls'])
                                     if counters['gs_pose_optimization_calls'] else None,
    }


def main():
    O.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((O / 'preparation.json').read_text())
    assert preparation['status'] == 'PREPARED_NOT_RUN', preparation['status']
    assert sha(R / 'scripts' / WORKER) == preparation['worker_sha256'], \
        'uploaded worker differs from preparation'
    mapper = R / preparation['source_tree'] / 'dg_slam/gs_tracking_mapping.py'
    assert mapper.exists(), 'source tree missing: %s' % preparation['source_tree']
    assert sha(mapper) == preparation['source_mapper_sha256'], 'source mapper differs from the arm C patch'

    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    free = shutil.disk_usage(R).free
    assert free > MIN_FREE_BYTES, 'free disk below reserve: %.2f GiB' % (free / 1024 ** 3)

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
        'scope': 'CHY KITTI00 0-199 per-step budget ladder; arm C source tree unchanged',
        'preparation_sha256': sha(O / 'preparation.json'),
        'native_receipt_sha256': sha(W / 'receipt.json'),
        'free_bytes_at_start': free,
        'gates': {},
    }
    save = lambda: (O / 'pipeline_receipt.json').write_text(json.dumps(receipt, indent=2))
    save()

    try:
        for arm in ARMS:
            arm_dir = O / arm['label']
            arm_dir.mkdir(parents=True, exist_ok=True)
            for phase, frames, run_name in [('smoke40', 40, arm['smoke']), ('full200', 200, arm['full'])]:
                if phase == 'full200':
                    receipt['gates'][arm['label'] + '_smoke'] = gate(
                        arm['smoke'], 40, arm, arm_dir / 'smoke40.log')
                receipt.update(status='%s_%s_RUNNING' % (arm['label'], phase.upper()), run_name=run_name)
                save()
                prior = completed_run(run_name, preparation['worker_sha256'])
                if prior is not None:
                    receipt['steps'].append('%s_%s_reused_existing_run' % (arm['label'], phase))
                else:
                    with (arm_dir / (phase + '.log')).open('w') as log:
                        completed = subprocess.run(
                            [sys.executable, str(R / 'scripts' / WORKER),
                             '--run-name', run_name, '--frames', str(frames), '--gs-refine', 'on',
                             '--cap-mode', arm['cap_mode'],
                             '--cap-translation-mm', str(arm['cap_translation_mm']),
                             '--cap-rotation-deg', str(arm['cap_rotation_deg'])],
                            cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
                    (arm_dir / (phase + '.exit')).write_text(str(completed.returncode) + '\n')
                    assert completed.returncode == 0, (arm['label'], phase, completed.returncode)
                    receipt['steps'].append('%s_%s_completed' % (arm['label'], phase))
                save()
            receipt['gates'][arm['label'] + '_full'] = gate(arm['full'], 200, arm, arm_dir / 'full200.log')
            save()
        receipt['status'] = 'COMPLETED'
        save()
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
