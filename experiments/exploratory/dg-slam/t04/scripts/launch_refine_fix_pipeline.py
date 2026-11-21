"""Server coordinator: deploy the refine-fix source tree, then run the C arm.

Arm C is the single-factor counterpart of B_on: same Stage A+B source tree, same
selector, with only the refinement objective changed. Nothing else varies.

Gate additions over the refine-off pipeline: the arm must actually refine
(gs_pose_optimization_calls > 0) and the frozen support must be observable, that is
every keyframe that refined at least once must report pixel_support_min equal to
pixel_support_max in selection_history.jsonl. If that fails, the freeze did not take
effect and the run must not be scored as arm C.

Counterpart protocol:
    00_任务边界与协议/精修目标函数修复_C_20261013.md
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
O = R / 'stage_refinefix'

MIN_FREE_BYTES = 4 * 1024 ** 3
RUN_TIMEOUT_S = 1800
WORKER = 'run_stage_c.py'
SMOKE_RUN = 'KITTI00_200_C_smoke40_20261013_01'
FULL_RUN = 'KITTI00_200_C_on_20261013_01'

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest(run_name):
    directory = R / '04_reproduction' / run_name
    return directory, json.loads((directory / 'run_manifest.json').read_text())


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


def gate(run_name, frames, preparation):
    directory, manifest = load_manifest(run_name)
    assert manifest['status'] == 'COMPLETED', (run_name, manifest.get('error'))

    counters = manifest['counters']
    assert counters['input_frames'] == counters['input_frames_submitted'] == frames, counters
    assert counters['gs_pose_optimization_calls'] > 0, 'refine-fix arm performed no pose updates'
    assert counters['gs_mapping_calls'] > 0 and counters['gs_render_calls'] > 0, counters
    assert manifest['gt_access_audit']['blocked_gt_accesses'] == 0
    assert manifest['selection_protocol']['refine_objective_fix']['min_frozen_support'] \
        == preparation['thresholds']['min_frozen_support_px']

    history = [json.loads(x) for x in (directory / 'selection_history.jsonl').read_text().splitlines()]
    assert history, 'selection_history.jsonl is empty'
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

    fallbacks = [h for h in history if h.get('fallback')]
    return {
        'run_name': run_name,
        'frames': frames,
        'keyframes': manifest['mapped_keyframes'],
        'gaussians': manifest.get('gaussian_count'),
        'wall_seconds': manifest['wall_seconds'],
        'counters': counters,
        'refined_keyframes': len(refined),
        'fallback_keyframes': len(fallbacks),
        'fallback_reasons': sorted({h.get('reason') for h in fallbacks if h.get('reason')}),
        'frozen_support_min': min(h['pixel_support_min'] for h in refined),
        'frozen_support_max': max(h['pixel_support_max'] for h in refined),
    }


def main():
    O.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((O / 'preparation.json').read_text())
    assert preparation['status'] == 'PREPARED_NOT_RUN', preparation['status']
    assert sha(R / 'scripts' / WORKER) == preparation['worker_sha256'], \
        'uploaded worker differs from preparation'

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
        'reused_runs': [],
        'scope': 'CHY KITTI00 0-199 arm C: Stage A+B source with the refinement objective fixed',
        'preparation_sha256': sha(O / 'preparation.json'),
        'native_receipt_sha256': sha(W / 'receipt.json'),
        'free_bytes_at_start': free,
    }
    save = lambda: (O / 'pipeline_receipt.json').write_text(json.dumps(receipt, indent=2))
    save()

    try:
        target = R / preparation['source_tree']
        if target.exists():
            assert sha(target / 'dg_slam/gs_tracking_mapping.py') \
                == preparation['patched_sha256']['dg_slam/gs_tracking_mapping.py'], \
                'existing source tree does not carry the expected patch'
            receipt['steps'].append('source_tree_already_deployed_and_verified')
            save()
        else:
            with (O / 'deploy.log').open('w') as log:
                deployed = subprocess.run([sys.executable, str(R / 'scripts/deploy_refine_fix.py')],
                                          cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            assert deployed.returncode == 0, ('deploy', deployed.returncode)
            receipt['steps'].append('source_tree_deployed')
            save()

        for phase, frames, run_name in [('smoke40', 40, SMOKE_RUN), ('full200', 200, FULL_RUN)]:
            if phase == 'full200':
                receipt['smoke_gate'] = gate(SMOKE_RUN, 40, preparation)
            receipt.update(status='%s_RUNNING' % phase.upper(), run_name=run_name)
            save()
            prior = completed_run(run_name, preparation['worker_sha256'])
            if prior is not None:
                receipt['reused_runs'].append({'phase': phase, 'run_name': run_name})
                receipt['steps'].append('%s_reused_existing_run' % phase)
            else:
                with (O / (phase + '.log')).open('w') as log:
                    completed = subprocess.run(
                        [sys.executable, str(R / 'scripts' / WORKER),
                         '--run-name', run_name, '--frames', str(frames), '--gs-refine', 'on'],
                        cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S)
                (O / (phase + '.exit')).write_text(str(completed.returncode) + '\n')
                assert completed.returncode == 0, (phase, completed.returncode)
                receipt['steps'].append('%s_completed' % phase)
            save()

        receipt['full_gate'] = gate(FULL_RUN, 200, preparation)
        receipt['status'] = 'COMPLETED'
        save()
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
