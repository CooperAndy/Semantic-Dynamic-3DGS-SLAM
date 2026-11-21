"""Build the step-budget ablation worker (arm D) and its preparation receipt.

Arm C showed that the registered per-step budget binds on 75.2 percent of refinement
iterations, so the budget itself is now the dominant mechanism and its value is
unvalidated. This ablation varies only that budget, on the same source tree.

Local only: no GPU, no network.

Counterpart protocol:
    00_任务边界与协议/精修单步预算消融_D_20261013.md
"""
from pathlib import Path
import ast
import hashlib
import json

R = Path(__file__).resolve().parents[1]
P = R / 'scripts'
OUT = R / '02_Gaussian适配审计' / '精修单步预算消融_20261013'

SOURCE_TREE = 'source_v5_fixB'
SOURCE_MAPPER_SHA = '14abd881715a72996471f45fe55bfc104a18e0fff44d63c7c6ff40cee7d9b2e7'
DATE = '20261013'

# The registered arm C (5 mm / 0.05 deg) is already evaluated and is reused as the
# middle point; only the three new points below are run.
REGISTERED_ARM = {
    'label': 'registered_5mm_0.05deg',
    'run_name': 'KITTI00_200_C_on_%s_01' % DATE,
    'cap_translation_mm': 5.0,
    'cap_rotation_deg': 0.05,
    'status': 'ALREADY_EVALUATED_REUSE',
}

NEW_ARMS = [
    {'label': 'tight_1mm_0.01deg', 'run': 'KITTI00_200_D_tight_%s_01' % DATE,
     'smoke': 'KITTI00_200_D_tight_smoke40_%s_01' % DATE,
     'cap_translation_mm': 1.0, 'cap_rotation_deg': 0.01, 'cap_mode': 'on'},
    {'label': 'loose_20mm_0.2deg', 'run': 'KITTI00_200_D_loose_%s_01' % DATE,
     'smoke': 'KITTI00_200_D_loose_smoke40_%s_01' % DATE,
     'cap_translation_mm': 20.0, 'cap_rotation_deg': 0.2, 'cap_mode': 'on'},
    {'label': 'no_cap_freeze_only', 'run': 'KITTI00_200_D_nocap_%s_01' % DATE,
     'smoke': 'KITTI00_200_D_nocap_smoke40_%s_01' % DATE,
     'cap_translation_mm': 0.0, 'cap_rotation_deg': 0.0, 'cap_mode': 'off'},
]

WORKER_EDITS = [
    ('cli_args',
     "    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);"
     "p.add_argument('--gs-refine',choices=['on','off'],required=True);"
     "p.add_argument('--frames', type=int, choices=[40,200], required=True);a=p.parse_args()\n",
     "    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);"
     "p.add_argument('--gs-refine',choices=['on','off'],required=True);"
     "p.add_argument('--frames', type=int, choices=[40,200], required=True);"
     "p.add_argument('--cap-mode',choices=['on','off'],required=True);"
     "p.add_argument('--cap-translation-mm',type=float,required=True);"
     "p.add_argument('--cap-rotation-deg',type=float,required=True);a=p.parse_args()\n"),
    ('cfg_budget',
     "    cfg.update(t04_freeze_support=True, t04_min_frozen_support=128, t04_step_cap=True,\n"
     "               t04_step_cap_translation_m=0.005, t04_step_cap_rotation_deg=0.05)\n",
     "    cfg.update(t04_freeze_support=True, t04_min_frozen_support=128,\n"
     "               t04_step_cap=(a.cap_mode=='on'),\n"
     "               t04_step_cap_translation_m=a.cap_translation_mm/1000.0,\n"
     "               t04_step_cap_rotation_deg=a.cap_rotation_deg)\n"),
]

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt_path = OUT / 'preparation.json'
    if receipt_path.exists():
        raise SystemExit('Refusing to overwrite existing preparation receipt: %s' % receipt_path)

    base = (P / 'run_stage_c.py').read_text(encoding='utf-8')
    for name, old, new in WORKER_EDITS:
        count = base.count(old)
        assert count == 1, 'anchor %s matched %d times' % (name, count)
        base = base.replace(old, new)
    ast.parse(base)
    (P / 'run_stage_d.py').write_text(base, encoding='utf-8')

    receipt = {
        'status': 'PREPARED_NOT_RUN',
        'date_client': '2026-10-13',
        'scope': 'CHY KITTI00 0-199 step-budget ablation; arm C source tree unchanged',
        'design': 'only the per-step translation/rotation budget varies; freeze, minimum coverage guard, selector and source tree stay identical',
        'source_tree': SOURCE_TREE,
        'source_mapper_sha256': SOURCE_MAPPER_SHA,
        'worker_sha256': sha(P / 'run_stage_d.py'),
        'ladder': [REGISTERED_ARM] + [
            {'label': a['label'], 'run_name': a['run'], 'smoke_run_name': a['smoke'],
             'cap_translation_mm': a['cap_translation_mm'], 'cap_rotation_deg': a['cap_rotation_deg'],
             'cap_mode': a['cap_mode'], 'status': 'THIS_PREPARATION'}
            for a in NEW_ARMS],
        'why': 'arm C clamped 1954 of 2600 refinement iterations (75.2 percent); the registered budget is the dominant mechanism and was never validated',
        'held_constant': [
            'source_v5_fixB tree and its mapper hash',
            'frozen support set and min frozen support 128 px',
            'Stage A depth admission guard and Stage B observed-depth selector',
            'seed 1219, 50 tracking iterations, learning rates, BA false, mapping depth 40 m',
            '640x192 inputs, byte-identical source frames, RTX 5090 / sm120 natives',
        ],
        'acceptance': {
            'exit_code': 0,
            'manifest_status': 'COMPLETED',
            'input_frames': 200,
            'blocked_gt_accesses': 0,
            'map_finite_all_keyframes': True,
            'frozen_support_constant_within_a_keyframe': 'pixel_support_min == pixel_support_max',
            'cap_mode_on_requires_pose_updates': 'gs_pose_optimization_calls > 0',
        },
        'limits': [
            'single exposed DEV window and single seed; DEV diagnostic only',
            'arms are independent online runs, not bitwise paired maps',
            'the step cap is a post-step projection and does not reset the Adam state',
            'the no-cap arm still has the frozen support set, so it is not the same configuration as B_on',
        ],
        'deploy': 'no new source tree; source_v5_fixB is reused and its mapper hash is re-verified',
        'launcher': 'scripts/launch_cap_ablation_pipeline.py',
        'evaluator': 'scripts/evaluate_cap_ablation.py',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Prepared step-budget ablation worker run_stage_d.py')
    print('  worker sha256: %s' % receipt['worker_sha256'])
    print('  ladder:')
    for arm in receipt['ladder']:
        print('    %-26s %6s mm %6s deg  %s' % (
            arm['label'], arm['cap_translation_mm'], arm['cap_rotation_deg'], arm['status']))
    print('  receipt: %s' % receipt_path)


if __name__ == '__main__':
    main()
