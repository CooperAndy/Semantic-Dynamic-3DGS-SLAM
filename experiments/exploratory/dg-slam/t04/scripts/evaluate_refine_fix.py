"""Local CPU evaluation of arm C against the frozen refine-switch matrix.

Reuses the measurement helpers of evaluate_refine_off.py so the protocol is identical
(P2->P0, SE(3), scale correction disabled, fixed input-frame RPE). Adds the C_on arm and
reports the two pre-registered contrasts: C_on vs B_on (same source tree, objective not
fixed) and C_on vs the refine-off floor.

Counterpart protocol:
    00_任务边界与协议/精修目标函数修复_C_20261013.md
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_refine_off as base  # noqa: E402

R = base.R
RUN = base.RUN
OUT = R / '05_报告与归档' / '精修目标函数修复_20261013'

C_RUN_DIR = RUN / 'stage_refinefix/C_on/full200'
C_RUN_NAME = 'KITTI00_200_C_on_20261013_01'

ARMS = list(base.ARMS) + [('C', 'on', C_RUN_DIR, C_RUN_NAME)]


def freeze_evidence(run_dir):
    path = run_dir / 'selection_history.jsonl'
    if not path.exists():
        return {'available': False}
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    refined = [r for r in rows if 'pixel_support_min' in r]
    fallen = [r for r in rows if r.get('fallback')]
    constant = [r for r in refined if r['pixel_support_min'] == r['pixel_support_max']]
    return {
        'available': True,
        'keyframes': len(rows),
        'refined_keyframes': len(refined),
        'fallback_keyframes': len(fallen),
        'fallback_source_frames': [r['source_frame'] for r in fallen],
        'fallback_reasons': sorted({r.get('reason') for r in fallen if r.get('reason')}),
        'keyframes_with_constant_frozen_support': len(constant),
        'freeze_holds_everywhere': len(refined) > 0 and len(constant) == len(refined),
    }


def main():
    if not (C_RUN_DIR / 'trajectories.npz').exists():
        raise SystemExit('Arm C has not been run and mirrored yet: %s' % C_RUN_DIR)
    if (OUT / 'metrics.json').exists():
        raise SystemExit('Refusing to overwrite existing evaluation: %s' % (OUT / 'metrics.json'))
    OUT.mkdir(parents=True, exist_ok=True)

    gt = base.load_kitti(R / '01_ORB基线/历史评价/seq00_gt_kitti_200.txt').astype(np.float64)
    delta = np.array(json.loads((R / '01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])

    records = [base.measure(variant, refine, run_dir, run_name, gt, delta)
               for variant, refine, run_dir, run_name in ARMS]
    by_key = {(r['variant'], r['refine']): r for r in records}

    c = by_key.get(('C', 'on'))
    if not c or c['status'] != 'EVALUATED':
        raise SystemExit('Arm C has not been run and mirrored yet: %s' % C_RUN_DIR)
    b_on = by_key[('B', 'on')]
    floor = sorted((by_key[(v, 'off')] for v in ('V2', 'A', 'B')), key=lambda r: r['rpe1_translation_m'])[0]

    contrast = {
        'C_on_vs_B_on': {
            'note': 'same source tree, only the refinement objective differs',
            'ate_change_percent': 100 * (c['ate_m'] / b_on['ate_m'] - 1),
            'rpe1_change_percent': 100 * (c['rpe1_translation_m'] / b_on['rpe1_translation_m'] - 1),
            'rpe10_change_percent': 100 * (c['rpe10_translation_m'] / b_on['rpe10_translation_m'] - 1),
            'rpe1_rotation_change_deg': c['rpe1_rotation_deg'] - b_on['rpe1_rotation_deg'],
        },
        'C_on_vs_floor': {
            'floor_run': floor['run_name'],
            'ate_change_percent': 100 * (c['ate_m'] / floor['ate_m'] - 1),
            'rpe1_change_percent': 100 * (c['rpe1_translation_m'] / floor['rpe1_translation_m'] - 1),
            'rpe10_change_percent': 100 * (c['rpe10_translation_m'] / floor['rpe10_translation_m'] - 1),
        },
    }

    # Pre-registered verdict, judged on RPE only.
    beats_b_on = (c['rpe1_translation_m'] < b_on['rpe1_translation_m']
                  and c['rpe10_translation_m'] < b_on['rpe10_translation_m'])
    reaches_floor = (c['rpe1_translation_m'] <= floor['rpe1_translation_m']
                     and c['rpe10_translation_m'] <= floor['rpe10_translation_m'])
    if beats_b_on and reaches_floor:
        code = 'REFINEMENT_NO_LONGER_HARMFUL'
    elif beats_b_on:
        code = 'FIX_IMPROVES_BUT_STILL_BEHIND_FLOOR'
    else:
        code = 'FIX_INSUFFICIENT'
    verdict = {
        'code': code,
        'judged_on': 'RPE1 and RPE10 translation only; ATE deliberately not used',
        'beats_B_on_on_both_rpe': beats_b_on,
        'reaches_refine_off_floor_on_both_rpe': reaches_floor,
    }

    report = {
        'status': 'COMPLETE',
        'date_client': '2026-10-13',
        'protocol': '00_任务边界与协议/精修目标函数修复_C_20261013.md',
        'matrix': records,
        'contrast': contrast,
        'verdict': verdict,
        'freeze_evidence_arm_C': freeze_evidence(C_RUN_DIR),
        'limits': [
            'single exposed DEV window and single seed; not a formal adoption test',
            'independent online runs are not bitwise paired maps',
            'trajectory includes the author offline filler; not strictly per-frame online tracking',
            'GT is read only by this CPU evaluator; the runtime never sees poses',
            'the step cap is a post-step projection and does not reset the Adam state',
        ],
        'input_hashes': {
            'trajectories': {r['run_name']: base.sha(run_dir / 'trajectories.npz')
                             for (variant, refine, run_dir, run_name), r in
                             zip(ARMS, records) if r['status'] == 'EVALUATED'},
        },
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    with (OUT / 'matrix.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'refine', 'run_name', 'ATE_m', 'RPE1_trans_m', 'RPE10_trans_m',
                         'RPE1_rot_deg', 'path_length_m', 'umeyama_scale', 'RMSE_sim3_m', 'pose_updates'])
        for record in records:
            counters = record.get('counters') or {}
            writer.writerow([record['variant'], record['refine'], record['run_name'],
                             record.get('ate_m'), record.get('rpe1_translation_m'),
                             record.get('rpe10_translation_m'), record.get('rpe1_rotation_deg'),
                             record.get('path_length_m'), record.get('umeyama_scale'),
                             record.get('rmse_sim3_m'), counters.get('gs_pose_optimization_calls')])

    print('%-4s %-4s %9s %9s %9s %8s' % ('var', 'ref', 'ATE', 'RPE1_t', 'RPE10_t', 'RPE1rot'))
    for record in records:
        print('%-4s %-4s %9.6f %9.6f %9.6f %8.4f' % (
            record['variant'], record['refine'], record['ate_m'],
            record['rpe1_translation_m'], record['rpe10_translation_m'], record['rpe1_rotation_deg']))
    print('C vs B_on : ATE %+.2f%%, RPE1 %+.2f%%, RPE10 %+.2f%%' % (
        contrast['C_on_vs_B_on']['ate_change_percent'], contrast['C_on_vs_B_on']['rpe1_change_percent'],
        contrast['C_on_vs_B_on']['rpe10_change_percent']))
    print('C vs floor: ATE %+.2f%%, RPE1 %+.2f%%, RPE10 %+.2f%%' % (
        contrast['C_on_vs_floor']['ate_change_percent'], contrast['C_on_vs_floor']['rpe1_change_percent'],
        contrast['C_on_vs_floor']['rpe10_change_percent']))
    print('verdict   : %s' % code)
    print('freeze    : %s' % json.dumps(report['freeze_evidence_arm_C'], ensure_ascii=False))


if __name__ == '__main__':
    main()
