"""Local CPU evaluation of the per-step budget ladder.

Reuses the measurement helpers of evaluate_refine_off.py so the protocol is identical.
The ladder is tight 1 mm / registered 5 mm / loose 20 mm / no cap, all on the frozen
arm C source tree with the freeze and the minimum coverage guard active.

Counterpart protocol:
    00_任务边界与协议/精修单步预算消融_D_20261013.md
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
OUT = R / '05_报告与归档' / '精修单步预算消融_20261013'

LADDER = [
    ('D-tight', 1.0, 0.01, RUN / 'stage_capablation/tight_1mm_0.01deg/full200',
     'KITTI00_200_D_tight_20261013_01'),
    ('D-reg', 5.0, 0.05, RUN / 'stage_refinefix/C_on/full200',
     'KITTI00_200_C_on_20261013_01'),
    ('D-loose', 20.0, 0.2, RUN / 'stage_capablation/loose_20mm_0.2deg/full200',
     'KITTI00_200_D_loose_20261013_01'),
    ('D-nocap', None, None, RUN / 'stage_capablation/no_cap_freeze_only/full200',
     'KITTI00_200_D_nocap_20261013_01'),
]

# The refine-off floor from the completed refine-switch experiment.
FLOOR = {'run_name': 'KITTI00_200_B_off_20261013_01',
         'rpe1_translation_m': 0.041864, 'rpe10_translation_m': 0.312666, 'ate_m': 0.811929}


def clamp_events(run_dir):
    log = run_dir / 'tracking.log'
    if not log.exists():
        return None
    text = log.read_text(errors='replace')
    return {'step_clamped': text.count('"event": "step_clamped"'),
            'support_frozen': text.count('"event": "support_frozen"')}


def main():
    missing = [name for _, _, _, d, name in LADDER if not (d / 'trajectories.npz').exists()]
    if missing:
        raise SystemExit('Ladder arms not mirrored yet: %s' % missing)
    if (OUT / 'metrics.json').exists():
        raise SystemExit('Refusing to overwrite existing evaluation: %s' % (OUT / 'metrics.json'))
    OUT.mkdir(parents=True, exist_ok=True)

    gt = base.load_kitti(R / '01_ORB基线/历史评价/seq00_gt_kitti_200.txt').astype(np.float64)
    delta = np.array(json.loads((R / '01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])

    records = []
    for label, cap_mm, cap_deg, run_dir, run_name in LADDER:
        record = base.measure(label, 'on', run_dir, run_name, gt, delta)
        record['cap_translation_mm'] = cap_mm
        record['cap_rotation_deg'] = cap_deg
        record['clamps'] = clamp_events(run_dir)
        counters = record.get('counters') or {}
        iterations = counters.get('gs_pose_optimization_calls')
        if record['clamps'] and iterations:
            record['clamp_rate_of_iterations'] = record['clamps']['step_clamped'] / iterations
        records.append(record)

    # Pre-registered verdict, judged on RPE1 and RPE10 only.
    ranked = sorted(records, key=lambda r: r['rpe1_translation_m'])
    best = ranked[0]
    if best['variant'] == 'D-reg':
        code = 'CAP_INTERIOR_OPTIMUM'
    elif best['variant'] == 'D-tight':
        code = 'CAP_STILL_TOO_LOOSE'
    else:
        code = 'CAP_HURTS'
    reaches_floor = (best['rpe1_translation_m'] <= FLOOR['rpe1_translation_m']
                     and best['rpe10_translation_m'] <= FLOOR['rpe10_translation_m'])
    verdict = {
        'code': code,
        'judged_on': 'RPE1 and RPE10 translation only; ATE deliberately not used',
        'best_arm': best['variant'],
        'ladder_order_by_rpe1': [r['variant'] for r in ranked],
        'reaches_refine_off_floor': reaches_floor,
        'note': 'if the best arm never clamped, the budget was never binding and the arm is '
                'indistinguishable from the no-cap arm',
    }

    report = {
        'status': 'COMPLETE',
        'date_client': '2026-10-13',
        'protocol': '00_任务边界与协议/精修单步预算消融_D_20261013.md',
        'ladder': records,
        'refine_off_floor': FLOOR,
        'verdict': verdict,
        'limits': [
            'single exposed DEV window and single seed; not a formal adoption test',
            'independent online runs are not bitwise paired maps',
            'the step cap is a post-step projection and does not reset the Adam state',
            'clamp counts are read from the mirrored tracking.log, not from a structured counter',
        ],
        'input_hashes': {r['run_name']: base.sha(d / 'trajectories.npz')
                         for (_, _, _, d, _), r in zip(LADDER, records)},
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    with (OUT / 'ladder.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['arm', 'cap_translation_mm', 'cap_rotation_deg', 'ATE_m', 'RPE1_trans_m',
                         'RPE10_trans_m', 'RPE1_rot_deg', 'pose_updates', 'clamp_events',
                         'clamp_rate'])
        for record in records:
            counters = record.get('counters') or {}
            writer.writerow([record['variant'], record['cap_translation_mm'], record['cap_rotation_deg'],
                             record['ate_m'], record['rpe1_translation_m'], record['rpe10_translation_m'],
                             record['rpe1_rotation_deg'],
                             counters.get('gs_pose_optimization_calls'),
                             (record['clamps'] or {}).get('step_clamped'),
                             record.get('clamp_rate_of_iterations')])

    print('%-8s %8s %8s %9s %9s %9s %8s %9s' %
          ('arm', 'cap_mm', 'cap_deg', 'ATE', 'RPE1_t', 'RPE10_t', 'RPE1rot', 'clamp%'))
    for record in records:
        rate = record.get('clamp_rate_of_iterations')
        print('%-8s %8s %8s %9.6f %9.6f %9.6f %8.4f %9s' % (
            record['variant'], record['cap_translation_mm'], record['cap_rotation_deg'],
            record['ate_m'], record['rpe1_translation_m'], record['rpe10_translation_m'],
            record['rpe1_rotation_deg'], 'n/a' if rate is None else '%.1f%%' % (100 * rate)))
    print('refine-off floor RPE1 %.6f / RPE10 %.6f' % (FLOOR['rpe1_translation_m'], FLOOR['rpe10_translation_m']))
    print('verdict: %s (best %s), reaches floor: %s' % (code, best['variant'], reaches_floor))


if __name__ == '__main__':
    main()
