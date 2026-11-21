"""Local CPU paired evaluation of the refine-switch matrix: V2/A/B x off/on.

Reads the mirrored run directories, applies the frozen T04 protocol (P2->P0,
SE(3), scale correction disabled, fixed input-frame RPE) and reports the matrix
plus support and scale diagnostics. Arms that have not been run and mirrored yet
are reported as PENDING and never silently dropped; the pre-registered paired
conclusion is only emitted once all six arms are present.

Counterpart protocol:
    00_任务边界与协议/精修开关对照_A_off_B_off_预登记_20261013.md
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'scripts'))
from evaluate_kitti_tracking import score, load_kitti, rigid  # noqa: E402

RUN = R / '04_后续运行'
OUT = R / '05_报告与归档' / '精修开关对照_20261013'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

ARMS = [
    ('V2', 'off', RUN / 'matched_v2_off', 'matched_v2_off'),
    ('V2', 'on', RUN / 'matched_v2_on', 'matched_v2_on'),
    ('A', 'off', RUN / 'stage_refoff/A_off/full200', 'KITTI00_200_A_off_20261013_01'),
    ('A', 'on', RUN / 'stage_a_depthguard/full200', 'KITTI00_200_A_on_20261012_01'),
    ('B', 'off', RUN / 'stage_refoff/B_off/full200', 'KITTI00_200_B_off_20261013_01'),
    ('B', 'on', RUN / 'stage_b_depthselection/full200', 'KITTI00_200_B_on_20261012_01'),
]


def umeyama(pred, truth):
    mu_p, mu_t = pred.mean(0), truth.mean(0)
    p0, t0 = pred - mu_p, truth - mu_t
    u, d, vt = np.linalg.svd(t0.T @ p0 / len(pred))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    scale = np.trace(np.diag(d) @ correction) / p0.var(0).sum()
    return scale, mu_t - scale * rotation @ mu_p, rotation


def to_p0(quat_vectors, delta):
    """(200,7) DROID pose vectors -> P0 world matrices, no scale correction."""
    from scipy.spatial.transform import Rotation
    matrices = np.tile(np.eye(4), (len(quat_vectors), 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(quat_vectors[:, 3:]).as_matrix()
    matrices[:, :3, 3] = quat_vectors[:, :3]
    out = matrices.astype(np.float64).copy()
    out[:, :3, 3] += out[:, :3, :3] @ delta
    rigid(out)
    return out


def keyframe_diagnostics(run_dir):
    """Median adjacent-keyframe RPE for coarse and refined poses, plus their gap."""
    pose_path = run_dir / 'pose_stages.npz'
    if not pose_path.exists():
        return {'available': False}
    from scipy.spatial.transform import Rotation
    stages = np.load(pose_path, allow_pickle=False)
    coarse = stages['coarse'].astype(np.float64)
    refined = stages['refined'].astype(np.float64)

    def median_rpe(chain):
        rel = np.linalg.inv(chain[:-1]) @ chain[1:]
        translation = np.linalg.norm(rel[:, :3, 3], axis=1)
        rotation = Rotation.from_matrix(rel[:, :3, :3]).magnitude()
        return float(np.median(translation)), float(np.rad2deg(np.median(rotation)))

    coarse_t, coarse_r = median_rpe(coarse)
    refined_t, refined_r = median_rpe(refined)
    return {
        'available': True,
        'keyframes': int(len(coarse)),
        'coarse_median_rpe1_translation_m': coarse_t,
        'coarse_median_rpe1_rotation_deg': coarse_r,
        'refined_median_rpe1_translation_m': refined_t,
        'refined_median_rpe1_rotation_deg': refined_r,
        'max_coarse_refined_position_delta_m': float(np.abs(coarse[:, :3, 3] - refined[:, :3, 3]).max()),
        'note': 'adjacent-keyframe spacing; not comparable with input-frame RPE',
    }


def measure(variant, refine, run_dir, run_name, gt, delta):
    manifest_path = run_dir / 'run_manifest.json'
    trajectory_path = run_dir / 'trajectories.npz'
    exit_path = run_dir / 'tracking.exit'
    if not trajectory_path.exists():
        return {'variant': variant, 'refine': refine, 'run_name': run_name,
                'status': 'PENDING_NOT_MIRRORED', 'expected_dir': str(run_dir)}

    record = {'variant': variant, 'refine': refine, 'run_name': run_name, 'status': 'EVALUATED'}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        record['manifest_status'] = manifest.get('status')
        record['counters'] = manifest.get('counters')
        record['blocked_gt_accesses'] = manifest.get('gt_access_audit', {}).get('blocked_gt_accesses')
        record['gt_access_blocked'] = manifest.get('gt_access_audit', {}).get('blocked_gt_accesses') == 0
    if exit_path.exists():
        record['exit_code'] = exit_path.read_text().strip()

    estimated = np.load(trajectory_path, allow_pickle=False)['estimated']
    assert estimated.shape == (200, 7) and np.isfinite(estimated).all(), run_name
    assert np.allclose(np.linalg.norm(estimated[:, 3:], axis=1), 1, atol=1e-5), run_name
    matrices = to_p0(estimated, delta)

    metrics, aligned, errors = score(matrices, gt)
    record['ate_m'] = metrics['ate_translation_rmse_m']
    record['ate_mean_m'] = metrics['ate_mean_m']
    record['ate_max_m'] = metrics['ate_max_m']
    for horizon in (1, 10):
        record['rpe%d_translation_m' % horizon] = metrics['rpe_delta%d' % horizon]['translation_rmse_m']
        record['rpe%d_rotation_deg' % horizon] = metrics['rpe_delta%d' % horizon]['rotation_rmse_deg']

    positions = aligned[:, :3, 3]
    truth_positions = gt[:, :3, 3]
    record['path_length_m'] = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    scale, translation, rotation = umeyama(positions, truth_positions)
    sim3 = (scale * (rotation @ positions.T)).T + translation
    record['umeyama_scale'] = float(scale)
    record['rmse_sim3_m'] = float(np.sqrt(((sim3 - truth_positions) ** 2).sum(1).mean()))
    record['rmse_frames_0_19_m'] = float(np.sqrt((errors[:20] ** 2).mean()))
    record['rmse_frames_20_199_m'] = float(np.sqrt((errors[20:] ** 2).mean()))
    record['keyframes'] = keyframe_diagnostics(run_dir)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_kitti(R / '01_ORB基线/历史评价/seq00_gt_kitti_200.txt').astype(np.float64)
    delta = np.array(json.loads((R / '01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])

    records = [measure(variant, refine, run_dir, run_name, gt, delta)
               for variant, refine, run_dir, run_name in ARMS]

    orb_path = RUN / 'orb_matched_v2/CameraTrajectory.txt'
    orb_reference = None
    if orb_path.exists():
        values = np.loadtxt(orb_path, encoding='utf-8').reshape(-1, 3, 4)
        orb = np.tile(np.eye(4), (200, 1, 1))
        orb[:, :3, :3] = values[:, :, :3]
        orb[:, :3, 3] = values[:, :, 3]
        orb = orb.astype(np.float64).copy()
        orb[:, :3, 3] += orb[:, :3, :3] @ delta
        rigid(orb)
        metrics, aligned, errors = score(orb, gt)
        positions = aligned[:, :3, 3]
        scale, translation, rotation = umeyama(positions, gt[:, :3, 3])
        sim3 = (scale * (rotation @ positions.T)).T + translation
        orb_reference = {
            'role': 'external reference, not part of the 3x2 refine matrix',
            'ate_m': metrics['ate_translation_rmse_m'],
            'rpe1_translation_m': metrics['rpe_delta1']['translation_rmse_m'],
            'rpe10_translation_m': metrics['rpe_delta10']['translation_rmse_m'],
            'rpe1_rotation_deg': metrics['rpe_delta1']['rotation_rmse_deg'],
            'path_length_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            'umeyama_scale': float(scale),
            'rmse_sim3_m': float(np.sqrt(((sim3 - gt[:, :3, 3]) ** 2).sum(1).mean())),
            'rmse_frames_0_19_m': float(np.sqrt((errors[:20] ** 2).mean())),
            'rmse_frames_20_199_m': float(np.sqrt((errors[20:] ** 2).mean())),
        }

    by_key = {(r['variant'], r['refine']): r for r in records}
    complete = all(r['status'] == 'EVALUATED' for r in records)

    # Pre-registered rule 7: judged on RPE only, never on ATE alone.
    conclusion = {'code': 'PENDING_OFF_ARMS', 'rule': '7'}
    if complete:
        verdicts = {}
        for variant in ('A', 'B'):
            off, on = by_key[(variant, 'off')], by_key[(variant, 'on')]
            verdicts[variant] = {
                'off_better_rpe1': off['rpe1_translation_m'] < on['rpe1_translation_m'],
                'off_better_rpe10': off['rpe10_translation_m'] < on['rpe10_translation_m'],
                'rpe1_change_percent': 100 * (off['rpe1_translation_m'] / on['rpe1_translation_m'] - 1),
                'rpe10_change_percent': 100 * (off['rpe10_translation_m'] / on['rpe10_translation_m'] - 1),
            }
        both_off_better = all(v['off_better_rpe1'] and v['off_better_rpe10'] for v in verdicts.values())
        both_off_worse = all(not v['off_better_rpe1'] and not v['off_better_rpe10'] for v in verdicts.values())
        code = 'REFINEMENT_HARMS' if both_off_better else ('REFINEMENT_HELPS' if both_off_worse else 'MIXED')
        conclusion = {
            'code': code,
            'rule': '7.1 / 7.2 / 7.3',
            'judged_on': 'RPE1 and RPE10 translation only; ATE deliberately not used',
            'per_variant': verdicts,
        }

    status = 'COMPLETE' if complete else 'PARTIAL_AWAITING_OFF_ARMS'
    # A partial run must not occupy the final filenames, otherwise the completed
    # evaluation would later be blocked by the no-overwrite guard.
    suffix = '' if complete else '.partial'
    metrics_path = OUT / ('refine_switch_metrics%s.json' % suffix)
    table_path = OUT / ('refine_switch_table%s.csv' % suffix)
    if metrics_path.exists():
        raise SystemExit('Refusing to overwrite existing evaluation: %s' % metrics_path)

    report = {
        'status': status,
        'date_client': '2026-10-13',
        'protocol': '00_任务边界与协议/精修开关对照_A_off_B_off_预登记_20261013.md',
        'evaluation_protocol': {
            'window': 'KITTI00 source frames 0-199',
            'alignment': 'SE(3) after P2->P0 conversion',
            'scale_correction': False,
            'rpe_horizons': [1, 10],
        },
        'matrix': records,
        'orb_reference': orb_reference,
        'paired_conclusion': conclusion,
        'diagnostics_not_results': {
            'umeyama_scale_and_sim3': 'diagnostic only; never a substitute for the SE(3) numbers',
            'segment_rmse': 'frames 0-19 vs 20-199, reported to localise rather than to conclude',
        },
        'limits': [
            'single exposed DEV window and single seed; not a formal adoption test',
            'independent online runs are not bitwise paired maps',
            'Gaussian uses resized 640x192 stereo-derived RGB-D; ORB uses original stereo 1241x376',
            'trajectory includes the author offline filler, so it is not strictly online per-frame tracking',
            'GT is read only by this CPU evaluator; the runtime never sees poses',
        ],
        'input_hashes': {
            'trajectories': {r['run_name']: sha(run_dir / 'trajectories.npz')
                             for (variant, refine, run_dir, run_name), r in
                             zip(ARMS, records) if r['status'] == 'EVALUATED'},
            'gt': sha(R / '01_ORB基线/历史评价/seq00_gt_kitti_200.txt'),
            'evaluator': sha(Path(__file__)),
        },
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    with table_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'refine', 'status', 'run_name', 'ATE_m', 'RPE1_trans_m',
                         'RPE10_trans_m', 'RPE1_rot_deg', 'path_length_m', 'umeyama_scale',
                         'RMSE_sim3_m', 'RMSE_f0_19_m', 'RMSE_f20_199_m', 'pose_updates'])
        for record in records:
            counters = record.get('counters') or {}
            writer.writerow([record['variant'], record['refine'], record['status'], record['run_name'],
                             record.get('ate_m'), record.get('rpe1_translation_m'),
                             record.get('rpe10_translation_m'), record.get('rpe1_rotation_deg'),
                             record.get('path_length_m'), record.get('umeyama_scale'),
                             record.get('rmse_sim3_m'), record.get('rmse_frames_0_19_m'),
                             record.get('rmse_frames_20_199_m'),
                             counters.get('gs_pose_optimization_calls')])

    print('status: %s' % report['status'])
    print('%-5s %-4s %-9s %9s %9s %9s %9s %8s' %
          ('var', 'ref', 'status', 'ATE', 'RPE1_t', 'RPE10_t', 'RPE1_rot', 'scale'))
    for record in records:
        if record['status'] == 'EVALUATED':
            print('%-5s %-4s %-9s %9.6f %9.6f %9.6f %9.4f %8.4f' %
                  (record['variant'], record['refine'], 'evaluated', record['ate_m'],
                   record['rpe1_translation_m'], record['rpe10_translation_m'],
                   record['rpe1_rotation_deg'], record['umeyama_scale']))
        else:
            print('%-5s %-4s %-9s' % (record['variant'], record['refine'], 'PENDING'))
    print('paired conclusion: %s' % conclusion['code'])
    print('written: %s' % metrics_path)


if __name__ == '__main__':
    main()
