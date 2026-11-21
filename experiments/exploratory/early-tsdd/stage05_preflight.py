#!/usr/bin/env python3
"""Read-only geometry/evidence checks for the restricted Stage-2 dev smoke."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--g2-root', type=Path, required=True)
    p.add_argument('--ims-packet', type=Path, required=True)
    p.add_argument('--tsdd-output', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    src = Path(__file__).parent / 'source/compute_motion_evidence_P1.py'
    spec = importlib.util.spec_from_file_location('legacy_motion', src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rigid = module.compute_rigid_flow
    report = {'status': 'ENGINEERING_PREFLIGHT_PASS', 'geometry_source_sha256': digest(src),
              'g2_real_pairs': {}, 't22_existing_evidence': {}, 'tsdd_smoke': {}}

    k = np.array([[100., 0., 6.], [0., 100., 5.], [0., 0., 1.]])
    depth = np.full((12, 16), 10., np.float32)
    eye = np.eye(4)
    f, valid, _ = rigid(depth, k, eye, eye)
    assert valid.all() and float(np.max(np.abs(f))) < 1e-4
    moved = eye.copy(); moved[0, 3] = 0.2
    f, valid, _ = rigid(depth, k, eye, moved)
    assert valid.any() and abs(float(np.median(f[..., 0][valid])) + 2.) < 1e-4
    assert float(np.max(np.abs(f[..., 1][valid]))) < 1e-4
    bad = depth.copy(); bad[0, 0] = 0
    _, valid_bad, _ = rigid(bad, k, eye, eye)
    assert not valid_bad[0, 0]
    report['synthetic'] = {'identity_max_abs_flow_lt_1e-4': True,
                           'camera_x_0p2m_at_10m_expected_minus_2px': True,
                           'zero_depth_invalid': True}

    for name in ('vkitti2', 'ims', 'kitti08'):
        base = a.g2_root / 'samples' / name
        mpath = base / 'manifest_dev.json'
        man = json.loads(mpath.read_text())
        camera = np.asarray(man['calibration']['K'], dtype=np.float64)
        pairs = []
        for seq in man['sequences']:
            x, y = seq['frames']
            assert x['next_frame_id'] == y['frame_id']
            d = np.load(base / x['depth']['path'], allow_pickle=False)
            t0, t1 = np.asarray(x['pose_c2w']), np.asarray(y['pose_c2w'])
            assert abs(np.linalg.det(t0[:3, :3]) - 1) < 1e-2
            assert abs(np.linalg.det(t1[:3, :3]) - 1) < 1e-2
            flow, valid, _ = rigid(d, camera, t0, t1)
            assert 0 < int(valid.sum()) <= d.size and np.isfinite(flow).all()
            pairs.append({'frames': [x['frame_id'], y['frame_id']],
                          'valid_fraction': float(valid.mean()),
                          'median_rigid_flow_px': float(np.median(np.linalg.norm(flow[valid], axis=1)))})
        report['g2_real_pairs'][name] = {'manifest_sha256': digest(mpath), 'pairs': pairs}

    flow_arrays = {}
    for route, folder in [('oracle', 'motion_gt_oracle_raft'),
                          ('predicted', 'motion_mask2former_pred_raft')]:
        root = a.ims_packet / folder
        summary = json.loads((root / 'generation_summary.json').read_text())
        assert summary['status'] == 'GENERATED' and summary['pairs'] == 7
        details = []
        for i in range(7):
            stem = f'{i:06d}_{i+1:06d}'
            j = json.loads((root / (stem + '.json')).read_text())
            assert j['frame_t'] == f'{i:06d}' and j['frame_next'] == f'{i+1:06d}'
            with np.load(root / (stem + '.npz'), allow_pickle=False) as z:
                assert z['observed_flow'].shape == z['rigid_flow'].shape == (375, 1242, 2)
                assert z['residual'].shape == z['valid_mask'].shape == (375, 1242)
                valid = z['valid_mask']
                assert valid.dtype == np.bool_ and np.isfinite(z['residual'][valid]).all()
                assert np.isfinite(z['observed_flow'][valid]).all()
                assert np.isfinite(z['rigid_flow'][valid]).all()
                flow_arrays.setdefault(i, {})[route] = z['observed_flow'].copy()
                valid_fraction = float(valid.mean())
            probs = [item['dynamic_probability'] for item in j['instances']]
            assert all(p is None or (0 <= p <= 1 and np.isfinite(p)) for p in probs)
            details.append({'pair': stem, 'valid_fraction': valid_fraction,
                            'instances': len(j['instances']),
                            'background_median_px': j['normalization']['background_median_px']})
        report['t22_existing_evidence'][route] = {
            'summary_sha256': digest(root / 'generation_summary.json'), 'pairs': details,
            'candidate_source': 'GT-oracle' if route == 'oracle' else 'Mask2Former-predicted'}
    report['t22_flow_route_mean_abs_difference_px'] = float(np.mean([
        np.mean(np.abs(x['oracle'] - x['predicted'])) for x in flow_arrays.values()]))

    out = a.tsdd_output
    s = json.loads((out / 'generation_summary.json').read_text())
    assert s['status'] == 'ENGINEERING_SMOKE' and s['frames'] == 8
    actual_counts = {'Static': 0, 'Uncertain': 0, 'Dynamic': 0}
    for i in range(8):
        with np.load(out / f'{i:06d}_states.npz', allow_pickle=False) as z:
            state = z['state_map']; states = z['states']
            assert state.shape == (375, 1242) and state.dtype == np.uint8
            assert set(np.unique(state).tolist()) <= {0, 1, 2, 255}
            assert set(np.unique(states).tolist()) <= {0, 1, 2}
            if i == 0:
                assert np.all(states == 1)
            for label, code in [('Static', 0), ('Uncertain', 1), ('Dynamic', 2)]:
                actual_counts[label] += int(np.count_nonzero(states == code))
    assert actual_counts == s['state_counts']
    report['tsdd_smoke'] = {'summary_sha256': digest(out / 'generation_summary.json'),
                            'frames': 8, 'instance_observations': int(s['total_instance_observations']),
                            'state_counts': actual_counts, 'uncertain_rate': s['uncertain_rate'],
                            'first_frame_all_uncertain': True}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'pairs': 9,
                      't22_pairs_per_route': 7, 'tsdd_frames': 8}))


if __name__ == '__main__':
    main()
