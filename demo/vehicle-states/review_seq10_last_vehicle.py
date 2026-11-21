"""Review the last Seq10 vehicle on the installed O3F map."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / '阶段2/动态语义Demo_设计/T07_动态语义独立副本'
BASE = DEMO / 'scene-data/KITTI10_O3F_1000'
RUN = Path(__file__).resolve().parent / 'seq10_o3f_20261116'
INPUT = RUN / 'object_inputs'
TRACKS = {690, 693, 695, 696, 703}
CONFIG = DEMO / '源码/scene-config.kitti10-o3f-1000.json'
OUTPUT = BASE / 'FinalTruckRemoval'
PREFIX = 'scene-data/KITTI10_O3F_1000/FinalTruckRemoval'


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def votes_for_vehicle(points, creation_frames, expanded=False, residual=False):
    from scipy.ndimage import binary_dilation, binary_erosion
    poses = {p['source_frame']: p for p in read_json(BASE / 'camera_poses.json')}
    intrinsics = np.array(read_json(INPUT / 'manifest.json')['intrinsics'], dtype=np.float32)
    observations = read_json(RUN / 'observations/observations.json')
    observations = [r for r in observations if r['track_id'] in TRACKS
                    and (r['raw_state'] == 'dynamic' or (expanded and r['source_frame'] == 950))]
    votes = np.zeros(len(points), dtype=np.uint16)
    conflicts = np.zeros_like(votes)
    recent_vote = np.zeros(len(points), dtype=bool)
    frame_hits = {}
    for row in observations:
        n = row['source_frame']
        instances = read_json(INPUT / f'instances/{n:06d}.json')['instances']
        target, = [i for i in instances if i['track_id'] == row['track_id']]
        if target['confidence'] < 0.5:
            continue
        with np.load(INPUT / f'instances/{n:06d}_masks.npz') as data:
            masks = data['masks']
        with np.load(INPUT / f'masked_depth/{n:06d}.npz') as data:
            depth = data['depth']
        pose = poses[n]
        camera = (points - np.array(pose['position'], dtype=np.float32)) @ np.array(pose['rotation'], dtype=np.float32)
        ids = np.flatnonzero((camera[:, 2] >= 0.5) & (camera[:, 2] <= 40) & (creation_frames <= n))
        x, y, z = camera[ids].T
        u = np.rint(intrinsics[0, 0] * x / z + intrinsics[0, 2]).astype(int)
        v = np.rint(intrinsics[1, 1] * y / z + intrinsics[1, 2]).astype(int)
        in_view = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        ids, u, v, z = ids[in_view], u[in_view], v[in_view], z[in_view]
        measured = depth[v, u]
        if residual:
            tolerance = 0.8 + 0.10 * z
        elif expanded:
            tolerance = 0.45 + 0.075 * z
        else:
            tolerance = 0.18 + 0.035 * z
        in_depth = (measured > 0.5) & (np.abs(measured - z) <= tolerance)
        target_mask = masks[target['mask_index']]
        interior = (binary_dilation(target_mask, iterations=3) if expanded
                    else binary_erosion(target_mask, iterations=3, border_value=1))
        own = interior[v, u]
        hit = ids[own & in_depth]
        votes[hit] += 1
        recent_vote[hit[creation_frames[hit] >= n - (5 if expanded else 2)]] = True
        frame_hits[n] = hit
        others = [i['mask_index'] for i in instances if i['track_id'] not in TRACKS
                  and i['category'] in ('car', 'truck', 'bus', 'van') and i['confidence'] >= 0.5]
        if others:
            conflicts[ids[np.any(masks[others][:, v, u], axis=0) & in_depth]] += 1
    return votes, conflicts, recent_vote, frame_hits


def analyze():
    with np.load(ROOT / '特调地图/seq10_o3f_1000_delivery/params.npz') as data:
        points = data['means3D'].astype(np.float32)
        creation = data['timestep'].reshape(-1)
        opacity = 1 / (1 + np.exp(-np.clip(data['logit_opacities'].reshape(-1), -30, 30)))
        scale = np.exp(data['log_scales'].reshape(-1))
    with np.load(BASE / 'ObjectStates/evidence/gaussian_region_ids.npz') as data:
        unassigned = data['region_id'] < 0
    ids = np.flatnonzero(unassigned & (creation >= 950) & (opacity >= 0.1) & (scale <= 0.25))
    votes, conflicts, recent_vote, frame_hits = votes_for_vehicle(points[ids], creation[ids])
    selected = (votes >= 4) & (conflicts == 0) & recent_vote
    chosen = ids[selected]
    # Use points supported when the truck body is prominent in the original image.
    anchor_frame = 985
    anchor_candidates = ids[frame_hits[anchor_frame][selected[frame_hits[anchor_frame]]]]
    center = np.median(points[anchor_candidates], axis=0)
    anchor = int(anchor_candidates[np.argmin(np.linalg.norm(points[anchor_candidates] - center, axis=1))])
    mask = np.zeros(len(points), dtype=bool)
    mask[chosen] = True
    frame_counts = {n: int(selected[hit].sum()) for n, hit in frame_hits.items()}
    return points, mask, anchor, anchor_frame, frame_counts


def expand_selection(core):
    from scipy.spatial import cKDTree

    with np.load(ROOT / '特调地图/seq10_o3f_1000_delivery/params.npz') as data:
        points = data['means3D'].astype(np.float32)
        creation = data['timestep'].reshape(-1)
        opacity = 1 / (1 + np.exp(-np.clip(data['logit_opacities'].reshape(-1), -30, 30)))
        scale = np.exp(data['log_scales'].reshape(-1))
    with np.load(BASE / 'ObjectStates/evidence/gaussian_region_ids.npz') as data:
        unassigned = data['region_id'] < 0
    ids = np.flatnonzero(~core & unassigned & (creation >= 925) & (opacity >= 0.015) & (scale <= 0.65))
    distance = cKDTree(points[core]).query(points[ids], distance_upper_bound=0.85)[0]
    near = np.isfinite(distance)
    ids, distance = ids[near], distance[near]
    votes, conflicts, recent_vote, frame_hits = votes_for_vehicle(points[ids], creation[ids], expanded=True)
    selected = (conflicts == 0) & ((recent_vote & (votes >= 2)) | ((distance <= 0.4) & (votes >= 4)))
    mask = core.copy()
    mask[ids[selected]] = True
    frame_counts = {n: int(selected[hit].sum()) for n, hit in frame_hits.items()}

    # The looser depth pass stays near an established vehicle surface and
    # excludes points with repeated, high-confidence static semantic evidence.
    with np.load(RUN / 'semantic/semantic_gaussian_labels.npz') as data:
        assert np.array_equal(data['source_gaussian_index'], np.arange(len(points)))
        static_labels = np.isin(data['label'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        protected = static_labels & (data['confidence'] >= 0.75) & (data['observation_count'] >= 4)
    ids = np.flatnonzero(~mask & ~protected & unassigned & (creation >= 925) & (opacity >= 0.015) & (scale <= 0.65))
    distance = cKDTree(points[mask]).query(points[ids], distance_upper_bound=0.9)[0]
    near = np.isfinite(distance)
    ids, distance = ids[near], distance[near]
    votes, conflicts, recent_vote, frame_hits = votes_for_vehicle(points[ids], creation[ids], expanded=True, residual=True)
    selected = (conflicts == 0) & (
        ((distance <= 0.40) & (votes >= 2)) |
        (recent_vote & (votes >= 4)) |
        ((distance <= 0.7) & (votes >= 6))
    )
    mask[ids[selected]] = True
    for frame, hit in frame_hits.items():
        frame_counts[frame] = frame_counts.get(frame, 0) + int(selected[hit].sum())
    ids = np.flatnonzero(~mask & ~protected & unassigned & (creation >= 945) & (scale <= 0.40))
    # Interior fragments can have inconsistent stereo depth at the silhouette.
    # Require dense local support, without growing the mask from new points.
    neighbors = cKDTree(points[mask]).query(points[ids], k=8, distance_upper_bound=0.45)[0]
    fragments = ids[(neighbors[:, 0] <= 0.25) & np.isfinite(neighbors[:, -1])]
    mask[fragments] = True
    return points, mask, frame_counts, len(fragments)


def ply_rows(path):
    with path.open('rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            assert line
            header += line
    count = int(re.search(rb'element vertex (\d+)', header)[1])
    assert len(re.findall(rb'^property float ', header, re.M)) == 17
    assert path.stat().st_size == len(header) + count * 68
    return header, np.memmap(path, dtype='<f4', mode='r', offset=len(header), shape=(count, 17))


def write_and_verify_maps(source, mask, points, output=OUTPUT):
    header, rows = ply_rows(source)
    count = int(mask.sum())
    assert np.array_equal(rows[:, :3], points)
    reduced_header = re.sub(rb'element vertex \d+', f'element vertex {len(rows) - count}'.encode(), header)
    removed = output / 'RGB_removed.ply'
    highlight = output / 'RGB_highlight.ply'
    red = ((np.array([1.0, 0.0, 0.0]) - 0.5) / 0.28209479177387814).astype('<f4')
    with removed.open('wb') as cut, highlight.open('wb') as preview:
        cut.write(reduced_header)
        preview.write(header)
        for start in range(0, len(rows), 200_000):
            block = rows[start:start + 200_000].copy()
            selected = mask[start:start + len(block)]
            block[~selected].tofile(cut)
            block[selected, 6:9] = red
            block.tofile(preview)
    _, cut = ply_rows(removed)
    _, preview = ply_rows(highlight)
    assert len(cut) == len(rows) - count and len(preview) == len(rows)
    offset = 0
    for start in range(0, len(rows), 200_000):
        block = rows[start:start + 200_000]
        selected = mask[start:start + len(block)]
        verified = preview[start:start + len(block)]
        retained = block[~selected]
        assert np.array_equal(cut[offset:offset + len(retained)], retained)
        assert np.array_equal(verified[~selected], retained)
        assert np.array_equal(verified[selected, :6], block[selected, :6])
        assert np.array_equal(verified[selected, 9:], block[selected, 9:])
        assert np.all(verified[selected, 6:9] == red)
        offset += len(retained)
    return removed, highlight


def install():
    config = read_json(CONFIG)
    base = read_json(BASE / 'manifest.json')
    snapshot = read_json(BASE / 'ObjectStates/snapshot.json')
    assert config['sceneId'] == snapshot['scene_id'] == 'kitti10_o3f_000000_000999'
    assert not any(obj['object_id'] == 'V015' for obj in snapshot['objects'])
    source = BASE / 'RGB.ply'
    before = sha256(source)
    assert before == base['files']['RGB.ply']['sha256']
    points, mask, anchor, anchor_frame, frame_counts = analyze()
    count = int(mask.sum())
    assert 0 < count < len(points) and mask[anchor]
    assert frame_counts[anchor_frame] > 0

    OUTPUT.mkdir(exist_ok=True)
    removed, highlight = write_and_verify_maps(source, mask, points)
    assert sha256(source) == before
    mask_path = OUTPUT / 'selected_mask.npz'
    np.savez_compressed(mask_path, selected=mask)
    observations = read_json(RUN / 'observations/observations.json')
    observations = [row for row in observations if row['track_id'] in TRACKS]
    used_frames = [frame for frame, hits in frame_counts.items() if hits > 0]
    score = float(np.median([row['raw_score'] for row in observations]))
    assert score > 0.6 and len(observations) == 46
    xyz = points[anchor]
    snapshot['objects'].append({
        'track_id': 15, 'object_id': 'V015', 'category': 'truck', 'state': '动态',
        'final_state': 'dynamic', 'fixed_state': True, 'score': score,
        'final_reason': 'reviewed_cross_track_motion_and_direct_map_point_support',
        'source_track_ids': sorted(TRACKS), 'display_estimate': True,
        'world_xyz': xyz.tolist(), 'viewer_xyz': (xyz * [-1, -1, 1]).tolist(),
        'anchor_source_gaussian_index': anchor,
        'anchor_status': 'actual_selected_O3F_gaussian',
        'anchor_verified': False, 'semantic_anchor_bound': False,
        'region_gaussian_count': count,
        'aggregate_evidence': {'first_frame': min(used_frames), 'last_frame': max(used_frames)},
    })
    counts = dict(Counter(obj['final_state'] for obj in snapshot['objects']))
    for key in ('object_counts', 'track_counts', 'positioned_track_counts'):
        snapshot[key] = counts
    for key in ('positioned_object_count', 'positioned_track_count'):
        snapshot[key] = len(snapshot['objects'])
    write_json(OUTPUT / 'snapshot.json', snapshot)
    manifest = {
        'scene': config['sceneId'], 'object_id': 'V015', 'previous_tmd_object_id': 'V012',
        'source_track_ids': sorted(TRACKS), 'source_observations': len(observations),
        'raw_dynamic_observations': sum(row['raw_state'] == 'dynamic' for row in observations),
        'median_raw_score': score, 'source_map_sha256': base['params_sha256'],
        'source_rgb_sha256': before, 'anchor_frame': anchor_frame, 'anchor_index': anchor,
        'frame_support': frame_counts, 'removed_gaussians': count, 'retained_gaussians': len(mask) - count,
        'selection': 'Unassigned O3F points created from frame 950 onward, opacity >= 0.1, scale <= 0.25 m, '
                     'at least four projected matches inside eroded target masks with consistent depth; '
                     'at least one match within two frames of creation and no depth-consistent other-vehicle match.',
        'mask_sha256': sha256(mask_path), 'removed_rgb_sha256': sha256(removed),
        'highlight_rgb_sha256': sha256(highlight),
    }
    write_json(OUTPUT / 'manifest.json', manifest)
    config['motionSnapshot'] = f'{PREFIX}/snapshot.json'
    config['dynamicVehicleRemoval'] = {
        'version': 2, 'removedCount': count, 'gaussianCount': int(len(mask) - count),
        'maskSha256': manifest['mask_sha256'], 'manifest': f'{PREFIX}/manifest.json',
        'highlights': {'F1': f'{PREFIX}/RGB_highlight.ply'},
        'variants': {'F1': {'rgb': f'{PREFIX}/RGB_removed.ply', 'rgbWithTrajectory': None}},
        'warning': 'Seq10 V015 truck selection uses multi-frame mask and depth matches. '
                   'Preview the selection before hiding. Some remnants may remain.',
    }
    write_json(CONFIG, config)
    print(f'Installed V015 with {count:,} selected points; anchor {anchor}, frame {anchor_frame}, score {score:.3f}.')


def install_expanded():
    config = read_json(CONFIG)
    base_manifest = read_json(OUTPUT / 'manifest.json')
    original = BASE / 'RGB.ply'
    original_sha = sha256(original)
    mask_path = OUTPUT / 'selected_mask.npz'
    assert original_sha == base_manifest['source_rgb_sha256']
    assert sha256(mask_path) == base_manifest['mask_sha256']
    with np.load(mask_path) as data:
        core = data['selected']
    points, mask, frame_counts, fragments = expand_selection(core)
    count = int(mask.sum())
    assert np.all(mask[core]) and count > int(core.sum())
    with np.load(BASE / 'ObjectStates/evidence/gaussian_region_ids.npz') as data:
        assert not np.any(mask & (data['region_id'] >= 0))
    output = BASE / 'FinalTruckRemovalExpanded'
    output.mkdir(exist_ok=True)
    removed, highlight = write_and_verify_maps(original, mask, points, output)
    assert sha256(original) == original_sha
    selected_path = output / 'selected_mask.npz'
    np.savez_compressed(selected_path, selected=mask, additional=mask & ~core)
    manifest = {
        'scene': config['sceneId'], 'object_id': base_manifest['object_id'],
        'source_rgb_sha256': original_sha, 'original_mask_sha256': base_manifest['mask_sha256'],
        'original_count': int(core.sum()), 'removed_gaussians': count,
        'additional_projected_gaussians': count - int(core.sum()) - fragments,
        'additional_near_surface_fragments': fragments,
        'additional_projection_support': frame_counts, 'retained_gaussians': len(mask) - count,
        'mask_sha256': sha256(selected_path), 'removed_rgb_sha256': sha256(removed),
        'highlight_rgb_sha256': sha256(highlight),
        'selection': 'Original V015 mask plus nearby unassigned points with repeated target-mask/depth support '
                     'and close fragments supported by at least eight nearby selected points. '
                     'Existing object regions are excluded from all passes.',
    }
    write_json(output / 'manifest.json', manifest)
    snapshot = read_json(OUTPUT / 'snapshot.json')
    truck, = [obj for obj in snapshot['objects'] if obj['object_id'] == manifest['object_id']]
    assert mask[truck['anchor_source_gaussian_index']]
    truck['region_gaussian_count'] = count
    write_json(output / 'snapshot.json', snapshot)
    prefix = f'{PREFIX}Expanded'
    config['motionSnapshot'] = f'{prefix}/snapshot.json'
    config['dynamicVehicleRemoval'] = {
        'version': 2, 'removedCount': count, 'gaussianCount': len(mask) - count,
        'maskSha256': manifest['mask_sha256'], 'manifest': f'{prefix}/manifest.json',
        'highlights': {'F1': f'{prefix}/RGB_highlight.ply'},
        'variants': {'F1': {'rgb': f'{prefix}/RGB_removed.ply', 'rgbWithTrajectory': None}},
        'warning': 'Expanded V015 truck selection. Preview to inspect the selected points. '
                   'Surfaces occluded in the source images can still have gaps.',
    }
    write_json(CONFIG, config)
    print(f'Installed expanded V015 mask: {count:,} points, including {count - int(core.sum()):,} additional points.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expand', action='store_true', help='Build the expanded mask from the installed original trial.')
    args = parser.parse_args()
    if args.expand:
        install_expanded()
    else:
        install()
