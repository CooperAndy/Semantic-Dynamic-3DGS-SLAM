"""Review isolated compound alarms using later whole-instance mask/depth evidence.

Writes a separate calculation; does not alter the baseline clustering or Demo.
This is prediction consistency evidence, not physical-object ground truth.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from aggregate_vehicle_states import Config, decide, rejects, save_json, sha
from fuse_reference_seq09 import project, read

POLICY = dict(name='isolated_compound_whole_instance_confirmation_v1',
              max_conflict_frames=1, min_confirmations=3, sample_gap=4,
              min_confirmation_span=8, min_detection_confidence=.8,
              min_projected_coverage=.7)


def run(args):
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    summary = read(source / 'summary.json')
    for name, digest in summary['source_hashes'].items():
        if sha(name) != digest:
            raise ValueError(f'Changed baseline input: {name}')
    resolved = read(source / 'resolved_profile.json')
    paths = {k: Path(v) for k, v in resolved['resolved_paths'].items()}
    profile = resolved['profile']
    motion = Config(**profile['parameters']['motion'])
    region = profile['parameters']['region']
    inp = paths['instance_inputs']
    input_manifest = read(inp / 'manifest.json')
    entries = {e['source_frame']: e for e in input_manifest['frames']}
    K = np.asarray(input_manifest['intrinsics'])
    poses = {r['source_frame']: r for r in read(paths['camera_poses'])}
    observations = {(r['source_frame'], r['track_id']): r for r in read(paths['observation_run'] / 'observations.json')}
    audit = read(source / 'associations.json')
    objects = deepcopy(read(source / 'objects.json'))
    dedup = read(source / 'deduplicated_observations.json')
    reviews, changed, checked_hashes = [], [], {}
    for obj in objects:
        conflicts = obj['compound_conflicts']
        if not conflicts:
            continue
        review = dict(object_id=obj['object_id'], original_conflicts=conflicts,
                      resolution='retained_uncertain', explanations=[])
        reviews.append(review)
        conflict_frames = {c['frame'] for c in conflicts}
        if len(conflict_frames) > POLICY['max_conflict_frames']:
            review['reason'] = 'repeated_compound_alarm'
            continue
        rows = [r for r in audit if r['object_id'] == obj['object_id']
                and not rejects(observations[r['source_frame'], r['track_id']], motion)]
        all_resolved = True
        for conflict in conflicts:
            centers = []
            for track in conflict['tracks']:
                row = next(r for r in rows if r['source_frame'] == conflict['frame'] and r['track_id'] == track)
                centers.append(next(c['support_world_center'] for c in row['candidates']
                                    if c['region_index'] == obj['region_index']))
            confirmations = []
            for row in rows:
                n = row['source_frame']
                # Review only later observations in this existing object's trajectory.
                if n <= max(conflict_frames) or n in conflict_frames:
                    continue
                candidate = next(c for c in row['candidates'] if c['region_index'] == obj['region_index'])
                if (row['detection_confidence'] < POLICY['min_detection_confidence'] or
                        candidate['projected_coverage'] < POLICY['min_projected_coverage']):
                    continue
                entry = entries[n]
                files = [('mask_sha256', inp / f'instances/{n:06d}_masks.npz'),
                         ('instance_sha256', inp / f'instances/{n:06d}.json'),
                         ('masked_depth_sha256', inp / f'masked_depth/{n:06d}.npz')]
                for key, path in files:
                    digest = sha(path)
                    if digest != entry[key]:
                        raise ValueError(f'Changed prepared input: {path}')
                    checked_hashes[str(path)] = digest
                meta = read(files[1][1])
                instance = next(o for o in meta['instances'] if o['track_id'] == row['track_id'])
                with np.load(files[0][1], allow_pickle=False) as z:
                    mask = z['masks'][instance['mask_index']]
                with np.load(files[2][1], allow_pickle=False) as z:
                    depth = z['depth']
                u, v, z = project(np.asarray(centers), poses[n], K)
                visible = ((z >= motion.depth_min) & (z <= motion.depth_max) &
                           (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0]))
                if not visible.all() or not mask[v, u].all():
                    continue
                measured = depth[v, u]
                tolerance = region['depth_abs_tolerance_m'] + region['depth_relative_tolerance'] * z
                if not ((measured > 0) & (np.abs(measured - z) <= tolerance)).all():
                    continue
                confirmations.append(dict(frame=n, track_id=row['track_id'],
                                          projected_coverage=candidate['projected_coverage'],
                                          detection_confidence=row['detection_confidence'],
                                          depth_residual_m=np.abs(measured-z).tolist()))
            # One confirmation per separated frame; duplicate IDs cannot inflate support.
            sampled = []
            for c in sorted(confirmations, key=lambda r: (r['frame'], -r['projected_coverage'])):
                if not sampled or c['frame'] - sampled[-1]['frame'] >= POLICY['sample_gap']:
                    sampled.append(c)
            explained = (len(sampled) >= POLICY['min_confirmations'] and
                         sampled[-1]['frame'] - sampled[0]['frame'] >= POLICY['min_confirmation_span'])
            all_resolved &= explained
            review['explanations'].append(dict(conflict=conflict, support_centers=centers,
                                               confirmations=confirmations, sampled_confirmations=sampled,
                                               explained=bool(explained)))
        if all_resolved:
            evidence = [r for r in dedup if r['object_id'] == obj['object_id']]
            result = decide(evidence, motion, profile.get('enable_static_hold', True))
            if result['final_state'] != 'uncertain':
                review.update(resolution='isolated_alarm_explained_by_later_whole_instance',
                              restored_state=result['final_state'])
                obj.update(final_state=result['final_state'],
                           state={'static': '\u9759\u6001', 'dynamic': '\u52a8\u6001'}[result['final_state']],
                           final_reason=result['final_reason'], compound_review=review)
                changed.append(obj['object_id'])
    out.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, out / path.name)
    save_json(out / 'objects.json', objects)
    save_json(out / 'compound_review.json', dict(policy=POLICY, reviews=reviews, changed_objects=changed,
              checked_prepared_hashes=checked_hashes, geometry_changed=False,
              limitation='An isolated alarm can be explained by repeated model agreement. This does not prove unique physical identity; sustained or unexplained alarms stay uncertain.'))
    summary.update(object_state_counts=dict(Counter(o['final_state'] for o in objects)),
                   compound_review_policy=POLICY, compound_review_changed_objects=changed,
                   parent_calculation=str(source), demo_installed=False)
    for path in [source / 'objects.json', source / 'associations.json', source / 'summary.json', Path(__file__).resolve(), paths['camera_poses']]:
        summary['source_hashes'][str(path)] = sha(path)
    # The original CSV remains in the parent calculation; write a consistent revised CSV.
    from aggregate_vehicle_states import save_csv
    save_csv(out / 'object_summary.csv', [{k: v for k, v in o.items()
             if k not in ('world_anchor','world_center','bbox_world_min','bbox_world_max','compound_review')} for o in objects])
    save_json(out / 'summary.json', summary)
    print(json.dumps(dict(changed_objects=changed, state_counts=summary['object_state_counts'],
                         reviews=reviews), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
