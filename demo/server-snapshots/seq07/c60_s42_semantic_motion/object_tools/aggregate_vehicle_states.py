"""Offline fixed track-segment labels; Python standard library only.

Outputs are predictions, not ground truth or verified physical map objects.
Existing inputs and Demo assets are never modified.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

VERSION = 'fixed-track-static-hold-v1'
STATES = ('static', 'dynamic', 'uncertain')


@dataclass(frozen=True)
class Config:
    depth_min: float = .5
    depth_max: float = 40.
    min_valid_pixels: int = 100
    min_valid_ratio: float = .5
    sample_gap: int = 8
    min_samples: int = 3
    static_threshold: float = .4
    dynamic_threshold: float = .6
    far_min: float = 20.
    hold_min_observations: int = 3
    hold_max_gap: int = 3
    hold_min_span: int = 2
    min_approach_m: float = 3.
    min_later_dynamic: int = 2
    segment_gap: int = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def rejects(row, cfg):
    reasons = []
    if row['category'] not in ('car', 'truck', 'bus', 'van'): reasons.append('non_vehicle')
    if not row['history_complete'] or row['raw_state'] == 'unknown': reasons.append('missing_history')
    if not finite(row['raw_score']) or not 0 <= row['raw_score'] <= 1: reasons.append('invalid_score')
    z = row['depth_z_m']
    if not finite(z) or not cfg.depth_min <= z <= cfg.depth_max: reasons.append('depth_outside_gate')
    if row['valid_depth_pixels'] < cfg.min_valid_pixels: reasons.append('too_few_depth_pixels')
    if row['valid_depth_ratio'] < cfg.min_valid_ratio: reasons.append('low_depth_ratio')
    if row['identity_conflict']: reasons.append('identity_conflict')
    return reasons


def sample_rows(rows, gap):
    result = []
    for row in rows:
        if not result or row['source_frame'] - result[-1]['source_frame'] >= gap:
            result.append(row)
    return result


def summarize_score(samples, cfg):
    if len(samples) < cfg.min_samples: return 'uncertain', None, 'insufficient_samples'
    score = median(r['raw_score'] for r in samples)
    if score <= cfg.static_threshold: return 'static', score, 'whole_segment_score'
    if score >= cfg.dynamic_threshold: return 'dynamic', score, 'whole_segment_score'
    return 'uncertain', score, 'score_in_uncertain_band'


def static_hold(eligible, cfg):
    """Require an early far static run BEFORE any eligible dynamic observation.

    Consecutive observations share model windows; they are not independent votes.
    Depth approach is checked with both a net drop and a robust temporal slope.
    """
    run = []
    candidates = []
    for row in eligible:
        if row['raw_score'] >= .5:
            break
        if row['depth_z_m'] >= cfg.far_min and row['raw_score'] <= cfg.static_threshold:
            if run and row['source_frame'] - run[-1]['source_frame'] > cfg.hold_max_gap:
                run = []
            run.append(row)
            if len(run) >= cfg.hold_min_observations and run[-1]['source_frame'] - run[0]['source_frame'] >= cfg.hold_min_span:
                candidates.append(list(run))
        else:
            run = []
    for evidence in candidates:
        far_z = median(r['depth_z_m'] for r in evidence)
        later = [r for r in eligible if r['source_frame'] > evidence[-1]['source_frame'] and r['raw_score'] >= .5 and r['depth_z_m'] <= far_z - cfg.min_approach_m]
        if len(later) < cfg.min_later_dynamic: continue
        trend_rows = [r for r in eligible if evidence[0]['source_frame'] <= r['source_frame'] <= later[-1]['source_frame']]
        slopes = [(b['depth_z_m']-a['depth_z_m'])/(b['source_frame']-a['source_frame']) for i, a in enumerate(trend_rows) for b in trend_rows[i+1:]]
        slope = median(slopes) if slopes else 0
        if slope < 0:
            return dict(applied=True, far_static_frames=[r['source_frame'] for r in evidence], later_dynamic_frames=[r['source_frame'] for r in later], far_depth_median=far_z, depth_slope_m_per_frame=slope)
    return dict(applied=False, far_static_frames=[], later_dynamic_frames=[])


def decide(rows, cfg, hold_enabled):
    valid = [r for r in rows if not rejects(r, cfg)]
    sampled = sample_rows(valid, cfg.sample_gap)
    state, score, reason = summarize_score(sampled, cfg)
    hold = static_hold(valid, cfg) if hold_enabled else dict(applied=False, far_static_frames=[], later_dynamic_frames=[])
    if hold['applied']:
        state, reason = 'static', 'far_static_then_near_dynamic_hold'
    return dict(final_state=state, final_reason=reason, score_median=score,
                valid_count=len(valid), sample_count=len(sampled),
                sampled_frames=[r['source_frame'] for r in sampled], static_hold=hold)


def load_frames(directory, cfg):
    paths = sorted(Path(directory).glob('*.json'))
    frames, rows, manifest = [], [], []
    last_by_track, segment_number = {}, Counter()
    map_hashes = set()
    for path in paths:
        frame = json.loads(path.read_text(encoding='utf-8-sig'))
        n = int(frame['reference_frame'])
        if path.stem != f'{n:06d}': raise ValueError(f'Frame filename mismatch: {path}')
        if frames and n != frames[-1]['reference_frame'] + 1: raise ValueError('Non-contiguous input frames')
        map_hashes.add(frame['source_map_sha256'])
        seen = set()
        window = frame.get('window', [])
        history = window == list(range(n-7, n+1))
        for obj in frame['objects']:
            track = int(obj['track_id'])
            if track in seen: raise ValueError(f'Duplicate track {track} in frame {n}')
            seen.add(track)
            prior = last_by_track.get(track)
            if prior and (n-prior[0] > cfg.segment_gap or obj['category'] != prior[1]): segment_number[track] += 1
            key = f'{track}:{segment_number[track]}'
            last_by_track[track] = (n, obj['category'])
            pixels, area = obj.get('valid_depth_pixels', 0), obj.get('mask_area', 0)
            if not finite(pixels) or not finite(area) or pixels < 0 or area < pixels: raise ValueError('Invalid depth/mask pixel counts')
            for v in (obj.get('score'), obj.get('depth_m')):
                if v is not None and not finite(v): raise ValueError('Non-finite numeric input')
            rows.append(dict(source_frame=n, track_id=track, track_segment_id=key,
                             category=obj['category'], bbox=obj['bbox'], raw_score=obj.get('score'),
                             raw_state=obj['raw_state'], depth_z_m=obj.get('depth_m'),
                             valid_depth_pixels=pixels, mask_area=area, valid_depth_ratio=pixels/max(1, area),
                             history_complete=history, identity_conflict=bool(obj.get('identity_conflict', False)),
                             edge_truncated=obj['bbox'][0] <= 0 or obj['bbox'][1] <= 0 or obj['bbox'][2] >= frame['image_width'] or obj['bbox'][3] >= frame['image_height']))
        frames.append(frame)
        manifest.append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    if not frames: raise ValueError('No input snapshots')
    if len(map_hashes) != 1: raise ValueError('Mixed source maps')
    return frames, rows, manifest


def run(args):
    cfg = Config()
    frames, rows, manifest = load_frames(args.frames, cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for row in rows: groups[row['track_segment_id']].append(row)
    summary_rows, candidates, fixed = [], [], {}
    # Predeclared exploratory grid: no automatic winner selection without truth.
    policies = [('strict8', 8, 3, .4, .6), ('short4', 4, 2, .4, .6), ('narrow4', 4, 2, .45, .55)]
    configs = []
    for policy, gap, minimum, lo, hi in policies:
        for depth_min, depth_max in [(.5, 40.), (5., 40.), (10., 30.), (15., 30.)]:
            for far_min in [15., 20., 25.]:
                c = replace(cfg, sample_gap=gap, min_samples=minimum, static_threshold=lo,
                            dynamic_threshold=hi, depth_min=depth_min, depth_max=depth_max, far_min=far_min)
                name = f'{policy}_z{depth_min:g}-{depth_max:g}_far{far_min:g}'
                configs.append((name, c))
    for name, c in configs:
        results = {}
        for key, observations in groups.items():
            d, e = decide(observations, c, False), decide(observations, c, True)
            record = dict(config=name, track_segment_id=key, track_id=observations[0]['track_id'],
                          first_frame=observations[0]['source_frame'], last_frame=observations[-1]['source_frame'],
                          observations=len(observations), D_state=d['final_state'], E_state=e['final_state'],
                          final_reason=e['final_reason'], valid_count=e['valid_count'], sample_count=e['sample_count'],
                          score_median=e['score_median'], sampled_frames=e['sampled_frames'],
                          hold=e['static_hold']['applied'], far_static_frames=e['static_hold']['far_static_frames'],
                          later_dynamic_frames=e['static_hold']['later_dynamic_frames'],
                          identity_status='unverified_track_segment')
            candidates.append(record)
            results[key] = record
        dcounts = Counter(v['D_state'] for v in results.values())
        ecounts = Counter(v['E_state'] for v in results.values())
        in_gate = {k for k, v in groups.items() if any(finite(r['depth_z_m']) and c.depth_min <= r['depth_z_m'] <= c.depth_max for r in v)}
        summary_rows.append(dict(config=name, track_segments=len(groups),
                                 segments_with_depth_in_gate=len(in_gate),
                                 segments_with_eligible_observations=sum(v['valid_count'] > 0 for v in results.values()),
                                 **{f'D_{s}': dcounts[s] for s in STATES},
                                 **{f'E_{s}': ecounts[s] for s in STATES},
                                 **{f'E_in_gate_{s}': sum(results[k]['E_state'] == s for k in in_gate) for s in STATES},
                                 hold_triggered=sum(v['hold'] for v in results.values()),
                                 changed_D_to_E=sum(v['D_state'] != v['E_state'] for v in results.values()),
                                 E_decided_fraction=1-ecounts['uncertain']/len(groups)))
        if name == args.fixed_config: fixed = results
    if not fixed: raise ValueError(f'Unknown fixed config: {args.fixed_config}')
    fixed_cfg = dict(configs)[args.fixed_config]
    save_csv(out/'sweep_summary.csv', summary_rows)
    save_csv(out/'candidate_tracks.csv', candidates)
    save_json(out/'configs.json', {name: asdict(c) for name, c in configs})
    for row in rows:
        why = rejects(row, fixed_cfg)
        row.update(eligible=not why, reject_reason=why,
                   final_state=fixed[row['track_segment_id']]['E_state'],
                   final_reason=fixed[row['track_segment_id']]['final_reason'])
    save_csv(out/'observations.csv', rows)
    save_json(out/'observations.json', rows)
    save_csv(out/'fixed_tracks.csv', fixed.values())
    save_json(out/'fixed_tracks.json', fixed)
    template = [dict(track_segment_id=k, true_object_id='', true_state='', identity_ok='', split_or_merge_notes='', reviewer='', evidence_frames='', notes='') for k in groups]
    save_csv(out/'labels_template.csv', template)
    save_json(out/'input_manifest.json', dict(source=str(Path(args.frames).resolve()), source_map_sha256=frames[0]['source_map_sha256'], frames=manifest))
    snapshot_dir = out/'fixed_snapshots'; snapshot_dir.mkdir()
    for frame in frames:
        for obj in frame['objects']:
            key = next(r['track_segment_id'] for r in rows if r['source_frame'] == frame['reference_frame'] and r['track_id'] == obj['track_id'])
            result = fixed[key]
            obj.update(track_segment_id=key, final_state=result['E_state'], final_reason=result['final_reason'], offline_fixed_label=True)
        frame['label_scope'] = 'Offline fixed track-segment predictions; raw_state/state remain original; map object identity unverified.'
        save_json(snapshot_dir/f"{frame['reference_frame']:06d}.json", frame)
    # Hash inputs again to prove the experiment did not alter them.
    assert all(sha(Path(args.frames)/p['name']) == p['sha256'] for p in manifest)
    report = dict(version=VERSION, script_sha256=sha(__file__), frames=len(frames),
                  source_frame_range=[frames[0]['reference_frame'], frames[-1]['reference_frame']],
                  observations=len(rows), raw_state_counts=dict(Counter(r['raw_state'] for r in rows)),
                  unique_track_ids=len({r['track_id'] for r in rows}), track_segments=len(groups),
                  short_segments_1_to_4=sum(len(v) <= 4 for v in groups.values()),
                  default_config=args.fixed_config, default_summary=next(r for r in summary_rows if r['config'] == args.fixed_config),
                  selected_config_status='DIAGNOSTIC_REFERENCE_NOT_ACCEPTED_FOR_DEMO',
                  default_reject_counts=dict(Counter(x for r in rows for x in r['reject_reason'])),
                  source_unchanged=True, accuracy_status='NOT_EVALUATED_NO_HUMAN_LABELS',
                  scope='Exploratory configurations; no automatic best choice. Track IDs are not physical vehicles. Fixed snapshots do not change original raw_state/state or map assets.')
    save_json(out/'run_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate(args):
    rows = read_csv(args.labels)
    labels = {}
    for r in rows:
        key = r['track_segment_id']
        if key in labels: raise ValueError(f'Duplicate label: {key}')
        if r.get('true_state') not in ('', 'static', 'dynamic', 'uncertain', 'state_change'): raise ValueError('Invalid true_state')
        labels[key] = r
    predictions = read_csv(args.candidates)
    keys = {r['track_segment_id'] for r in predictions}
    if set(labels)-keys: raise ValueError('Label contains unknown segment IDs')
    by_config = defaultdict(list)
    for r in predictions: by_config[r['config']].append(r)
    output = []
    for config, pred in by_config.items():
        for group in ('D', 'E'):
            matched = [(p, labels.get(p['track_segment_id'], {})) for p in pred]
            scored = [(p, l) for p, l in matched if l.get('true_state') in ('static', 'dynamic') and l.get('identity_ok') == 'yes' and l.get('true_object_id') and l.get('reviewer') and l.get('evidence_frames')]
            confusion = Counter((l['true_state'], p[f'{group}_state']) for p, l in scored)
            tp, fp = confusion['dynamic', 'dynamic'], confusion['static', 'dynamic']
            fn = confusion['dynamic', 'static'] + confusion['dynamic', 'uncertain']
            decided = sum(p[f'{group}_state'] != 'uncertain' for p, _ in scored)
            correct = sum(p[f'{group}_state'] == l['true_state'] for p, l in scored)
            true_static = sum(l['true_state'] == 'static' for _, l in scored)
            true_dynamic = len(scored)-true_static
            output.append(dict(config=config, group=group, total_segments=len(pred), scored_segments=len(scored),
                               unique_true_objects=len({l['true_object_id'] for _, l in scored}),
                               unscored_segments=len(pred)-len(scored),
                               decided_coverage=decided/len(scored) if scored else None,
                               decided_accuracy=correct/decided if decided else None,
                               static_as_dynamic_rate=fp/true_static if true_static else None,
                               dynamic_as_static_rate=confusion['dynamic','static']/true_dynamic if true_dynamic else None,
                               dynamic_precision=tp/(tp+fp) if tp+fp else None,
                               dynamic_recall=tp/(tp+fn) if tp+fn else None,
                               dynamic_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                               confusion={f'{a}_to_{b}': confusion[a,b] for a in ('static','dynamic') for b in STATES},
                               scope='Track-segment diagnostic only; physical-object weighting and held-out split require verified identity grouping. Dynamic uncertain counts as FN.'))
    target = Path(args.output); target.mkdir(parents=True, exist_ok=False)
    save_csv(target/'evaluation.csv', output)
    save_json(target/'evaluation.json', output)
    print(json.dumps(dict(scored=max(r['scored_segments'] for r in output), output=str(target))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('run'); p.add_argument('--frames', required=True); p.add_argument('--output', required=True)
    p.add_argument('--fixed-config', default='strict8_z0.5-40_far20', help='Diagnostic reference; choose a different frozen config explicitly after review.')
    p = subs.add_parser('evaluate'); p.add_argument('--candidates', required=True); p.add_argument('--labels', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else evaluate)(args)


if __name__ == '__main__': main()
