"""Install fixed old-seq09 track labels into T07, with a restorable backup.

The newer C50S42 scene and all original motion/map assets remain unchanged.
Run aggregation first. This installer does not select parameters using truth.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median

from PIL import Image, ImageDraw, ImageFont

OLD_ID = 'kitti09_orb_f200_full'
NEW_ID = 'kitti09_conf050_seed42'
STATE_TEXT = {'static': '静态', 'dynamic': '动态', 'uncertain': '不确定'}
COLORS = {'static': '#368cf5', 'dynamic': '#f28c28', 'uncertain': '#838b98'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo-root', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--reference-images', required=True, help='Original unannotated seq09 RGB JPEGs keyed by source frame')
    p.add_argument('--backup-name', default='20261114_seq09_fixed_states_v1')
    a = p.parse_args()
    demo, run, images = Path(a.demo_root).resolve(), Path(a.run_dir).resolve(), Path(a.reference_images).resolve()
    code, assets = demo/'源码', demo/'scene-data'
    backup = demo/'修改记录'/a.backup_name
    target = assets/'KITTI09_FixedStates'
    if backup.exists() or target.exists(): raise FileExistsError('Backup or release already exists; restore/review before reinstalling.')
    original = assets/'KITTI09_MotionDemo'
    manifest = read(run/'input_manifest.json')
    if Path(manifest['source']).resolve() != (original/'frames').resolve(): raise ValueError('Run must use the ORIGINAL map motion snapshots')
    for item in manifest['frames']:
        if sha(original/'frames'/item['name']) != item['sha256']: raise ValueError('Input predictions changed since aggregation')
    old_config = read(code/'scene-config.kitti09-orb-f200.json')
    if old_config['sceneId'] != OLD_ID: raise ValueError('Unexpected scene ID')
    evidence = read(run/'observations.json')
    predictions = read(run/'fixed_tracks.json')
    config_name = read(run/'run_summary.json')['default_config']
    groups = defaultdict(list)
    for row in evidence: groups[row['track_segment_id']].append(row)
    if len({v['track_id'] for v in predictions.values()}) != len(predictions):
        raise ValueError('Multiple segments share a track ID; resolve map identities before this installation')
    source_snapshots = {int(p.stem): read(p) for p in sorted((original/'frames').glob('*.json'))}
    source_objects = {(n, o['track_id']): o for n, s in source_snapshots.items() for o in s['objects']}
    source_audit = read(images.parent/'source_audit.json')
    audited_images = {r['source_frame']: r['review_jpeg_sha256'] for r in source_audit['images']}
    for n in source_snapshots:
        if sha(images/f'{n:06d}.jpg') != audited_images[n]: raise ValueError('Review image hash mismatch')
    # Reuse no new-map anchors: every representative is taken from this old map.
    objects, representative_frames = [], {}
    for key, result in predictions.items():
        rows = groups[key]
        anchored = [r for r in rows if source_objects[(r['source_frame'], r['track_id'])].get('display_estimate') is True]
        valid_anchored = [r for r in anchored if r['eligible']]
        candidates = valid_anchored or anchored
        if candidates:
            if result['E_state'] == 'static':
                center = [median(source_objects[(r['source_frame'], r['track_id'])]['viewer_xyz'][i] for r in candidates) for i in range(3)]
                chosen = min(candidates, key=lambda r: sum((source_objects[(r['source_frame'],r['track_id'])]['viewer_xyz'][i]-center[i])**2 for i in range(3)))
            else:
                middle = median(r['source_frame'] for r in candidates)
                chosen = min(candidates, key=lambda r: abs(r['source_frame']-middle))
        else: chosen = rows[len(rows)//2]
        obj = deepcopy(source_objects[(chosen['source_frame'], chosen['track_id'])])
        obj.update(source_track_id=obj['track_id'], track_segment_id=key, representative_source_frame=chosen['source_frame'],
                   source_raw_score=obj.get('score'), source_raw_display_state=obj.get('state'),
                   final_state=result['E_state'], state=STATE_TEXT[result['E_state']],
                   score=result['score_median'], fixed_state=True, static_hold_applied=result['hold'],
                   aggregate_evidence=dict(first_frame=result['first_frame'], last_frame=result['last_frame'],
                                           observations=result['observations'], valid_count=result['valid_count'],
                                           sampled_frames=result['sampled_frames'], final_reason=result['final_reason'],
                                           far_static_frames=result['far_static_frames'], later_dynamic_frames=result['later_dynamic_frames']),
                   identity_status='unverified_track_segment', anchor_verified=False)
        if not candidates:
            obj.update(display_estimate=False, viewer_xyz=None, world_xyz=None, anchor_status='no_valid_3d_anchor_in_track_segment')
        objects.append(obj)
        representative_frames[key] = chosen['source_frame']
    default_frame = int(old_config.get('defaultCameraFrame', 399))
    snapshot = deepcopy(source_snapshots[default_frame])
    snapshot.update(motion_model='Pi3MOS + fixed track aggregation', window=[],
                    fixed_track_labels=True, label_mode='offline_fixed_track_segments', source_frames=[300,499],
                    aggregation_config=config_name, objects=objects,
                    reference_image=f'scene-data/KITTI09_FixedStates/reference/{default_frame:06d}.jpg',
                    history_note='One fixed label per track segment, using the complete 300-499 interval and a far-static hold rule. Offline output uses future observations relative to early frames.',
                    position_note='One representative observed anchor per track segment on the original map. No cross-ID identity merge, verified Gaussian binding, or occlusion handling.',
                    accuracy_status='NOT_EVALUATED_NO_HUMAN_LABELS',
                    reference_image_note='Illustrative observation only; all track-segment labels are aggregated over the complete interval.')
    visible = [o for o in objects if o.get('display_estimate')]
    snapshot['track_counts'] = dict(Counter(o['final_state'] for o in objects))
    snapshot['positioned_track_counts'] = dict(Counter(o['final_state'] for o in visible))
    snapshot['positioned_track_count'] = len(visible)
    snapshot['unpositioned_track_count'] = len(objects)-len(visible)
    backup.mkdir(parents=True, exist_ok=False)
    backup_files = ['源码/scene-catalog.json','源码/scene-config.kitti09-orb-f200.json',
                    '源码/interactive_viewer.html','源码/t07.js','使用说明.md']
    before = {}
    for relative in backup_files:
        path = demo/relative
        dest = backup/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); before[relative] = sha(path)
    new_config_path = code/'scene-config.kitti09-c50-s42.json'
    new_config_hash = sha(new_config_path)
    # Capture small, authoritative manifests for the unmodified maps/new scene.
    protected = [new_config_path, assets/'KITTI09_C50S42/KITTI09_MotionDemo/index.json',
                 assets/'KITTI09_C50S42/KITTI09_MotionDemo/snapshot.json',
                 assets/'KITTI09_F1/KITTI09_F1_manifest.json', assets/'KITTI09_F200/KITTI09_F200_manifest.json']
    protected.extend((assets/'KITTI09_C50S42/KITTI09_MotionDemo/frames').glob('*.json'))
    protected_hashes = {str(p.relative_to(demo)): sha(p) for p in protected}
    write(backup/'backup_manifest.json', dict(before=before, protected=protected_hashes, asset_added='scene-data/KITTI09_FixedStates'))
    target.mkdir(); (target/'reference').mkdir(); (target/'evidence').mkdir()
    write(target/'snapshot.json', snapshot)
    write(target/'tracks.json', predictions)
    # The sidebar image uses the same fixed labels as the map, on a single illustrative original RGB frame.
    im = Image.open(images/f'{default_frame:06d}.jpg').convert('RGB')
    draw = ImageDraw.Draw(im); font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    by_detection = {(r['source_frame'],r['track_id']): r['track_segment_id'] for r in evidence}
    for obj in source_snapshots[default_frame]['objects']:
        key = by_detection[default_frame,obj['track_id']]; state = predictions[key]['E_state']
        box = obj['bbox']; draw.rectangle(box, outline=COLORS[state], width=2)
        label = f"Track #{obj['track_id']} | {state.upper()} | fixed"
        x, y = box[0], max(0, box[1]-18)
        width = draw.textbbox((0,0),label,font=font)[2]+6
        draw.rectangle((x,y,min(im.width,x+width),y+18),fill='#132035'); draw.text((x+2,y),label,font=font,fill=COLORS[state])
    im.save(target/f'reference/{default_frame:06d}.jpg', quality=93)
    for name in ['input_manifest.json','run_summary.json','configs.json','candidate_tracks.csv','sweep_summary.csv','fixed_tracks.csv','observations.csv','labels_template.csv']:
        shutil.copy2(run/name, target/'evidence'/name)
    shutil.copy2(Path(__file__).with_name('aggregate_vehicle_states.py'),target/'evidence/aggregate_vehicle_states.py')
    write(target/'manifest.json', dict(method='offline_fixed_track_segments_with_far_static_hold', config=config_name,
                                       parameters=read(run/'configs.json')[config_name],
                                       source_map_sha256=manifest['source_map_sha256'],
                                       source_frames=[300,499], source_snapshot_hashes=manifest['frames'],
                                       track_count=len(objects), positioned_track_count=len(visible),
                                       track_counts=snapshot['track_counts'], positioned_track_counts=snapshot['positioned_track_counts'],
                                       static_hold_tracks=[o['track_id'] for o in objects if o['static_hold_applied']],
                                       representative_frames=representative_frames,
                                       adoption='USER_REQUESTED_DEMO_TRIAL_NOT_ACCURACY_ACCEPTANCE',
                                       identity='One marker per track segment; physical vehicle identity across track IDs is unverified.',
                                       backup=str(backup)))
    old_config.update(title='KITTI 09 · Original map · Fixed Vehicle States',
                      motionIndex=None, motionSnapshot='scene-data/KITTI09_FixedStates/snapshot.json',
                      motionReferenceImageTemplate=None,
                      motionLabelMode='offline_fixed_track_segments',
                      motionStateManifest='scene-data/KITTI09_FixedStates/manifest.json',
                      motionStateConfig=config_name)
    catalog = read(code/'scene-catalog.json')
    for scene in catalog['scenes']:
        if scene['id'] == OLD_ID: scene['label'] = 'Seq 09 · 000300–000499 · Fixed States'
    write(code/'scene-config.kitti09-orb-f200.json', old_config)
    write(code/'scene-catalog.json', catalog)
    assert all(sha(demo/p)==h for p,h in protected_hashes.items())
    assert sha(new_config_path) == new_config_hash
    assert all(sha(original/'frames'/item['name']) == item['sha256'] for item in manifest['frames'])
    receipt = dict(scene=OLD_ID, config=config_name, replacement='TSDD entry replaced with Pi3MOS fixed track aggregation',
                   track_counts=snapshot['track_counts'], positioned_track_counts=snapshot['positioned_track_counts'],
                   positioned_tracks=len(visible), unpositioned_tracks=len(objects)-len(visible),
                   static_hold_tracks=[o['track_id'] for o in objects if o['static_hold_applied']],
                   original_motion_unchanged=True, new_scene_unchanged=True, map_assets_modified=False,
                   backup=str(backup), accuracy_evaluated=False)
    write(target/'installation_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
