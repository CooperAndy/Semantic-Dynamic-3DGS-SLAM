"""Publish a reviewed seq10 calculation as a reversible separate asset version."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from aggregate_vehicle_states import save_json, sha
from install_seq10_tmd_object_demo import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo-root', required=True)
    p.add_argument('--objects-run', required=True)
    p.add_argument('--rgb-dir', required=True)
    args = p.parse_args()
    root, run, rgb = map(Path, [args.demo_root, args.objects_run, args.rgb_dir])
    cfgpath = root / '源码/scene-config.kitti10-tmd-tsdd-s2.json'
    cfg = read(cfgpath)
    old = root / 'scene-data/KITTI10_TMD_TSDD_seed2/ObjectStates'
    prefix = 'scene-data/KITTI10_TMD_TSDD_seed2/ObjectStates_v2'
    asset = root / prefix
    backup = root / '修改记录/20261115_seq10_isolated_compound_review_v2'
    if asset.exists() or backup.exists():
        raise FileExistsError('Review release already exists')
    assert cfg['sceneId'] == 'kitti10_tmd_tsdd_s2'
    assert cfg['motionSnapshot'] == 'scene-data/KITTI10_TMD_TSDD_seed2/ObjectStates/snapshot.json'
    objects = read(run / 'objects.json')
    summary = read(run / 'summary.json')
    review = read(run / 'compound_review.json')
    assert review['geometry_changed'] is False
    for path, digest in {**summary['source_hashes'], **review['checked_prepared_hashes']}.items():
        if sha(path) != digest:
            raise ValueError(f'Changed calculation input: {path}')
    parent = Path(summary['parent_calculation'])
    baseline = {o['object_id']: o for o in read(parent / 'objects.json')}
    changed = set(review['changed_objects'])
    assert changed
    geometry_keys = ['region_index', 'world_anchor', 'world_center', 'anchor_source_gaussian_index',
                     'gaussian_count', 'bbox_world_min', 'bbox_world_max']
    for obj in objects:
        previous = baseline[obj['object_id']]
        assert all(obj[k] == previous[k] for k in geometry_keys)
        if obj['object_id'] not in changed:
            assert obj == previous
    assert sha(run/'gaussian_region_ids.npz') == sha(parent/'gaussian_region_ids.npz')
    old_manifest = read(old / 'manifest.json')
    protected = read(Path(old_manifest['backup']) / 'backup_manifest.json')['protected']
    protected.update({str(q.relative_to(root)): sha(q) for q in old.rglob('*') if q.is_file()})
    protected['源码/scene-config.kitti10-orb-f1-resume796.json'] = sha(root/'源码/scene-config.kitti10-orb-f1-resume796.json')
    backup.mkdir(parents=True)
    before = {}
    for name in ['源码/scene-config.kitti10-tmd-tsdd-s2.json', '使用说明.md']:
        dest = backup / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, dest)
        before[name] = sha(root/name)
    save_json(backup/'backup_manifest.json', dict(before=before, protected=protected, asset_added=prefix))
    shutil.copytree(old, asset)
    for q in run.iterdir():
        if q.is_file():
            shutil.copy2(q, asset/'evidence'/q.name)
    shutil.copy2(Path(__file__), asset/'evidence'/Path(__file__).name)
    script = Path(__file__).parent/'reuse/review_isolated_compounds.py'
    shutil.copy2(script, asset/'evidence'/script.name)
    snapshot = deepcopy(read(old/'snapshot.json'))
    by_id = {o['object_id']: o for o in objects}
    for displayed in snapshot['objects']:
        obj = by_id[displayed['object_id']]
        if obj['object_id'] in changed:
            for key in ['state','final_state','final_reason']:
                displayed[key] = obj[key]
            displayed['aggregate_evidence']['final_reason'] = obj['final_reason']
            displayed['compound_review'] = obj['compound_review']
    counts = dict(Counter(o['final_state'] for o in objects))
    for key in ['object_counts','track_counts','positioned_track_counts']:
        snapshot[key] = counts
    snapshot.update(association_manifest=f'{prefix}/manifest.json',
                    aggregation_config='semantic-region-v1 + isolated-compound-review-v1 + short4_z0.5-40_far20',
                    compound_review_policy=review['policy'])
    snapshot['reference_image'] = snapshot['reference_image'].replace('/ObjectStates/','/ObjectStates_v2/')
    save_json(asset/'snapshot.json', snapshot)
    # A diagnostic reference shows both neighboring regions; it does not replace model scores.
    frame = 436
    rows = [r for r in read(run/'deduplicated_observations.json')
            if r['source_frame'] == frame and by_id[r['object_id']]['final_state'] != 'uncertain']
    im = Image.open(rgb/f'{frame:06d}.png').convert('RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for row in rows:
        label = f"{row['object_id']} | {by_id[row['object_id']]['final_state'].upper()}"
        x,y,x2,y2 = row['bbox']
        color = '#36b6ff' if row['object_id'] != 'V011' else '#ffb54b'
        draw.rectangle((x,y,x2,y2),outline=color,width=2)
        # The label for the rear car sits above its own box, avoiding the foreground label.
        ty = max(0,y-40 if row['object_id']=='V011' else y-20)
        draw.text((max(0,x),ty),label,fill=color,font=font,stroke_width=1,stroke_fill='#101820')
    diagnostic = f'{prefix}/reference/000436_neighbor_review.jpg'
    im.save(root/diagnostic,quality=95)
    manifest = deepcopy(old_manifest)
    manifest.update(summary=summary, backup=str(backup), compound_review='evidence/compound_review.json',
                    diagnostic_reference_image=diagnostic, original_release_preserved=True,
                    limitation='Isolated conflict review is model consistency evidence; no object ground truth evaluation.')
    save_json(asset/'manifest.json',manifest)
    cfg.update(motionSnapshot=f'{prefix}/snapshot.json',motionStateManifest=f'{prefix}/manifest.json',
               motionStateConfig=snapshot['aggregation_config'])
    save_json(cfgpath,cfg)
    guide = root/'使用说明.md'
    content = guide.read_text(encoding='utf-8')
    content = content.replace('displays 11 Static plus one Dynamic region; four Uncertain regions are hidden.',
                              'displays 12 Static plus one Dynamic region; three Uncertain regions are hidden. An isolated compound alarm was explained by later whole-instance mask/depth evidence; neighboring V010 and V011 now both have Static labels. The original ObjectStates release is retained; the active release is ObjectStates_v2.')
    guide.write_text(content,encoding='utf-8')
    assert all(sha(root/name)==digest for name,digest in protected.items())
    receipt = dict(scene=cfg['sceneId'], state_counts=counts, changed_objects=sorted(changed),
                   shown_regions=sum(v for k,v in counts.items() if k!='uncertain'),
                   geometry_changed=False, old_release_preserved=True, new_seq10_scene_modified=False,
                   backup=str(backup), diagnostic_reference_image=diagnostic)
    save_json(asset/'installation_receipt.json',receipt)
    save_json(backup/'installed_state.json',dict(files={name:sha(root/name) for name in before}))
    print(receipt)


if __name__=='__main__':
    main()
