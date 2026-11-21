"""Install the old seq09 semantic-object label trial; retain the previous track trial."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageFont
from aggregate_vehicle_states import save_json,sha


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--demo-root',required=True)
    p.add_argument('--objects-run',required=True);p.add_argument('--rgb-dir',required=True)
    args=p.parse_args();root,run,rgb=map(Path,[args.demo_root,args.objects_run,args.rgb_dir])
    code=root/'源码';asset=root/'scene-data/KITTI09_ObjectStates'
    backup=root/'修改记录/20261114_seq09_semantic_objects_v1'
    if asset.exists() or backup.exists():raise FileExistsError('Semantic object release already exists')
    objects=read(run/'objects.json');summary=read(run/'summary.json');parameters=read(run/'config.json')
    for path,expected in summary['source_hashes'].items():
        if sha(path)!=expected:raise ValueError(f'Input/source changed: {path}')
    cfg=read(code/'scene-config.kitti09-orb-f200.json')
    protected=[code/'scene-config.kitti09-c50-s42.json',root/'scene-data/KITTI09_C50S42/KITTI09_MotionDemo/index.json',root/'scene-data/KITTI09_C50S42/KITTI09_MotionDemo/snapshot.json']
    protected+=list((root/'scene-data/KITTI09_C50S42/KITTI09_MotionDemo/frames').glob('*.json'))
    protected+=list((root/'scene-data/KITTI09_MotionDemo/frames').glob('*.json'))
    protected_hashes={str(p.relative_to(root)):sha(p) for p in protected}
    backup.mkdir(parents=True,exist_ok=False)
    before={}
    for name in ['源码/interactive_viewer.html','源码/t07.js','源码/scene-catalog.json','源码/scene-config.kitti09-orb-f200.json','使用说明.md']:
        dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest);before[name]=sha(root/name)
    save_json(backup/'backup_manifest.json',dict(before=before,protected=protected_hashes,asset_added='scene-data/KITTI09_ObjectStates'))
    asset.mkdir();(asset/'reference').mkdir();(asset/'evidence').mkdir()
    snapshot=deepcopy(read(root/'scene-data/KITTI09_FixedStates/snapshot.json'))
    display=[]
    for obj in objects:
        anchor=obj['world_anchor']
        display.append(dict(track_id=obj['region_index']+1,object_id=obj['object_id'],category=obj['category'],
                            score=obj['score'],state=obj['state'],final_state=obj['final_state'],
                            final_reason=obj['final_reason'],source_track_ids=obj['source_track_ids'],
                            display_estimate=True,world_xyz=anchor,viewer_xyz=[-anchor[0],-anchor[1],anchor[2]],
                            fixed_state=True,anchor_verified=False,semantic_anchor_bound=True,
                            anchor_source_gaussian_index=obj['anchor_source_gaussian_index'],
                            anchor_status='actual_gaussian_in_predicted_semantic_vehicle_region',
                            gaussian_region_index=obj['region_index'],region_gaussian_count=obj['gaussian_count'],
                            identity_status=obj['identity_status'],
                            aggregate_evidence=dict(first_frame=obj['first_frame'],last_frame=obj['last_frame'],
                                                    valid_count=obj['valid_observation_count'],
                                                    unique_observation_frames=obj['unique_observation_frames'],
                                                    sampled_frames=obj['sampled_frames'],final_reason=obj['final_reason'],
                                                    static_hold=obj['static_hold'])))
    counts=dict(Counter(o['final_state'] for o in display))
    snapshot.update(objects=display,semantic_object_labels=True,fixed_track_labels=True,
                    label_mode='offline_semantic_vehicle_objects',motion_model='Pi3MOS + semantic vehicle objects',
                    aggregation_config='semantic-region-v1 + short4_z0.5-40_far20',
                    reference_image='scene-data/KITTI09_ObjectStates/reference/000399_objects.jpg',
                    position_note='Actual Gaussian nearest each predicted vehicle region median; region grouping uses original instance masks and RAFT depth. Physical vehicle identity and accuracy are not human verified.',
                    history_note='Fixed semantic-region labels aggregate unique source frames over 300-499; multiple tracks in one region/frame contribute only one model score.',
                    object_counts=counts,positioned_object_count=len(display),
                    track_counts=counts,positioned_track_counts=counts,positioned_track_count=len(display),unpositioned_track_count=0,
                    association_manifest='scene-data/KITTI09_ObjectStates/manifest.json',
                    reference_image_note='Illustrative original frame; boxes show only matched decided semantic objects, one per region.')
    save_json(asset/'snapshot.json',snapshot)
    associations=read(run/'deduplicated_observations.json')
    frame=399;im=Image.open(rgb/f'{frame:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    state_by_id={o['object_id']:o['final_state'] for o in display}
    for row in associations:
        if row['source_frame']!=frame or state_by_id[row['object_id']]=='uncertain':continue
        state=state_by_id[row['object_id']];color={'static':'#368cf5','dynamic':'#f28c28'}[state]
        b=row['bbox'];draw.rectangle(b,outline=color,width=2)
        text=f"{row['object_id']} | {state.upper()}";x,y=b[0],max(0,b[1]-18)
        w=draw.textbbox((0,0),text,font=font)[2]+6
        draw.rectangle((x,y,min(im.width,x+w),y+18),fill='#132035');draw.text((x+2,y),text,font=font,fill=color)
    im.save(asset/'reference/000399_objects.jpg',quality=93)
    for path in run.iterdir():
        if path.is_file():shutil.copy2(path,asset/'evidence'/path.name)
    for name in ['fuse_semantic_vehicle_objects.py','aggregate_vehicle_states.py','prepare_object_inputs.py']:
        shutil.copy2(Path(__file__).with_name(name),asset/'evidence'/name)
    save_json(asset/'manifest.json',dict(method='predicted_semantic_3d_regions_with_mask_depth_association',
                                       source_map_sha256=summary['source_map_sha256'],
                                       source_frames=[300,499],parameters=parameters,summary=summary,
                                       gaussian_region_ids='evidence/gaussian_region_ids.npz',
                                       label_anchor='Actual original Gaussian nearest region coordinate-wise median',
                                       original_maps_modified=False,motion_accuracy_verified=False,
                                       adoption='USER_REQUESTED_DEMO_TRIAL',backup=str(backup)))
    cfg.update(title='KITTI 09 · Original map · Object Vehicle States',
               motionSnapshot='scene-data/KITTI09_ObjectStates/snapshot.json',
               motionIndex=None,motionReferenceImageTemplate=None,
               motionLabelMode='offline_semantic_vehicle_objects',
               motionStateManifest='scene-data/KITTI09_ObjectStates/manifest.json',
               motionStateConfig='semantic-region-v1 + short4_z0.5-40_far20',hideUncertainMotion=True)
    save_json(code/'scene-config.kitti09-orb-f200.json',cfg)
    catalog=read(code/'scene-catalog.json')
    for scene in catalog['scenes']:
        if scene['id']=='kitti09_orb_f200_full':scene['label']='Seq 09 · 000300–000499 · Object States'
    save_json(code/'scene-catalog.json',catalog)
    if not all(sha(root/name)==h for name,h in protected_hashes.items()):raise ValueError('Protected predictions changed')
    receipt=dict(scene=cfg['sceneId'],candidate_regions=len(objects),state_counts=counts,
                 shown_regions=sum(o['final_state']!='uncertain' for o in display),
                 uncertain_hidden=True,one_label_per_region=True,
                 duplicate_track_pairs_resolved={'77/103':'V005','24/49':'V002','46/52':'V003','93/95':'V007'},
                 source_raw_predictions_and_new_scene_unchanged=True,original_map_files_modified=False,
                 accuracy_verified=False,backup=str(backup))
    save_json(asset/'installation_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
