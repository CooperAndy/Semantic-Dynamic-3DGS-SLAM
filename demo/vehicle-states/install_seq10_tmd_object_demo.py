"""Install object-level vehicle states only on the ORIGINAL seq10 TMD scene."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageFont
from aggregate_vehicle_states import save_json,sha


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--demo-root',required=True)
    p.add_argument('--objects-run',required=True);p.add_argument('--raw-frames',required=True);p.add_argument('--rgb-dir',required=True)
    a=p.parse_args();root,run,raw,rgb=map(Path,[a.demo_root,a.objects_run,a.raw_frames,a.rgb_dir])
    code=root/'源码';base=root/'scene-data/KITTI10_TMD_TSDD_seed2';asset=base/'ObjectStates'
    backup=root/'修改记录/20261115_seq10_tmd_object_states_v1'
    if asset.exists() or backup.exists():raise FileExistsError('This seq10 release already exists')
    cfgpath=code/'scene-config.kitti10-tmd-tsdd-s2.json';cfg=read(cfgpath)
    assert cfg['sceneId']=='kitti10_tmd_tsdd_s2' and cfg['sourceFrames']==[0,999]
    objects=read(run/'objects.json');summary=read(run/'summary.json');params=read(run/'config.json')
    source=read(base/'manifest.json')
    assert summary['source_map_sha256']==source['source_files_sha256']['params.npz']
    for path,expected in summary['source_hashes'].items():
        if sha(path)!=expected:raise ValueError(f'Calculation input changed: {path}')
    decided=[o for o in objects if o['final_state']!='uncertain']
    if not decided:raise ValueError('No decided object states; investigate inputs rather than install an empty display')
    protected=[base/'RGB.ply',base/'Semantic.ply',base/'camera_poses.json',base/'original/params.npz',base/'original/semantic_state_final.npz',code/'scene-config.kitti09-orb-f200.json',code/'scene-config.kitti09-c50-s42.json']
    protected+=list(p for p in (root/'scene-data/KITTI09_ObjectStates').rglob('*') if p.is_file())
    protected+=list(p for p in (root/'scene-data/KITTI09_C50S42/KITTI09_ObjectStates').rglob('*') if p.is_file())
    protected_hashes={str(p.relative_to(root)):sha(p) for p in protected}
    before={};backup.mkdir(parents=True,exist_ok=False)
    for name in ['源码/scene-config.kitti10-tmd-tsdd-s2.json','源码/scene-catalog.json','使用说明.md']:
        q=backup/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,q);before[name]=sha(root/name)
    save_json(backup/'backup_manifest.json',dict(before=before,protected=protected_hashes,asset_added='scene-data/KITTI10_TMD_TSDD_seed2/ObjectStates'))
    asset.mkdir();(asset/'reference').mkdir();(asset/'evidence').mkdir()
    associations=read(run/'deduplicated_observations.json');states={o['object_id']:o['final_state'] for o in objects}
    by_frame=defaultdict(list)
    for row in associations:
        if states[row['object_id']]!='uncertain':by_frame[row['source_frame']].append(row)
    frame=max(by_frame,key=lambda n:(len(by_frame[n]),sum(r['mask_area'] for r in by_frame[n]),-abs(n-499)))
    snapshot=deepcopy(read(raw/f'{frame:06d}.json'));display=[]
    for obj in objects:
        anchor=obj['world_anchor']
        display.append(dict(track_id=obj['region_index']+1,object_id=obj['object_id'],category=obj['category'],score=obj['score'],state=obj['state'],final_state=obj['final_state'],final_reason=obj['final_reason'],source_track_ids=obj['source_track_ids'],display_estimate=True,world_xyz=anchor,viewer_xyz=[-anchor[0],-anchor[1],anchor[2]],fixed_state=True,anchor_verified=False,semantic_anchor_bound=True,anchor_source_gaussian_index=obj['anchor_source_gaussian_index'],anchor_status='actual_gaussian_in_original_TMD_semantic_vehicle_region',gaussian_region_index=obj['region_index'],region_gaussian_count=obj['gaussian_count'],identity_status=obj['identity_status'],aggregate_evidence=dict(first_frame=obj['first_frame'],last_frame=obj['last_frame'],valid_count=obj['valid_observation_count'],unique_observation_frames=obj['unique_observation_frames'],sampled_frames=obj['sampled_frames'],final_reason=obj['final_reason'],static_hold=obj['static_hold'])))
    counts=dict(Counter(o['final_state'] for o in objects));prefix='scene-data/KITTI10_TMD_TSDD_seed2/ObjectStates'
    snapshot.update(objects=display,semantic_object_labels=True,fixed_track_labels=True,label_mode='offline_semantic_vehicle_objects',motion_model='Pi3MOS + semantic vehicle objects',source_frames=[0,999],window=[],aggregation_config='semantic-region-v1 + short4_z0.5-40_far20',reference_image=f'{prefix}/reference/{frame:06d}_objects.jpg',history_note='One fixed state per predicted vehicle region from full 0-999 interval; raw Pi3MOS causal8 scores, one model score per region/frame.',position_note='One actual Gaussian near each original TMD map semantic region center. TMD mapping used TSDD; these displayed motion states use Pi3MOS.',object_counts=counts,positioned_object_count=len(display),track_counts=counts,positioned_track_counts=counts,positioned_track_count=len(display),unpositioned_track_count=0,association_manifest=f'{prefix}/manifest.json',reference_image_note='Illustrative frame selected by decided region count and mask area, not ground truth or evaluation.',accuracy_status='NOT_EVALUATED_NO_OBJECT_GROUND_TRUTH')
    snapshot.pop('display_2d_only',None);save_json(asset/'snapshot.json',snapshot)
    im=Image.open(rgb/f'{frame:06d}.png').convert('RGB');draw=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for row in by_frame[frame]:
        state=states[row['object_id']];color={'static':'#368cf5','dynamic':'#f28c28'}[state];box=row['bbox']
        draw.rectangle(box,outline=color,width=2);text=f"{row['object_id']} | {state.upper()}";x,y=box[0],max(0,box[1]-18);w=draw.textbbox((0,0),text,font=font)[2]+6
        draw.rectangle((x,y,min(im.width,x+w),y+18),fill='#132035');draw.text((x+2,y),text,font=font,fill=color)
    im.save(asset/f'reference/{frame:06d}_objects.jpg',quality=93)
    for q in run.iterdir():
        if q.is_file():shutil.copy2(q,asset/'evidence'/q.name)
    for q in [Path(__file__),Path(__file__).parent/'prepare_seq10_tmd_motion.py',Path(__file__).parent/'reuse/run_vehicle_objects.py']:
        shutil.copy2(q,asset/'evidence'/q.name)
    save_json(asset/'manifest.json',dict(method='predicted_semantic_3d_regions_with_mask_depth_association',source_map_sha256=summary['source_map_sha256'],source_frames=[0,999],parameters=params,summary=summary,gaussian_count=cfg['gaussianCount'],gaussian_region_ids='evidence/gaussian_region_ids.npz',semantic_source='Original TMD accumulated Cityscapes-19 votes, 950 integrated mapper frames; retained map covers 0-999',motion_source='Existing full1000 Pi3MOS causal8 float32 instance medians; original TSDD scores preserved but not reused as probabilities',depth_source='Existing full raw metric RAFT stereo depth from same RGB frames; confidence-hard mapping depth diagnosis saved separately',birth_frame_filter=False,original_maps_modified=False,motion_accuracy_verified=False,reference_frame_selection='After predictions: prefer decided region count, then mask area, then proximity to frame499; illustrative only',backup=str(backup)))
    cfg.update(title='KITTI 10 · Original TMD TSDD S2 · Object States',motionIndex=None,motionSnapshot=f'{prefix}/snapshot.json',motionReferenceImageTemplate=None,motionLabelMode='offline_semantic_vehicle_objects',motionStateManifest=f'{prefix}/manifest.json',motionStateConfig='semantic-region-v1 + short4_z0.5-40_far20',hideUncertainMotion=True,dynamicVehicleRemoval=None)
    save_json(cfgpath,cfg)
    catalog=read(code/'scene-catalog.json')
    for scene in catalog['scenes']:
        if scene['id']==cfg['sceneId']:scene['label']='Seq 10 · 000000–000999 · TMD S2 · Object States'
    save_json(code/'scene-catalog.json',catalog)
    if not all(sha(root/name)==h for name,h in protected_hashes.items()):raise ValueError('Protected maps or successful seq09 scenes changed')
    receipt=dict(scene=cfg['sceneId'],source_frames=[0,999],map_gaussians=cfg['gaussianCount'],candidate_regions=len(objects),state_counts=counts,shown_regions=len(decided),reference_frame=frame,uncertain_hidden=True,original_map_unchanged=True,seq09_success_scenes_unchanged=True,new_seq10_scene_modified=False,backup=str(backup))
    save_json(asset/'installation_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
