"""Install the successful semantic-object state method on the C50S42 map.

Target-map regions/indices/anchors must have been recomputed before installation.
The visually accepted old-map scene and frozen release are protected by hashes.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageFont
from aggregate_vehicle_states import save_json,sha
import json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--demo-root',required=True)
    p.add_argument('--objects-run',required=True);p.add_argument('--rgb-dir',required=True)
    a=p.parse_args();root,run,rgb=map(Path,[a.demo_root,a.objects_run,a.rgb_dir])
    code=root/'源码';base=root/'scene-data/KITTI09_C50S42'
    target=base/'KITTI09_ObjectStates';backup=root/'修改记录/20261115_seq09_c50s42_object_states_v1'
    if target.exists() or backup.exists():raise FileExistsError('Target release or backup already exists')
    cfg_path=code/'scene-config.kitti09-c50-s42.json';cfg=read(cfg_path)
    if cfg['sceneId']!='kitti09_conf050_seed42':raise ValueError('Wrong target scene')
    summary=read(run/'summary.json');objects=read(run/'objects.json');parameters=read(run/'config.json')
    sm=read(base/'KITTI09_F1_Semantic/semantic_manifest.json')
    if summary['source_map_sha256']!=sm['source_params_sha256']:raise ValueError('Output was computed for another Gaussian map')
    for path,h in summary['source_hashes'].items():
        if sha(path)!=h:raise ValueError(f'Calculation source changed: {path}')
    protected=[code/'scene-config.kitti09-orb-f200.json',code/'scene-catalog.json',code/'interactive_viewer.html',code/'t07.js',
               base/'KITTI09_MotionDemo/index.json',base/'KITTI09_MotionDemo/snapshot.json',
               base/'KITTI09_F1/KITTI09_F1_RGB.ply',base/'KITTI09_F200/KITTI09_F200_RGB.ply',
               base/'KITTI09_F1_Semantic/KITTI09_F1_Semantic.ply',base/'KITTI09_F1_Semantic/semantic_gaussian_labels.npz']
    protected+=list((base/'KITTI09_MotionDemo/frames').glob('*.json'))
    protected+=list(p for p in (root/'scene-data/KITTI09_ObjectStates').rglob('*') if p.is_file())
    protected_hashes={str(p.relative_to(root)):sha(p) for p in protected}
    backup.mkdir(parents=True,exist_ok=False)
    before={}
    for name in ['源码/scene-config.kitti09-c50-s42.json','使用说明.md']:
        dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest);before[name]=sha(root/name)
    save_json(backup/'backup_manifest.json',dict(before=before,protected=protected_hashes,asset_added='scene-data/KITTI09_C50S42/KITTI09_ObjectStates'))
    target.mkdir();(target/'reference').mkdir();(target/'evidence').mkdir()
    frame=int(cfg['defaultCameraFrame'])
    snapshot=deepcopy(read(base/f'KITTI09_MotionDemo/frames/{frame:06d}.json'))
    display=[]
    for obj in objects:
        anchor=obj['world_anchor']
        display.append(dict(track_id=obj['region_index']+1,object_id=obj['object_id'],category=obj['category'],
                            score=obj['score'],state=obj['state'],final_state=obj['final_state'],final_reason=obj['final_reason'],
                            source_track_ids=obj['source_track_ids'],display_estimate=True,
                            world_xyz=anchor,viewer_xyz=[-anchor[0],-anchor[1],anchor[2]],fixed_state=True,
                            anchor_verified=False,semantic_anchor_bound=True,
                            anchor_source_gaussian_index=obj['anchor_source_gaussian_index'],
                            anchor_status='actual_gaussian_in_target_C50S42_semantic_vehicle_region',
                            gaussian_region_index=obj['region_index'],region_gaussian_count=obj['gaussian_count'],
                            identity_status=obj['identity_status'],
                            aggregate_evidence=dict(first_frame=obj['first_frame'],last_frame=obj['last_frame'],
                              valid_count=obj['valid_observation_count'],unique_observation_frames=obj['unique_observation_frames'],
                              sampled_frames=obj['sampled_frames'],final_reason=obj['final_reason'],static_hold=obj['static_hold'])))
    counts=dict(Counter(o['final_state'] for o in display))
    prefix='scene-data/KITTI09_C50S42/KITTI09_ObjectStates'
    snapshot.update(objects=display,semantic_object_labels=True,fixed_track_labels=True,
                    label_mode='offline_semantic_vehicle_objects',motion_model='Pi3MOS + semantic vehicle objects',
                    source_frames=[300,499],source_map_sha256=sm['source_params_sha256'],window=[],
                    aggregation_config='semantic-region-v1 + short4_z0.5-40_far20',
                    reference_image=f'{prefix}/reference/{frame:06d}_objects.jpg',
                    position_note='Each anchor is an actual Gaussian in this C50S42 map, near the corresponding predicted vehicle-region center. No old-map indices or positions copied.',
                    history_note='Fixed semantic-region labels aggregate unique source frames over 300-499; one model score per region/frame. Offline output may use future observations relative to early frames.',
                    object_counts=counts,positioned_object_count=len(display),track_counts=counts,
                    positioned_track_counts=counts,positioned_track_count=len(display),unpositioned_track_count=0,
                    association_manifest=f'{prefix}/manifest.json',accuracy_status='NOT_EVALUATED_NO_OBJECT_GROUND_TRUTH',
                    reference_image_note='Illustrative original frame with one decided label per matched semantic region.')
    save_json(target/'snapshot.json',snapshot)
    im=Image.open(rgb/f'{frame:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    states={o['object_id']:o['final_state'] for o in display}
    for row in read(run/'deduplicated_observations.json'):
        if row['source_frame']!=frame or states[row['object_id']]=='uncertain':continue
        state=states[row['object_id']];color={'static':'#368cf5','dynamic':'#f28c28'}[state]
        box=row['bbox'];draw.rectangle(box,outline=color,width=2)
        text=f"{row['object_id']} | {state.upper()}";x,y=box[0],max(0,box[1]-18);w=draw.textbbox((0,0),text,font=font)[2]+6
        draw.rectangle((x,y,min(im.width,x+w),y+18),fill='#132035');draw.text((x+2,y),text,font=font,fill=color)
    im.save(target/f'reference/{frame:06d}_objects.jpg',quality=93)
    for source in run.iterdir():
        if source.is_file():shutil.copy2(source,target/'evidence'/source.name)
    profile=read(run/'resolved_profile.json')
    paths=profile['resolved_paths'];input_manifest=Path(paths['instance_inputs'])/'manifest.json'
    shutil.copy2(input_manifest,target/'evidence/instance_input_binding.json')
    for source in [Path(__file__),Path(__file__).parent/'reuse/run_vehicle_objects.py',Path(__file__).parent/'reuse/core/fuse_reference_seq09.py',Path(__file__).parent/'reuse/core/aggregate_vehicle_states.py']:
        shutil.copy2(source,target/'evidence'/source.name)
    save_json(target/'manifest.json',dict(method='predicted_semantic_3d_regions_with_mask_depth_association',
      source_map_sha256=sm['source_params_sha256'],source_frames=[300,499],parameters=parameters,summary=summary,
      gaussian_region_ids='evidence/gaussian_region_ids.npz',gaussian_count=cfg['gaussianCount'],
      label_anchor='Actual C50S42 Gaussian nearest each target-region median',
      transferred_method_from='kitti09_orb_f200_full',copied_old_map_indices=False,
      original_maps_modified=False,motion_accuracy_verified=False,
      legacy_removal=dict(enabled=False,reason='Original cleanup mask uses raw per-frame states, not these fixed object states.',original_configuration=cfg.get('dynamicVehicleRemoval')),
      adoption='USER_REQUESTED_MAIN_PRESENTATION_SCENE',backup=str(backup)))
    cfg.update(motionIndex=None,motionSnapshot=f'{prefix}/snapshot.json',motionReferenceImageTemplate=None,
      motionLabelMode='offline_semantic_vehicle_objects',motionStateManifest=f'{prefix}/manifest.json',
      motionStateConfig='semantic-region-v1 + short4_z0.5-40_far20',hideUncertainMotion=True,dynamicVehicleRemoval=None)
    save_json(cfg_path,cfg)
    if not all(sha(root/name)==h for name,h in protected_hashes.items()):raise ValueError('Protected old scene, source predictions or map files changed')
    receipt=dict(scene=cfg['sceneId'],state_counts=counts,shown_regions=sum(o['final_state']!='uncertain' for o in display),
      target_regions_recomputed=True,old_successful_scene_unchanged=True,map_assets_unchanged=True,
      raw_predictions_unchanged=True,legacy_per_frame_removal_disabled=True,uncertain_hidden=True,backup=str(backup))
    save_json(target/'installation_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
