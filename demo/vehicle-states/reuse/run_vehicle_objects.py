"""Configurable cross-sequence calculation using the frozen successful seq09 core.

Only writes a new result directory. It never installs or changes a Demo scene.
Currently requires the same metric-Y-vertical, Cityscapes-19 schema as seq09.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

CORE=Path(__file__).resolve().parent/'core'
sys.path.insert(0,str(CORE))
from aggregate_vehicle_states import Config,decide,rejects,save_csv,save_json,sha
from fuse_reference_seq09 import (FusionConfig,cluster_vehicle_points,compound_conflicts,
    deduplicate_frame_observations,load_ply,match_region,project,read)


def resolve_profile(path, root_override=None):
    path=Path(path).resolve();profile=read(path)
    root=Path(root_override).resolve() if root_override else (path.parent/profile['workspace_root']).resolve()
    paths={}
    for key in ['semantic_dir','raw_frames','camera_poses','instance_inputs','observation_run']:
        value=profile['paths'].get(key)
        if not value:raise ValueError(f'Profile is not ready: fill paths.{key}')
        paths[key]=(root/value).resolve()
        if not paths[key].exists():raise FileNotFoundError(f'{key}: {paths[key]}')
    if profile.get('world_vertical_axis')!='y' or profile.get('viewer_signs')!=[-1,-1,1]:
        raise ValueError('This version requires metric Y-vertical map coordinates and Viewer [-x,-y,z]. Adapt coordinates explicitly before reuse.')
    if profile.get('semantic_classes')!={'car':13,'truck':14,'bus':15}:
        raise ValueError('Frozen core currently requires Cityscapes-19 vehicle IDs 13/14/15.')
    cfg=FusionConfig(**profile['parameters']['region']);motion=Config(**profile['parameters']['motion'])
    if cfg.voxel_m<=0 or cfg.connection_m<=0 or not .5<=motion.depth_min<=motion.depth_max:
        raise ValueError('Invalid voxel, connection or depth bounds')
    if motion.sample_gap<1 or motion.min_samples<1 or not 0<=motion.static_threshold<motion.dynamic_threshold<=1:
        raise ValueError('Invalid evidence sampling/score thresholds')
    return profile,paths,cfg,motion


def run(args):
    profile,paths,cfg,motion=resolve_profile(args.profile,args.workspace_root)
    semantic,inputs,obsrun=paths['semantic_dir'],paths['instance_inputs'],paths['observation_run']
    sm=read(semantic/'semantic_manifest.json');im=read(inputs/'manifest.json')
    expected=list(range(profile['frame_range'][0],profile['frame_range'][1]+1))
    if [e['source_frame'] for e in im['frames']]!=expected:raise ValueError('Profile frame range and exported instance inputs disagree')
    if sm['source_params_sha256']!=im['source_map_sha256']:raise ValueError('Semantic map and instance/depth input map hashes disagree')
    depth_range=im.get('depth_range_m',[.5,40.])
    if motion.depth_min<depth_range[0] or motion.depth_max>depth_range[1]:raise ValueError('Motion depth gate exceeds prepared depth range; prepare matching inputs first')
    ply_path=semantic/profile['paths']['semantic_ply']
    if sha(ply_path)!=sm['semantic_ply_sha256']:raise ValueError('Semantic PLY hash mismatch')
    source_manifest=read(obsrun/'input_manifest.json')
    if source_manifest['source_map_sha256']!=sm['source_params_sha256']:raise ValueError('Observation run belongs to another map')
    if [int(Path(e['name']).stem) for e in source_manifest['frames']]!=expected:raise ValueError('Observation frame range mismatch')
    for e in source_manifest['frames']:
        if sha(paths['raw_frames']/e['name'])!=e['sha256']:raise ValueError('Raw motion snapshot changed after observation extraction')
    rows=read(obsrun/'observations.json');by_detection={(r['source_frame'],r['track_id']):r for r in rows}
    if len(by_detection)!=len(rows):raise ValueError('Duplicate source frame/track rows')
    poses={p['source_frame']:p for p in read(paths['camera_poses'])}
    if any(n not in poses for n in expected):raise ValueError('Camera poses do not cover the source-frame range')
    intrinsics=np.asarray(im['intrinsics'],dtype=float)
    if intrinsics.shape!=(3,3) or not np.isfinite(intrinsics).all():raise ValueError('Expected finite 3x3 intrinsics')
    ply=load_ply(ply_path)
    with np.load(semantic/'semantic_gaussian_labels.npz',allow_pickle=False) as z:
        if len(z['label'])!=len(ply) or not np.array_equal(z['source_gaussian_index'],np.arange(len(ply))):raise ValueError('Point indices/geometry mismatch')
        names=z['class_names'].tolist()
        if any(names[i]!=name for name,i in profile['semantic_classes'].items()):raise ValueError('Semantic class names/IDs disagree')
        labels,confidence,observations=z['label'],z['confidence'],z['observation_count']
        # The reference core assumes a nonempty eligible point set.
        opacity=1/(1+np.exp(-np.clip(ply['opacity'],-30,30)))
        scales=np.exp(np.maximum.reduce([ply[k] for k in ('scale_0','scale_1','scale_2')]))
        eligible=(np.isin(labels,[13,14,15])&(confidence>=cfg.semantic_confidence)&(observations>=cfg.min_semantic_observations)&(opacity>=cfg.min_opacity)&(scales<=cfg.max_scale_m))
        if eligible.any():
            regions,discarded,points,region_ids,gaussian_region,selected=cluster_vehicle_points(ply,labels,confidence,observations,cfg)
        else:
            regions,discarded=[],[];points=np.empty((0,3));region_ids=np.empty(0,dtype=int);gaussian_region=np.full(len(ply),-1,dtype='int16');selected=np.empty(0,dtype=int)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'gaussian_region_ids.npz',region_id=gaussian_region)
    save_json(out/'regions.json',regions);save_json(out/'discarded_components.json',discarded)
    associated=defaultdict(list);audit=[];alignment=[];seen=0
    for entry in im['frames']:
        n=entry['source_frame'];mp=inputs/f'instances/{n:06d}_masks.npz';jp=inputs/f'instances/{n:06d}.json';dp=inputs/f'masked_depth/{n:06d}.npz'
        if sha(mp)!=entry['mask_sha256'] or sha(jp)!=entry['instance_sha256'] or sha(dp)!=entry['masked_depth_sha256']:raise ValueError('Prepared instance/depth file hash mismatch')
        with np.load(mp,allow_pickle=False) as z:masks=z['masks']
        with np.load(dp,allow_pickle=False) as z:depth=z['depth']
        metadata=read(jp);meta_by_track={o['track_id']:o for o in metadata['instances']}
        raw=read(paths['raw_frames']/f'{n:06d}.json')
        if raw.get('sequence')!=profile['sequence'] or raw.get('scene_id')!=profile['scene_id']:raise ValueError(f'Wrong sequence/scene at frame {n}')
        height,width=int(raw['image_height']),int(raw['image_width'])
        if masks.shape[1:]!=(height,width) or depth.shape!=(height,width):raise ValueError('Image/mask/depth dimensions disagree')
        u,v,z=project(points,poses[n],intrinsics)
        visible=(z>=motion.depth_min)&(z<=motion.depth_max)&(u>=0)&(u<width)&(v>=0)&(v<height)
        q,ids,u,v,z=points[visible],region_ids[visible],u[visible],v[visible],z[visible]
        for obj in raw['objects']:
            seen+=1;row=dict(by_detection[n,obj['track_id']]);meta=meta_by_track[obj['track_id']]
            if row['raw_score']!=obj['score'] or row['raw_state']!=obj['raw_state']:raise ValueError('Observation run and raw model scores disagree')
            if meta['mask_index']!=obj['mask_index'] or meta['mask_area']!=obj['mask_area']:raise ValueError('Mask index/area disagreement')
            mask=masks[obj['mask_index']]
            if int(mask.sum())!=obj['mask_area']:raise ValueError('Mask area mismatch')
            if obj.get('display_estimate'):
                au,av,_=project(np.asarray([obj['world_xyz']]),poses[n],intrinsics)
                ok=0<=au[0]<width and 0<=av[0]<height and bool(mask[av[0],au[0]])
                alignment.append(ok)
                if not ok:raise ValueError(f'Coordinate mismatch: original anchor outside mask at frame {n}, track {obj["track_id"]}')
            dc=float(meta['confidence']);match,candidates,reason=match_region(q,ids,u,v,z,mask,depth,cfg)
            if dc<cfg.min_detection_confidence:match=None;reason='low_detection_confidence'
            audit.append(dict(source_frame=n,track_id=obj['track_id'],raw_score=obj['score'],raw_state=obj['raw_state'],detection_confidence=dc,association_reason=reason,object_id=regions[match['region_index']]['object_id'] if match else None,candidates=candidates))
            if match:
                row.update(match,detection_confidence=dc);associated[match['region_index']].append(row)
    if seen!=len(rows):raise ValueError('Observation extraction has extra/missing records')
    objects=[];dedup_records=[]
    for region in regions:
        matches=associated[region['region_index']];valid=[r for r in matches if not rejects(r,motion)]
        dedup=deduplicate_frame_observations(valid);result=decide(dedup,motion,profile.get('enable_static_hold',True))
        conflicts=compound_conflicts(valid)
        if conflicts:result.update(final_state='uncertain',final_reason='possible_compound_vehicle_region')
        for r in dedup:r['object_id']=region['object_id'];dedup_records.append(r)
        objects.append(dict(region,final_state=result['final_state'],state={'static':'静态','dynamic':'动态','uncertain':'不确定'}[result['final_state']],score=result['score_median'],final_reason=result['final_reason'],static_hold=result['static_hold'],source_track_ids=sorted({r['track_id'] for r in matches}),association_count=len(matches),valid_observation_count=len(valid),unique_observation_frames=len(dedup),sampled_frames=result['sampled_frames'],same_frame_duplicate_count=len(valid)-len(dedup),compound_conflicts=conflicts,first_frame=min((r['source_frame'] for r in matches),default=None),last_frame=max((r['source_frame'] for r in matches),default=None),identity_status='predicted_semantic_region_mask_and_depth_supported_not_human_verified'))
    save_json(out/'objects.json',objects);save_json(out/'associations.json',audit);save_csv(out/'associations.csv',audit)
    save_json(out/'deduplicated_observations.json',dedup_records)
    save_csv(out/'object_summary.csv',[{k:v for k,v in o.items() if k not in ('world_anchor','world_center','bbox_world_min','bbox_world_max')} for o in objects])
    save_json(out/'config.json',dict(region=asdict(cfg),motion=asdict(motion)))
    save_json(out/'resolved_profile.json',dict(profile=profile,resolved_paths={k:str(v) for k,v in paths.items()}))
    summary=dict(profile=profile['name'],sequence=profile['sequence'],frame_range=profile['frame_range'],source_map_sha256=sm['source_params_sha256'],candidate_regions=len(regions),quality_vehicle_gaussians=len(selected),raw_observations=len(rows),accepted_associations=sum(len(v) for v in associated.values()),association_reasons=dict(Counter(r['association_reason'] for r in audit)),object_state_counts=dict(Counter(o['final_state'] for o in objects)),original_anchor_reprojection_checks=len(alignment),original_anchor_reprojection_all_pass=all(alignment),static_hold_enabled=profile.get('enable_static_hold',True),source_hashes={str(p):sha(p) for p in [ply_path,semantic/'semantic_gaussian_labels.npz',inputs/'manifest.json',obsrun/'observations.json',Path(args.profile),CORE/'fuse_reference_seq09.py',CORE/'aggregate_vehicle_states.py',Path(__file__)]},accuracy_status='NOT_EVALUATED_NO_OBJECT_GROUND_TRUTH',demo_installed=False)
    save_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True)
    p.add_argument('--workspace-root');p.add_argument('--output',required=True)
    run(p.parse_args())
