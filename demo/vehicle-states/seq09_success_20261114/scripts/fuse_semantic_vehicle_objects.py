"""Bind raw per-frame motion evidence to predicted 3D vehicle semantic regions.

This is an auditable offline prototype, not verified physical-object ground truth.
No Gaussian/map assets, model weights or input predictions are changed.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from aggregate_vehicle_states import Config, decide, rejects, save_csv, save_json, sha


@dataclass(frozen=True)
class FusionConfig:
    voxel_m: float = .15
    connection_m: float = .45
    semantic_confidence: float = .55
    min_semantic_observations: int = 2
    min_opacity: float = .1
    max_scale_m: float = .25
    min_voxels: int = 100
    min_gaussians: int = 500
    min_height_m: float = .45
    max_height_m: float = 4.5
    min_long_m: float = 1.4
    min_short_m: float = .6
    max_long_m: float = 15.
    max_short_m: float = 6.5
    min_support_voxels: int = 8
    min_support_pixels: int = 6
    min_projected_coverage: float = .2
    min_depth_consistency: float = .5
    min_winner_share: float = .7
    min_detection_confidence: float = .5
    depth_abs_tolerance_m: float = .35
    depth_relative_tolerance: float = .08


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_ply(path):
    types={'float':'<f4','float32':'<f4','uchar':'u1','uint8':'u1','int':'<i4'}
    with Path(path).open('rb') as f:
        header=[]
        while True:
            b=f.readline()
            if not b:raise ValueError('Missing PLY end_header')
            line=b.decode('ascii').strip();header.append(line)
            if line=='end_header':break
        offset=f.tell()
    if header[1]!='format binary_little_endian 1.0':raise ValueError('Expected binary little-endian PLY')
    n=int(next(l for l in header if l.startswith('element vertex ')).split()[-1])
    fields=[]
    for l in header:
        if l.startswith('property '):
            _,kind,name=l.split();fields.append((name,types[kind]))
    return np.memmap(path,dtype=fields,mode='r',offset=offset,shape=(n,))


def component_reasons(points, nvox, cfg):
    lo,hi=np.quantile(points,[.02,.98],axis=0)
    ext=hi-lo;horizontal=sorted([float(ext[0]),float(ext[2])]); reasons=[]
    if nvox<cfg.min_voxels:reasons.append('too_few_voxels')
    if len(points)<cfg.min_gaussians:reasons.append('too_few_gaussians')
    if not cfg.min_height_m<=ext[1]<=cfg.max_height_m:reasons.append('height_outside_candidate_range')
    if not cfg.min_long_m<=horizontal[1]<=cfg.max_long_m:reasons.append('length_outside_candidate_range')
    if not cfg.min_short_m<=horizontal[0]<=cfg.max_short_m:reasons.append('width_outside_candidate_range')
    return reasons,lo,hi


def cluster_vehicle_points(ply, labels, confidence, observations, cfg):
    xyz=np.stack([ply[k] for k in ('x','y','z')],axis=1)
    opacity=1/(1+np.exp(-np.clip(ply['opacity'],-30,30)))
    scales=np.exp(np.maximum.reduce([ply[k] for k in ('scale_0','scale_1','scale_2')]))
    selected=np.flatnonzero(np.isin(labels,[13,14,15]) & (confidence>=cfg.semantic_confidence) &
                            (observations>=cfg.min_semantic_observations) & (opacity>=cfg.min_opacity) &
                            (scales<=cfg.max_scale_m) & np.isfinite(xyz).all(axis=1))
    points=xyz[selected].astype(np.float64)
    cells,inverse=np.unique(np.floor(points/cfg.voxel_m).astype(np.int32),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    centers=np.stack([np.bincount(inverse,weights=points[:,i])/counts for i in range(3)],axis=1)
    pairs=cKDTree(centers).query_pairs(cfg.connection_m,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs),dtype='int8'),(pairs[:,0],pairs[:,1])),shape=(len(cells),len(cells)))
    _,components=connected_components(graph,directed=False)
    component_points=components[inverse];voxel_counts=np.bincount(components)
    regions,discarded=[],[]
    for component,nvox in enumerate(voxel_counts):
        loc=np.flatnonzero(component_points==component)
        q=points[loc]
        reasons,lo,hi=component_reasons(q,int(nvox),cfg)
        if reasons:
            discarded.append(dict(component=int(component),points=len(q),voxels=int(nvox),reason=reasons));continue
        center=np.median(q,axis=0)
        anchor_index=int(selected[loc[np.argmin(np.sum((q-center)**2,axis=1))]])
        dominant=int(Counter(labels[selected[loc]].tolist()).most_common(1)[0][0])
        regions.append(dict(component=int(component),voxel_count=int(nvox),gaussian_count=len(q),
                            world_center=center.tolist(),world_anchor=xyz[anchor_index].astype(float).tolist(),
                            anchor_source_gaussian_index=anchor_index,
                            bbox_world_min=lo.tolist(),bbox_world_max=hi.tolist(),
                            category={13:'car',14:'truck',15:'bus'}[dominant],dominant_semantic_label=dominant))
    regions.sort(key=lambda r:(r['world_center'][2],r['world_center'][0]))
    voxel_region=np.full(len(centers),-1,dtype=np.int32)
    gaussian_region=np.full(len(ply),-1,dtype=np.int16)
    for i,region in enumerate(regions):
        region.update(region_index=i,object_id=f'V{i+1:03d}')
        voxel_region[components==region['component']]=i
        gaussian_region[selected[component_points==region['component']]]=i
    active=voxel_region>=0
    return regions,discarded,centers[active],voxel_region[active],gaussian_region,selected


def project(points, pose, intrinsics):
    cam=(points-np.asarray(pose['position']))@np.asarray(pose['rotation'])
    z=cam[:,2]
    safe_z=np.where(z>=.5,z,1.)
    u=intrinsics[0,0]*cam[:,0]/safe_z+intrinsics[0,2]
    v=intrinsics[1,1]*cam[:,1]/safe_z+intrinsics[1,2]
    u=np.where((z>=.5)&np.isfinite(u)&(np.abs(u)<2**30),np.rint(u),-1).astype(np.int64)
    v=np.where((z>=.5)&np.isfinite(v)&(np.abs(v)<2**30),np.rint(v),-1).astype(np.int64)
    return u,v,z


def match_region(points, region_ids, u, v, z, mask, depth, cfg):
    candidates=[]
    inside=mask[v,u]
    for region_id in np.unique(region_ids[inside]):
        selected=(region_ids==region_id)&inside
        measured=depth[v[selected],u[selected]]
        predicted=z[selected]
        good=(measured>0)&(np.abs(measured-predicted)<=cfg.depth_abs_tolerance_m+cfg.depth_relative_tolerance*predicted)
        support=int(good.sum()); in_mask=int(selected.sum());visible=int((region_ids==region_id).sum())
        pix=int(len(np.unique(v[selected][good]*mask.shape[1]+u[selected][good])))
        ratio=support/max(1,in_mask);coverage=support/max(1,visible)
        if support<cfg.min_support_voxels or pix<cfg.min_support_pixels or ratio<cfg.min_depth_consistency or coverage<cfg.min_projected_coverage:continue
        score=pix*ratio*math.sqrt(coverage)
        center=np.mean(points[selected][good],axis=0).tolist()
        candidates.append(dict(region_index=int(region_id),support_voxels=support,support_pixels=pix,
                               depth_consistency=ratio,projected_coverage=coverage,association_score=score,
                               support_world_center=center))
    candidates.sort(key=lambda r:(-r['association_score'],r['region_index']))
    if not candidates:return None,[],'no_depth_consistent_region_support'
    share=candidates[0]['association_score']/sum(r['association_score'] for r in candidates[:2])
    if share<cfg.min_winner_share:return None,candidates,'ambiguous_regions'
    best=dict(candidates[0],winner_share=share)
    return best,candidates,'matched'


def deduplicate_frame_observations(rows):
    """One raw-model score per object/frame, even if several track IDs overlap."""
    groups=defaultdict(list)
    for row in rows:groups[row['source_frame']].append(row)
    chosen=[]
    for _,group in sorted(groups.items()):
        winner=max(group,key=lambda r:(r['association_score']*r['detection_confidence'],r['mask_area'],-r['track_id']))
        winner=dict(winner,merged_same_frame_tracks=sorted(r['track_id'] for r in group))
        chosen.append(winner)
    return chosen


def bbox_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    return intersection/max(1,area(a)+area(b)-intersection)


def compound_conflicts(rows):
    frames=defaultdict(list);conflicts=[]
    for row in rows:frames[row['source_frame']].append(row)
    for n,group in frames.items():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                separation=float(np.linalg.norm(np.asarray(a['support_world_center'])-b['support_world_center']))
                if bbox_iou(a['bbox'],b['bbox'])<.02 and separation>1.5:
                    conflicts.append(dict(frame=n,tracks=[a['track_id'],b['track_id']],support_separation_m=separation))
    return conflicts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo-root',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--observation-run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();demo,inputs,run,out=map(Path,[args.demo_root,args.inputs,args.observation_run,args.output])
    out.mkdir(parents=True,exist_ok=False)
    cfg=FusionConfig(); motion_cfg=replace(Config(),sample_gap=4,min_samples=2)
    semantic=demo/'scene-data/KITTI09_F1_Semantic'
    semantic_manifest=read(semantic/'semantic_manifest.json'); input_manifest=read(inputs/'manifest.json')
    if semantic_manifest['source_params_sha256']!=input_manifest['source_map_sha256']:raise ValueError('Map/instance source mismatch')
    ply_path=semantic/'KITTI09_F1_Semantic.ply'
    if sha(ply_path)!=semantic_manifest['semantic_ply_sha256']:raise ValueError('Semantic PLY hash mismatch')
    ply=load_ply(ply_path)
    with np.load(semantic/'semantic_gaussian_labels.npz',allow_pickle=False) as z:
        labels=z['label'];confidence=z['confidence'];observations=z['observation_count']
        if not np.array_equal(z['source_gaussian_index'],np.arange(len(ply))):raise ValueError('Point index order mismatch')
        regions,discarded,points,region_ids,gaussian_region,selected=cluster_vehicle_points(ply,labels,confidence,observations,cfg)
    np.savez_compressed(out/'gaussian_region_ids.npz',region_id=gaussian_region)
    save_json(out/'regions.json',regions);save_json(out/'discarded_components.json',discarded)
    rows=read(run/'observations.json'); by_detection={(r['source_frame'],r['track_id']):r for r in rows}
    poses={p['source_frame']:p for p in read(demo/'scene-data/KITTI09_F200/KITTI09_F200_camera_poses.json')}
    intrinsics=np.asarray(input_manifest['intrinsics'])
    associated=defaultdict(list);audit=[];alignment=[]
    for entry in input_manifest['frames']:
        n=entry['source_frame']
        mask_path=inputs/f'instances/{n:06d}_masks.npz';meta_path=inputs/f'instances/{n:06d}.json';depth_path=inputs/f'masked_depth/{n:06d}.npz'
        if sha(mask_path)!=entry['mask_sha256'] or sha(meta_path)!=entry['instance_sha256'] or sha(depth_path)!=entry['masked_depth_sha256']:raise ValueError('Instance input hash mismatch')
        with np.load(mask_path,allow_pickle=False) as z:masks=z['masks']
        with np.load(depth_path,allow_pickle=False) as z:depth=z['depth']
        metadata=read(meta_path);meta_by_track={o['track_id']:o for o in metadata['instances']}
        raw=read(demo/f'scene-data/KITTI09_MotionDemo/frames/{n:06d}.json')
        u,v,z=project(points,poses[n],intrinsics)
        in_view=(z>=.5)&(z<=40)&(u>=0)&(u<1226)&(v>=0)&(v<370)
        q,ids,u,v,z=points[in_view],region_ids[in_view],u[in_view],v[in_view],z[in_view]
        for obj in raw['objects']:
            row=dict(by_detection[n,obj['track_id']]);meta=meta_by_track[obj['track_id']]
            if meta['mask_index']!=obj['mask_index'] or meta['mask_area']!=obj['mask_area']:raise ValueError('Mask index does not match raw motion object')
            mask=masks[obj['mask_index']]
            if int(mask.sum())!=obj['mask_area']:raise ValueError('Mask area mismatch')
            # Audit original anchors through the same K/pose/mask coordinates.
            if obj.get('display_estimate'):
                au,av,az=project(np.asarray([obj['world_xyz']]),poses[n],intrinsics)
                inside=0<=au[0]<1226 and 0<=av[0]<370 and bool(mask[av[0],au[0]])
                alignment.append(inside)
                if not inside:raise ValueError(f'Original anchor no longer projects inside its mask: {n}/{obj["track_id"]}')
            detection_confidence=float(meta['confidence'])
            match,candidates,reason=match_region(q,ids,u,v,z,mask,depth,cfg)
            if detection_confidence<cfg.min_detection_confidence:match=None;reason='low_detection_confidence'
            audit.append(dict(source_frame=n,track_id=obj['track_id'],raw_score=obj['score'],raw_state=obj['raw_state'],
                              detection_confidence=detection_confidence,association_reason=reason,
                              object_id=regions[match['region_index']]['object_id'] if match else None,
                              candidates=candidates))
            if match:
                row.update(match,detection_confidence=detection_confidence)
                associated[match['region_index']].append(row)
    objects=[];dedup_records=[]
    for region in regions:
        rid=region['region_index'];observations=associated[rid]
        eligible=[r for r in observations if not rejects(r,motion_cfg)]
        dedup=deduplicate_frame_observations(eligible)
        result=decide(dedup,motion_cfg,True)
        conflicts=compound_conflicts(eligible)
        # A fixed region cannot safely stand for multiple spatially distinct instances.
        if conflicts:result.update(final_state='uncertain',final_reason='possible_compound_vehicle_region')
        for r in dedup:
            r['object_id']=region['object_id'];dedup_records.append(r)
        obj=dict(region,final_state=result['final_state'],state={'static':'静态','dynamic':'动态','uncertain':'不确定'}[result['final_state']],
                 score=result['score_median'],final_reason=result['final_reason'],static_hold=result['static_hold'],
                 source_track_ids=sorted({r['track_id'] for r in observations}),
                 association_count=len(observations),valid_observation_count=len(eligible),
                 unique_observation_frames=len(dedup),sampled_frames=result['sampled_frames'],
                 same_frame_duplicate_count=len(eligible)-len(dedup),compound_conflicts=conflicts,
                 first_frame=min((r['source_frame'] for r in observations),default=None),
                 last_frame=max((r['source_frame'] for r in observations),default=None),
                 identity_status='predicted_semantic_region_mask_and_depth_supported_not_human_verified')
        objects.append(obj)
    save_json(out/'objects.json',objects);save_json(out/'associations.json',audit)
    save_csv(out/'associations.csv',audit);save_json(out/'deduplicated_observations.json',dedup_records)
    save_csv(out/'object_summary.csv',[{k:v for k,v in o.items() if k not in ('world_anchor','world_center','bbox_world_min','bbox_world_max')} for o in objects])
    save_json(out/'config.json',dict(region=asdict(cfg),motion=asdict(motion_cfg)))
    source_hashes={str(p):sha(p) for p in [ply_path,semantic/'semantic_gaussian_labels.npz',inputs/'manifest.json',run/'observations.json',Path(__file__)]}
    summary=dict(source_map_sha256=semantic_manifest['source_params_sha256'],quality_vehicle_gaussians=len(selected),
                 candidate_regions=len(regions),discarded_components=len(discarded),raw_observations=len(rows),
                 accepted_associations=sum(len(v) for v in associated.values()),
                 association_reasons=dict(Counter(a['association_reason'] for a in audit)),
                 object_state_counts=dict(Counter(o['final_state'] for o in objects)),
                 decided_regions=[o['object_id'] for o in objects if o['final_state']!='uncertain'],
                 merged_track_groups={o['object_id']:o['source_track_ids'] for o in objects if len(o['source_track_ids'])>1},
                 compound_regions=[o['object_id'] for o in objects if o['compound_conflicts']],
                 original_anchor_reprojection_checks=len(alignment),original_anchor_reprojection_all_pass=all(alignment),
                 source_hashes=source_hashes,accuracy_status='NOT_EVALUATED_NO_OBJECT_GROUND_TRUTH',
                 scope='One label per predicted semantic 3D region. Region grouping, missed regions, dynamic trails and physical identity still require user review.')
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
