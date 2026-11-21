"""CPU-only reuse of full seq10 Pi3MOS probabilities on the original TMD map.

No model inference, GPU allocation, training or source-data modification.
The old TSDD scores are not interpreted as Pi3MOS motion probabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def anchor(mask,depth,pose,k):
    y,x=np.nonzero(mask&np.isfinite(depth)&(depth>=.5)&(depth<=40))
    if len(x)<20:return dict(display_estimate=False,world_xyz=None,viewer_xyz=None,anchor_status='insufficient_valid_depth')
    z=depth[y,x];m=float(np.median(z));keep=np.abs(z-m)<=max(1.,.1*m);x,y,z=x[keep],y[keep],z[keep]
    if len(x)<20:return dict(display_estimate=False,world_xyz=None,viewer_xyz=None,anchor_status='insufficient_median_band_depth')
    c=np.stack(((x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z),axis=1)
    rotation=np.asarray(pose['rotation']);position=np.asarray(pose['position']);world=c@rotation.T+position;point=np.median(world,axis=0)
    def project(p):
        q=(p-position)@rotation;return np.array([k[0,0]*q[0]/q[2]+k[0,2],k[1,1]*q[1]/q[2]+k[1,2]])
    uv=project(point);u,v=np.rint(uv).astype(int);h,w=mask.shape
    if not(0<=u<w and 0<=v<h and mask[v,u]):
        point=world[np.argmin(np.linalg.norm(world-point,axis=1))];uv=project(point);u,v=np.rint(uv).astype(int)
    assert 0<=u<w and 0<=v<h and mask[v,u]
    return dict(display_estimate=True,world_xyz=point.tolist(),viewer_xyz=[-float(point[0]),-float(point[1]),float(point[2])],anchor_verified=False,anchor_status='raw_depth_anchor_reprojects_inside_mask_not_semantic_object_binding',anchor_valid_pixel_count=len(x),reprojection_uv=uv.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--poses',required=True)
    p.add_argument('--depth-dir',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    source=Path('/root/autodl-tmp/s23_effects_20261105_p16247/S23P_20261110')
    pred=source/'inputs/10';instances=source/'restored/bundles/10/instances';depthdir=Path(a.depth_dir)
    params=Path('/root/autodl-tmp/t07_tmd_extension_20261111/params10.npz')
    expected='e7f67d8dab6345f4e629abf68a5541235dd187c99b9c871cc6f7f4bcb8ac1b47'
    assert sha(params)==expected
    runtime=json.loads((pred/'runtime.json').read_text());complete=json.loads((pred/'prediction_complete.json').read_text())
    assert complete['frames']==len(runtime['rows'])==1000 and complete['predicted_windows']==993
    assert complete['manifest_sha256']==sha(pred/'observations/manifest.json')
    original_scores=[json.loads(line) for line in (pred/'vehicle_scores.jsonl').read_text().splitlines()]
    assert len(original_scores)==1000
    producer=json.loads((pred/'observations/manifest.json').read_text())['producer']
    assert producer['vehicle_motion']=='Pi3MOS eight-frame causal; median>=0.5'
    poses={p['source_frame']:p for p in json.loads(Path(a.poses).read_text())};assert set(poses)==set(range(1000))
    with np.load(params,allow_pickle=False) as z:k=z['intrinsics'];width,height=int(z['org_width']),int(z['org_height'])
    assert (width,height)==(1226,370) and np.allclose(k,runtime['K'],atol=.0001)
    for d in ['frames','object_inputs','object_inputs/instances','object_inputs/masked_depth']:(out/d).mkdir()
    provenance=[];raw_counts={'static':0,'dynamic':0,'unknown':0};start=time.monotonic()
    for n,row in enumerate(runtime['rows']):
        if n%25==0:
            if shutil.disk_usage(out).free<8*2**30:raise RuntimeError('Disk headroom below 8 GiB')
            save(out/'status.json',dict(stage='preparing',completed=n,total=1000,elapsed_seconds=time.monotonic()-start))
        assert row['frame_id']==row['source_frame_id']==n
        rgb=Path(row['rgb']);assert sha(rgb)==row['hashes']['rgb']
        mp=instances/f'{n:06d}_masks.npz';jp=instances/f'{n:06d}.json';dp=depthdir/f'{n:06d}_depth.npy'
        metadata=json.loads(jp.read_text())
        with np.load(mp,allow_pickle=False) as z:masks=z['masks']
        depth=np.load(dp,allow_pickle=False);assert depth.shape==masks.shape[1:]==(height,width)
        for instance in metadata['instances']:assert int(masks[instance['mask_index']].sum())==instance['mask_area']
        score_record=original_scores[n]
        assert score_record['frame']==n and score_record['window']==(list(range(n-7,n+1)) if n>=7 else [])
        score_by_id={o['track_id']:o for o in score_record['vehicles']}
        pp=None
        if n>=7:
            pp=pred/f'probabilities/{n:06d}.npz'
            with np.load(pp,allow_pickle=False) as z:low=z['probability'].astype('float32')
            assert low.shape==(280,910) and np.isfinite(low).all() and low.min()>=0 and low.max()<=1
        objects=[]
        for instance in metadata['instances']:
            if instance['category'] not in ('car','truck','bus','van'):continue
            mask=masks[instance['mask_index']].astype(bool)
            score_source=score_by_id[instance['track_id']]
            assert score_source['mask_index']==instance['mask_index'] and score_source['category']==instance['category']
            score=score_source['score']
            state=('dynamic' if score>=.5 else 'static') if score is not None else 'unknown';raw_counts[state]+=1
            assert state==score_source['raw_state']
            values=depth[mask];values=values[np.isfinite(values)&(values>=.5)&(values<=40)]
            obj=dict(track_id=instance['track_id'],category=instance['category'],bbox=instance['bbox'],mask_index=instance['mask_index'],mask_area=instance['mask_area'],score=score,raw_state=state,state={'static':'静态','dynamic':'动态','unknown':'未知'}[state],depth_m=float(np.median(values)) if len(values)>=20 else None,valid_depth_pixels=len(values),range_40m=len(values)>=20,reason='insufficient_real_history' if n<7 else None)
            obj.update(anchor(mask,depth,poses[n],k));objects.append(obj)
        snap=dict(dataset='KITTI odometry',sequence='10',scene_id='kitti10_tmd_tsdd_s2',camera='image_2',reference_frame=n,image_width=width,image_height=height,source_map_sha256=expected,motion_model='Pi3MOS',threshold=.5,window=list(range(n-7,n+1)) if n>=7 else [],history_note='Current and previous seven frames; full-interval offline display processing.',depth='Original TMD bundle metric RAFT depth, 0.5-40m; fixed map cameras.',objects=objects)
        save(out/f'frames/{n:06d}.json',snap)
        union=masks.any(0) if len(masks) else np.zeros(depth.shape,bool)
        values=np.where(union&np.isfinite(depth)&(depth>=.5)&(depth<=40),depth,0).astype('float32')
        compact=out/f'object_inputs/masked_depth/{n:06d}.npz';np.savez_compressed(compact,depth=values)
        shutil.copy2(mp,out/f'object_inputs/instances/{n:06d}_masks.npz');shutil.copy2(jp,out/f'object_inputs/instances/{n:06d}.json')
        provenance.append(dict(source_frame=n,mask_sha256=sha(mp),instance_sha256=sha(jp),source_depth_sha256=sha(dp),masked_depth_sha256=sha(compact),rgb_sha256=sha(rgb),probability_sha256=sha(pp) if pp else None))
        if n%100==0:print('PREPARED',n,'elapsed_s',round(time.monotonic()-start,1),flush=True)
    save(out/'object_inputs/manifest.json',dict(sequence='10',source_map_sha256=expected,intrinsics=k.tolist(),frames=provenance,depth_range_m=[.5,40.],source=str(source),depth_source=str(depthdir),depth_policy='Existing metric RAFT float32 depth from the same original RGB frames, restricted only to instance masks and 0.5-40m for object association. No new inference; original map unchanged.',pi3mos_probability_source=str(pred/'probabilities'),prediction_manifest_sha256=sha(pred/'observations/manifest.json'),runtime_sha256=sha(pred/'runtime.json'),original_vehicle_scores_sha256=sha(pred/'vehicle_scores.jsonl'),producer=producer,score_policy='Original float32 full-instance Pi3MOS median scores, same causal8/median>=0.5 policy as seq09. No float16 rescore and no TSDD scores used.'))
    save(out/'summary.json',dict(frames=1000,raw_state_counts=raw_counts,elapsed_seconds=time.monotonic()-start,model_inference_run=False,gpu_used=False,source_map_sha256=expected,original_prediction_complete=complete,source_pose_sha256=sha(a.poses)))
    save(out/'status.json',dict(stage='packing',completed=1000,total=1000))
    archive=out.parent/'seq10_object_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['frames','object_inputs','summary.json']:tar.add(out/name,arcname=f'seq10_prepared/{name}')
    save(out/'status.json',dict(stage='complete',completed=1000,total=1000,archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size))
    print(json.dumps(json.loads((out/'status.json').read_text())),flush=True)


if __name__=='__main__':main()
