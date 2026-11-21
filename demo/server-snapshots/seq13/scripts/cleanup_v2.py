"""CPU source-frame vehicle/motion association; complements temporal fused labels."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
import cv2

B=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115')
A=B/'annotations_20261107'
O=B/'vehicle_cleanup_v2_20261107'
SOURCE_SHA=json.loads((B/'field_verification.json').read_text())['F1_sha256']

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    O.mkdir(exist_ok=True);source=B/'runs/F1_appearance800/params.npz'
    assert sha(source)==SOURCE_SHA
    p=np.load(source);xyz=p['means3D'];born=p['timestep'].ravel().astype(int);K=p['intrinsics']
    labels=np.load(A/'fusion/semantic_gaussian_labels.npz')['label']
    old=np.load(B/'vehicle_cleanup_20261107/vehicle_removal_mask.npz')['remove']
    poses=json.loads((B/'viewer_assets/KITTI09_F200/KITTI09_F200_camera_poses.json').read_text())
    source_vehicle=np.zeros(len(xyz),bool);dense_dynamic=source_vehicle.copy();instance_dynamic=source_vehicle.copy()
    source_semantic=np.full(len(xyz),-1,np.int16);source_motion=np.full(len(xyz),np.nan,np.float16)
    rows=[];spot=[];target_ids=set()
    for t,pose in enumerate(poses):
        ids=np.flatnonzero(born==t)
        cam=(xyz[ids]-np.asarray(pose['position']))@np.asarray(pose['rotation'])
        z=cam[:,2];safe=np.maximum(z,1e-6)
        u=np.rint(K[0,0]*cam[:,0]/safe+K[0,2]).astype(np.int64)
        v=np.rint(K[1,1]*cam[:,1]/safe+K[1,2]).astype(np.int64)
        valid=(z>=.5)&(z<=40)&(u>=0)&(u<1241)&(v>=0)&(v<376)
        loc=np.flatnonzero(valid);uu,vv=u[loc],v[loc]
        sem=np.load(A/f'semantic/labels/{t:06d}_semantic.npy')
        conf=np.load(A/f'semantic/confidence/{t:06d}_confidence.npy')
        depth=np.load(B/f'raft_left/{t:06d}_depth.npy')
        prob=cv2.resize(np.load(A/f'motion/{t:06d}.npz')['prob'].astype(np.float32),(1241,376),interpolation=cv2.INTER_LINEAR) if t>=7 else np.full((376,1241),np.nan)
        source_semantic[ids[loc]]=sem[vv,uu];source_motion[ids[loc]]=prob[vv,uu]
        for idx in target_ids.intersection(ids.tolist()):
            j=int(np.searchsorted(ids,idx));ux,vy=int(u[j]),int(v[j])
            spot.append(dict(index=idx,birth=t,uv=[ux,vy],semantic=int(sem[vy,ux]),confidence=float(conf[vy,ux]),motion=float(prob[vy,ux]),source_depth=float(depth[vy,ux]),center_depth=float(z[j])))
        observed=depth[vv,uu]
        keep=np.isfinite(observed)&(observed>=.5)&(observed<=40)&(np.abs(z[loc]-observed)<=.35+.08*observed)
        keep &= np.isin(sem[vv,uu],[13,14,15])&(conf[vv,uu]>=.5)
        loc,uu,vv=loc[keep],uu[keep],vv[keep]
        hit=ids[loc];source_vehicle[hit]=True
        masks=np.load(A/f'instances/{t:06d}_masks.npz')['masks']
        raw=json.loads((A/f'motion/{t:06d}.json').read_text())
        dyn=np.zeros(len(hit),bool);sta=dyn.copy()
        for obj in raw['objects']:
            if obj['category'] not in ('car','truck','bus','van'):continue
            if obj['raw_state']=='dynamic':dyn |= masks[obj['mask_index'],vv,uu].astype(bool)
            elif obj['raw_state']=='static':sta |= masks[obj['mask_index'],vv,uu].astype(bool)
        dense_dynamic[hit]=np.isfinite(prob[vv,uu])&(prob[vv,uu]>=.5)&~sta
        instance_dynamic[hit]=dyn&~sta
        rows.append(dict(frame=t,source_vehicle=len(hit),dense_dynamic=int(dense_dynamic[hit].sum()),instance_dynamic=int(instance_dynamic[hit].sum())))
    added=(dense_dynamic|instance_dynamic)&~old
    remove=old|dense_dynamic|instance_dynamic
    assert np.all(source_vehicle[added]) and np.all(np.isin(source_semantic[added],[13,14,15]))
    np.savez_compressed(O/'vehicle_removal_mask.npz',remove=remove,legacy_remove=old,
        source_vehicle=source_vehicle,source_dense_dynamic=dense_dynamic,
        source_instance_dynamic=instance_dynamic,source_semantic=source_semantic,source_motion=source_motion)
    receipt=dict(schema='vehicle-cleanup-v2',sequence='09',source_frames=[0,199],source_params_sha256=SOURCE_SHA,
        source_count=len(xyz),legacy_removed_count=int(old.sum()),added_count=int(added.sum()),
        removed_count=int(remove.sum()),kept_count=int((~remove).sum()),
        source_vehicle_candidates=int(source_vehicle.sum()),source_dense_dynamic_count=int(dense_dynamic.sum()),
        source_instance_dynamic_count=int(instance_dynamic.sum()),
        added_fused_labels={str(k):v for k,v in Counter(map(int,labels[added])).items()},
        added_source_labels={str(k):v for k,v in Counter(map(int,source_semantic[added])).items()},
        added_without_source_vehicle_evidence=0,mask_sha256=sha(O/'vehicle_removal_mask.npz'),
        algorithm=dict(device='CPU; reuses existing predictions; no new GPU task',
            legacy='union with v1 conservative multi-frame fused-vehicle mask',
            source='timestep frame; saved map poses, intrinsics and exact centre projection',
            vehicle='source image Cityscapes car/truck/bus and confidence>=0.5; no dilation or bounding-box deletion',
            depth='RAFT-Stereo Z 0.5..40m and abs(centreZ-depth)<=0.35+0.08*depth',
            motion='source pixel Pi3MOS probability>=0.5 OR exact dynamic-vehicle instance mask; explicitly static instance pixels protected',
            history='source frames0..6 cannot add points: no complete motion history'),
        scope='Reversible predicted vehicle filtering. Final static-world semantic voting may relabel source vehicle ghosts as road/unknown; source-frame category is the ownership gate for additions. Not object GT or a reconstruction-quality claim.',
        per_frame=rows,large_splat_diagnosis=spot)
    (O/'mask_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('per_frame','large_splat_diagnosis')}))
    print('large_splat_diagnosis',json.dumps(spot))

if __name__=='__main__':main()
