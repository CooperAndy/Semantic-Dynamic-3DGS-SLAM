"""CPU-only, conservative per-Gaussian vehicle/motion association for a display derivative."""
import hashlib, json
from pathlib import Path
import numpy as np

B = Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114/seed42_demo_build')
A = B / 'annotations_20261107'
O = B / 'vehicle_cleanup_20261107'
SOURCE_SHA=json.loads((B/'field_verification.json').read_text())['F1_sha256']

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def main():
    O.mkdir(exist_ok=True)
    source = B / 'runs/F1_appearance800/params.npz'
    assert sha(source) == SOURCE_SHA
    with np.load(source, allow_pickle=False) as p:
        xyz, born, K = p['means3D'], p['timestep'].reshape(-1), p['intrinsics']
    with np.load(A/'fusion/semantic_gaussian_labels.npz') as s:
        labels = s['label']; assert np.array_equal(s['source_gaussian_index'], np.arange(len(xyz)))
    candidate = np.flatnonzero(np.isin(labels, [13, 14, 15]))
    points = xyz[candidate].astype(np.float64)
    poses = json.loads((B/'viewer_assets/KITTI09_F200/KITTI09_F200_camera_poses.json').read_text())
    dynamic = np.zeros(len(candidate), np.uint16)
    static = dynamic.copy(); conflict = dynamic.copy()
    per_frame = []
    for t, pose in enumerate(poses):
        raw = json.loads((A/f'motion/{t:06d}.json').read_text())
        with np.load(A/f'instances/{t:06d}_masks.npz') as m: masks = m['masks']
        depth = np.load(B/f'raft_left/{t:06d}_depth.npy')
        cam = (points - np.asarray(pose['position'])) @ np.asarray(pose['rotation'])
        z = cam[:,2]; safe = np.maximum(z, 1e-6)
        u = np.rint(K[0,0]*cam[:,0]/safe + K[0,2]).astype(np.int64)
        v = np.rint(K[1,1]*cam[:,1]/safe + K[1,2]).astype(np.int64)
        valid = (z>=.5)&(z<=40)&(u>=0)&(u<1226)&(v>=0)&(v<370)&(born[candidate]<=t)
        ids = np.flatnonzero(valid); uu,vv = u[ids],v[ids]
        observed = depth[vv,uu]
        agree = np.isfinite(observed)&(observed>=.5)&(observed<=40)&(np.abs(z[ids]-observed)<=.35+.08*observed)
        ids,uu,vv = ids[agree],uu[agree],vv[agree]
        dyn = np.zeros(len(candidate),bool); sta = dyn.copy()
        for obj in raw['objects']:
            if obj['category'] not in ('car','truck','bus','van') or obj['raw_state']=='unknown': continue
            hit = ids[masks[obj['mask_index'],vv,uu].astype(bool)]
            (dyn if obj['raw_state']=='dynamic' else sta)[hit] = True
        # Each frame contributes at most one vote, even with duplicate detections.
        dynamic += dyn & ~sta; static += sta & ~dyn; conflict += dyn & sta
        per_frame.append(dict(frame=t,dynamic=int(dyn.sum()),static=int(sta.sum()),conflict=int((dyn&sta).sum())))
    # Predeclared conservative rule: two independent-frame dynamic votes and >=60%
    # dynamic support, counting ambiguous overlapping detections against deletion.
    total = dynamic.astype(np.uint32)+static+conflict
    chosen = (dynamic>=2)&(dynamic.astype(np.float64)>=.6*np.maximum(total,1))
    remove = np.zeros(len(xyz),bool); remove[candidate[chosen]] = True
    assert not np.any(remove & ~np.isin(labels,[13,14,15]))
    np.savez_compressed(O/'vehicle_removal_mask.npz',remove=remove,candidate_index=candidate,
                        dynamic_votes=dynamic,static_votes=static,conflict_votes=conflict)
    receipt = dict(schema='vehicle-cleanup-v1',sequence='09',source_frames=[0,199],
        source_params_sha256=SOURCE_SHA,source_count=len(xyz),vehicle_candidate_count=len(candidate),
        removed_count=int(remove.sum()),kept_count=int((~remove).sum()),
        removed_classes={str(k):int((remove&(labels==k)).sum()) for k in (13,14,15)},
        non_vehicle_removed=0,unknown_removed=0,mask_sha256=sha(O/'vehicle_removal_mask.npz'),
        algorithm=dict(device='CPU; no CUDA/model inference',semantic_gate='fused Cityscapes car=13,truck=14,bus=15 only',
            projection='saved map camera poses and intrinsics; centre in exact instance mask; timestep <= frame',
            depth_gate='RAFT-Stereo Z in [0.5,40] m; abs(centreZ-depth)<=0.35+0.08*depth',
            motion='existing Pi3MOS per-instance median >=0.5; first 7 frames unknown',
            vote='one per frame; dynamic/static overlap is conflict; dynamic>=2 and dynamic/(dynamic+static+conflict)>=0.60'),
        scope='Predicted association for reversible demo display. Not object-GT, retraining, motion accuracy or static-map quality evidence.',
        per_frame=per_frame)
    (O/'mask_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='per_frame'}),flush=True)

if __name__=='__main__': main()
