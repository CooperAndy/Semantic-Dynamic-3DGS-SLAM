"""Export map-aligned semantic PLY and auditable per-frame vehicle snapshots."""
from pathlib import Path
import hashlib,json,shutil
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from export_s3_assets_reference import prepare_records,write_ply,CITYSCAPES_COLORS,UNKNOWN_COLOR,EXPECTED_CLASS_NAMES
R=Path(__file__).resolve().parent.parent
B=R.parent
MAP_SHA=json.loads((B/'field_verification.json').read_text())['F1_sha256']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def anchor(mask,depth,pose,ks):
    yy,xx=np.nonzero(mask&np.isfinite(depth)&(depth>=.5)&(depth<=40))
    if len(xx)<20:return {'display_estimate':False,'anchor_status':'insufficient_valid_depth','viewer_xyz':None,'world_xyz':None,'anchor_valid_pixel_count':int(len(xx))}
    z=depth[yy,xx];med=float(np.median(z));keep=np.abs(z-med)<=max(1.,.1*med);xx,yy,z=xx[keep],yy[keep],z[keep]
    if len(xx)<20:return {'display_estimate':False,'anchor_status':'insufficient_median_band_depth','viewer_xyz':None,'world_xyz':None,'anchor_valid_pixel_count':int(len(xx))}
    cam=np.stack(((xx-ks[0,2])*z/ks[0,0],(yy-ks[1,2])*z/ks[1,1],z),axis=1)
    rot=np.asarray(pose['rotation']);pos=np.asarray(pose['position']);world=cam@rot.T+pos;point=np.median(world,axis=0)
    method='median_band_3d_coordinate_median'
    def project(p):
        c=(p-pos)@rot;return np.array([ks[0,0]*c[0]/c[2]+ks[0,2],ks[1,1]*c[1]/c[2]+ks[1,2]])
    uv=project(point);u,v=np.rint(uv).astype(int)
    if not(0<=u<1226 and 0<=v<370 and mask[v,u]):
        point=world[np.argmin(np.linalg.norm(world-point,axis=1))];uv=project(point);u,v=np.rint(uv).astype(int);method='valid_sample_nearest_3d_median'
    assert 0<=u<1226 and 0<=v<370 and mask[v,u]
    return dict(display_estimate=True,anchor_verified=False,anchor_status='reprojects_in_instance_mask_not_verified_map_object_binding',anchor_method=method,anchor_valid_pixel_count=int(len(xx)),anchor_depth_median_m=med,world_xyz=point.tolist(),viewer_xyz=[-float(point[0]),-float(point[1]),float(point[2])],reprojection_uv=uv.tolist())

def main():
    root=R/'delivery';root.mkdir(exist_ok=False)
    sem=root/'KITTI09_F1_Semantic';sem.mkdir()
    motion=root/'KITTI09_MotionDemo';motion.mkdir();(motion/'frames').mkdir();(motion/'reference').mkdir()
    param=B/'runs/F1_appearance800/params.npz';assert sha(param)==MAP_SHA
    with np.load(param,allow_pickle=False) as f:p={k:f[k] for k in ('means3D','rgb_colors','log_scales','unnorm_rotations','logit_opacities','intrinsics')}
    with np.load(R/'fusion/semantic_gaussian_labels.npz',allow_pickle=False) as f:
        labels=f['label'];indices=f['source_gaussian_index'];names=tuple(map(str,f['class_names'].tolist()))
    assert len(labels)==len(p['means3D']) and names==EXPECTED_CLASS_NAMES and np.array_equal(indices,np.arange(len(labels)))
    colors=np.tile(UNKNOWN_COLOR,(len(labels),1));known=labels>=0;colors[known]=CITYSCAPES_COLORS[labels[known]]
    records=prepare_records(p['means3D'],colors,p['log_scales'],p['unnorm_rotations'],p['logit_opacities'])
    ply=sem/'KITTI09_F1_Semantic.ply';write_ply(ply,records)
    with ply.open('r+b') as f:
        head=f.read(1000);f.seek(0);f.write(head.replace(b'KITTI 21 frames 1200-1399',b'KITTI 09 frames 0300-0499'))
    # Confirm the semantic PLY retains the source geometry, opacity and point order.
    for name,values in [('x',p['means3D'][:,0]),('y',p['means3D'][:,1]),('z',p['means3D'][:,2]),('opacity',p['logit_opacities'][:,0])]:assert np.array_equal(records[name],values)
    for i in range(3):assert np.array_equal(records[f'scale_{i}'],p['log_scales'][:,0] if p['log_scales'].shape[1]==1 else p['log_scales'][:,i])
    for i in range(4):assert np.array_equal(records[f'rot_{i}'],p['unnorm_rotations'][:,i])
    shutil.copy2(R/'fusion/semantic_gaussian_labels.npz',sem/'semantic_gaussian_labels.npz')
    shutil.copy2(R/'fusion/semantic_fusion_summary.json',sem/'semantic_fusion_summary.json')
    shutil.copy2(R/'fusion/semantic_class_histogram.csv',sem/'semantic_class_histogram.csv')
    manifest=dict(sequence='09',source_frames=[0,199],gaussian_count=len(labels),assigned_count=int(known.sum()),unknown_count=int((~known).sum()),source_params_sha256=MAP_SHA,source_point_indices=f'0..{len(labels)-1}, unchanged order',same_geometry_opacity_and_point_order=True,semantic_ply_sha256=sha(ply),class_names=list(names),scope='Cityscapes-19 predicted semantic fusion; category is not measured motion; no semantic GT accuracy evaluation')
    (sem/'semantic_manifest.json').write_text(json.dumps(manifest,indent=2))
    poses=json.loads((B/'viewer_assets/KITTI09_F200/KITTI09_F200_camera_poses.json').read_text());assert len(poses)==200
    frame_entries=[];snapshots=[];all_counts=Counter();anchors=0
    for t in range(200):
        raw=json.loads((R/f'motion/{t:06d}.json').read_text());masks=np.load(R/f'instances/{t:06d}_masks.npz')['masks'];depth=np.load(B/f'raft_left/{t:06d}_depth.npy')
        objects=[]
        for obj in raw['objects']:
            observations=[]
            for previous in range(max(0,t-7),t+1):
                prior=json.loads((R/f'motion/{previous:06d}.json').read_text())
                match=next((o for o in prior['objects'] if o['track_id']==obj['track_id']),None)
                if match is not None:observations.append(dict(frame=previous,score=match['score'],state=match['raw_state']))
            state={'dynamic':'动态','static':'静态','unknown':'未知'}[obj['raw_state']]
            o={**obj,'state':state,'observations':observations,**anchor(masks[obj['mask_index']].astype(bool),depth,poses[t],p['intrinsics'])}
            all_counts[obj['raw_state']]+=1;anchors+=int(o['display_estimate']);objects.append(o)
        ref=Image.open(B/f'sequences/09/image_2/{t:06d}.png').convert('RGB');draw=ImageDraw.Draw(ref)
        for obj in objects:
            color={'dynamic':'#f28c28','static':'#368cf5','unknown':'#aaaaaa'}[obj['raw_state']];box=obj['bbox'];draw.rectangle(box,outline=color,width=2)
            score=f"{obj['score']:.3f}" if obj['score'] is not None else 'unknown-history'
            text=f"{obj['category']} #{obj['track_id']} {obj['raw_state']} {score}";x,y=box[0],max(0,box[1]-15);draw.rectangle((x,y,min(1225,x+len(text)*6+4),y+14),fill='#132035');draw.text((x+2,y+1),text,fill=color)
        ref.save(motion/f'reference/{t:06d}.jpg',quality=90)
        counts=dict(Counter(o['raw_state'] for o in objects));visible=[o for o in objects if o['display_estimate']]
        snap=dict(dataset='KITTI odometry',sequence='09',scene_id='kitti09_orb_f200_full',camera='image_2',reference_frame=t,image_width=1226,image_height=370,map='F1/F200 same geometry',source_map_sha256=MAP_SHA,depth='same-map RAFT-Stereo left metric Z',motion_model='Pi3MOS',threshold=.5,window=raw['window'],history_note='所选区间前7帧没有完整的区间内历史，状态未知。' if t<7 else '当前帧及前 7 帧，真实 8 帧因果窗口。',reference_image=f'scene-data/KITTI09_MotionDemo/reference/{t:06d}.jpg',position_note='Predicted locations from instance masks and RAFT-Stereo depth; reprojection checks pass, no verified object-level Gaussian binding or occlusion handling; original RGB map is unchanged.',objects=objects)
        (motion/f'frames/{t:06d}.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding='utf-8')
        states=set(o['raw_state'] for o in visible)
        frame_entries.append(dict(frame=t,snapshot=f'scene-data/KITTI09_MotionDemo/frames/{t:06d}.json',vehicles=len(objects),anchors=len(visible),counts=counts,history_available=t>=7,both_states='dynamic' in states and 'static' in states,visible_mask_area=sum(o['mask_area'] for o in visible)))
        snapshots.append(snap)
    candidates=[f for f in frame_entries if f['history_available'] and f['anchors']]
    chosen=max(candidates,key=lambda f:(f['both_states'],f['anchors'],f['visible_mask_area'],-abs(f['frame']-99))) if candidates else frame_entries[99]
    index=dict(scene_id='kitti09_orb_f200_full',sequence='09',source_map_sha256=MAP_SHA,default_frame=chosen['frame'],selection='Default frame chosen after predictions for illustration: prefer both predicted states if present, then valid depth anchors and mask area; not random sampling or accuracy evidence.',frames=frame_entries,scope='offline per-frame vehicle predictions and estimated anchors; no map updates or motion-driven deletion')
    (motion/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    (motion/'snapshot.json').write_text(json.dumps(snapshots[chosen['frame']],ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(R/'motion/summary.json',motion/'prediction_summary.json')
    report=dict(semantic=manifest,motion_counts=dict(all_counts),vehicle_frame_observations=sum(all_counts.values()),estimated_anchor_observations=anchors,default_frame=chosen['frame'],default_vehicle_count=chosen['vehicles'],default_anchor_count=chosen['anchors'],default_both_predicted_states=chosen['both_states'],original_map_sha256_unchanged=sha(param)==MAP_SHA)
    (root/'annotation_report.json').write_text(json.dumps(report,indent=2))
    files={str(q.relative_to(root)):{'bytes':q.stat().st_size,'sha256':sha(q)} for q in root.rglob('*') if q.is_file()}
    (root/'files_manifest.json').write_text(json.dumps(files,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
