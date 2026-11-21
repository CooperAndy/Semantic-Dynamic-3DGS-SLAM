from pathlib import Path
import argparse,json,hashlib
import numpy as np
from export_s3_assets_reference import prepare_records,write_ply,camera_poses,trajectory_records,viewer_settings
p=argparse.ArgumentParser();p.add_argument('--params',required=True);p.add_argument('--out',required=True);p.add_argument('--name',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with np.load(a.params,allow_pickle=False) as f: x={k:f[k] for k in f.files}
count=len(x['means3D']); assert count>100000
centers,rots=camera_poses(x);assert len(centers)==200
rgb=prepare_records(x['means3D'],x['rgb_colors'],x['log_scales'],x['unnorm_rotations'],x['logit_opacities'])
traj=trajectory_records(centers,np.asarray(x['keyframe_time_indices'],dtype=np.int64))
write_ply(out/f'{a.name}_RGB.ply',rgb)
write_ply(out/f'{a.name}_Trajectory.ply',traj)
write_ply(out/f'{a.name}_RGB_with_Trajectory.ply',np.concatenate((rgb,traj)))
ks=x['intrinsics'];w=int(x['org_width']);h=int(x['org_height'])
cams=[{'id':i,'source_frame':1200+i,'img_name':f'{i:06d}.png','width':w,'height':h,'position':centers[i].astype(float).tolist(),'rotation':rots[i].astype(float).tolist(),'fx':float(ks[0,0]),'fy':float(ks[1,1]),'pose_source':'ORB-SLAM3 estimated P2 c2w'} for i in range(200)]
(out/f'{a.name}_camera_poses.json').write_text(json.dumps(cams,indent=2)+'\n')
(out/f'{a.name}_viewer_settings.json').write_text(json.dumps(viewer_settings(x['means3D']),indent=2)+'\n')
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
manifest={'sequence':'KITTI21','frames':[1200,1399],'pose_source':'ORB-SLAM3 estimated','gaussian_count':count,'params_sha256':sha(Path(a.params)),'camera_count':len(cams),'rgb_ply_sha256':sha(out/f'{a.name}_RGB.ply')}
(out/f'{a.name}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
