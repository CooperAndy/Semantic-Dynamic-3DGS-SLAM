from pathlib import Path
import numpy as np,json,hashlib,cv2
from plyfile import PlyData
r=Path('/root/autodl-tmp/stage2_round16_gaussian_export_v1');out=[]
for row in json.loads((r/'manifest.json').read_text()):
 p=Path(row['ply']);source=Path(row['source']);z=np.load(source);v=PlyData.read(str(p))['vertex'].data;required=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];assert all(k in v.dtype.names for k in required);n=len(v);assert n==len(z['means3D'])==row['gaussians'];sample=np.linspace(0,n-1,min(n,10000),dtype=int);rgb=np.stack([v[f'f_dc_{i}'][sample] for i in range(3)],1)*.28209479177387814+.5;rot=np.stack([v[f'rot_{i}'][sample] for i in range(4)],1);scale=np.stack([v[f'scale_{i}'][sample] for i in range(3)],1);orig_scale=z['log_scales'][sample];orig_scale=np.tile(orig_scale,(1,3)) if orig_scale.shape[1]==1 else orig_scale;diff={'rgb_max':float(np.max(np.abs(rgb-z['rgb_colors'][sample]))),'log_scale_max':float(np.max(np.abs(scale-orig_scale))),'rot_max':float(np.max(np.abs(rot-z['unnorm_rotations'][sample]))),'opacity_logit_max':float(np.max(np.abs(v['opacity'][sample]-z['logit_opacities'][sample,0])))};assert all(np.isfinite(v[k]).all() for k in required);assert diff['rgb_max']<1e-6 and max(diff['log_scale_max'],diff['rot_max'],diff['opacity_logit_max'])==0;norm=np.linalg.norm(rot,axis=1);out.append({'name':row['name'],'vertices':n,'fields':required,'sample':len(sample),'max_diff':diff,'sample_quaternion_norm_min':float(norm.min()),'sample_quaternion_norm_max':float(norm.max()),'semantic_labels_in_ply':False,'viewer_import_test':'NOT_PERFORMED'})
video=r/'Scene02_D0_D3_trajectory.mp4';cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));read=0
while True:
 ok,im=cap.read()
 if not ok:break
 read+=1
assert read==frames==26;(Path('/root/autodl-tmp/stage2_round17_delivery_v1')/'ply_verification.json').write_text(json.dumps({'ply':out,'video_frames_decoded':read,'viewer':'no installed 3DGS viewer, not tested'},indent=2));print(json.dumps({'ply':out,'video_frames_decoded':read},indent=2))
