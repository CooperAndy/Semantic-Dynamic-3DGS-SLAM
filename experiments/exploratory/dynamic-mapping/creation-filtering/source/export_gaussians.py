from pathlib import Path
import numpy as np,hashlib,json
from plyfile import PlyData,PlyElement
out=Path('/root/autodl-tmp/stage2_round16_gaussian_export_v1');out.mkdir(exist_ok=False);sources={'E6a_seq00':'/root/autodl-tmp/SplaTAM_e6b/experiments/KITTI/seq00_e6a_confhard_holdout_200_0/params.npz','Scene02_D0':'/root/autodl-tmp/stage2_round16_scene02_v1/maps/D0/params.npz','Scene02_D3':'/root/autodl-tmp/stage2_round16_scene02_v1/maps/D3/params.npz'};records=[];names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for label,path in sources.items():
 p=Path(path);z=np.load(p);n=len(z['means3D']);v=np.empty(n,dtype=[(x,'<f4') for x in names]);means=z['means3D'];rgb=z['rgb_colors'];rot=z['unnorm_rotations'];scale=z['log_scales'];scale=np.tile(scale,(1,3)) if scale.shape[1]==1 else scale
 for i,k in enumerate(['x','y','z']):v[k]=means[:,i]
 for k in ['nx','ny','nz']:v[k]=0
 for i in range(3):v['f_dc_'+str(i)]=(rgb[:,i]-.5)/0.28209479177387814;v['scale_'+str(i)]=scale[:,i]
 v['opacity']=z['logit_opacities'].reshape(-1)
 for i in range(4):v['rot_'+str(i)]=rot[:,i]
 dst=out/(label+'.ply');PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(dst));read=PlyData.read(str(dst));assert len(read['vertex'])==n and np.array_equal(read['vertex']['x'],means[:,0]);extra={k:z[k] for k in ['cam_unnorm_rots','cam_trans','intrinsics','w2c','timestep','tsdd_gaussian_states','tsdd_gaussian_track_ids'] if k in z.files};np.savez_compressed(out/(label+'_metadata.npz'),**extra);records.append({'name':label,'source':str(p),'source_sha256':h(p),'source_bytes':p.stat().st_size,'gaussians':n,'ply':str(dst),'ply_sha256':h(dst),'ply_bytes':dst.stat().st_size,'metadata_sha256':h(out/(label+'_metadata.npz')),'caveat':'PLY stores 3DGS DC color, log scale, opacity logit and unnormalized quaternion; original NPZ retains complete source parameters'})
(out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
