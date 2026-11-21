"""Gradient-scale matching on exposed A200 DEV, not sensor-noise calibration."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch

A=Path('/root/autodl-tmp/A1_gaussian_confidence');O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');G=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957');R=Path('/root/autodl-tmp/stage3_g15_20261028_p47149')
CALIB0=Path('/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/00/calib.txt'); CALIB8=Path('/root/autodl-tmp/stage3_g05B_20261025/input_view/calib.txt')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def stereo(path,fx):
 lines=path.read_text().splitlines();p={l.split(':')[0]:np.array([float(x) for x in l.split(':')[1].split()]).reshape(3,4) for l in lines if l.startswith(('P2:','P3:'))}
 centers={k:-np.linalg.solve(v[:,:3],v[:,3]) for k,v in p.items()};baseline=float(abs(centers['P3'][0]-centers['P2'][0]));return {'f_px':float(fx),'baseline_x_m':baseline,'fb_px_m':float(fx*baseline),'calib_sha256':sha(path),'raw_projection_tx_difference':float(p['P2'][0,3]-p['P3'][0,3])}
def main():
 start=time.monotonic();torch.set_num_threads(2);out=O/'gradient_calibration_20261028.json';assert not out.exists()
 snapshot=A/'c_params99_20261027.npz';meta=A/'c_render_metadata.npz'
 if sha(snapshot)!='f936b3c7b15a14a8f14e3dff77d848bd6b2228a45c2068d71fa9d2a9e1c6d81e' or sha(meta)!='fb1abb863938a41f38ace9c652f96f1345031915cfc7b3777053293ed000c85f':raise RuntimeError('snapshot or metadata mismatch')
 ma=json.loads((G/'A200/inputs/runtime/input_manifest.json').read_text());mb=json.loads((G/'B200/inputs/runtime/input_manifest.json').read_text());pose=Path(ma['pose_file'])
 if sha(pose)!=ma['pose_sha256']:raise RuntimeError('pose hash mismatch')
 sa=stereo(CALIB0,ma['K'][0][0]);sb=stereo(CALIB8,mb['K'][0][0])
 with np.load(snapshot) as a:p={k:torch.tensor(a[k],device='cuda') for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales')}
 p['means3D']=torch.nn.Parameter(p['means3D']);p.update(cam_unnorm_rots=torch.nn.Parameter(torch.zeros((1,4,1),device='cuda')),cam_trans=torch.nn.Parameter(torch.zeros((1,3,1),device='cuda')))
 with np.load(meta) as a:K=a['intrinsics'].copy();ref=a['w2c'].copy();h=int(a['org_height']);w=int(a['org_width'])
 if not np.allclose(K,ma['K'],atol=1e-4) or (h,w)!=(ma['H'],ma['W']):raise RuntimeError('camera mismatch')
 poses=np.tile(np.eye(4),(200,1,1));poses[:,:3,:]=np.loadtxt(pose).reshape(200,3,4);W=np.linalg.inv(poses)
 sys.path.insert(0,str(O));sys.path.insert(0,str(R/'source'));sys.path.insert(0,str(R/'native'))
 from a2_losses import robust_loss
 import external_tracking as ext
 from utils.recon_helpers import setup_camera
 spec=importlib.util.spec_from_file_location('a2_cal_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 cam=setup_camera(w,h,K,ref);ref_t=torch.tensor(ref,device='cuda');results=[];input_hashes={}
 for frame in (85,90,95,98):
  row=ma['rows'][frame]
  for key in ('depth','confidence'):
   path=Path(row[key]);input_hashes[f'{frame}_{key}']=sha(path)
   if input_hashes[f'{frame}_{key}']!=row['hashes'][key]:raise RuntimeError('input mismatch')
  depth=torch.tensor(np.load(row['depth']),device='cuda');conf=torch.tensor(np.load(row['confidence']),device='cuda');ext.set_external(p,torch.tensor(W[frame],device='cuda',dtype=torch.float32),0,core)
  tr=core.transform_to_frame(p,0,gaussians_grad=True,camera_grad=False);ds,_,_=core.Renderer(raster_settings=cam)(**core.transformed_params2depthplussilhouette(p,ref_t,tr))
  mask=(depth>0)&(~torch.isnan(ds[0]))&(~torch.isnan(ds[2]-ds[0]**2));res=ds[0][mask]-depth[mask];d=depth[mask];c=conf[mask]
  if int(mask.sum())<1000:raise RuntimeError('insufficient sample')
  losses={'B0':res.abs().mean(),'B1':robust_loss(res,d,c,sa['fb_px_m'],'fixed'),'B2':robust_loss(res,d,c,sa['fb_px_m'],'distance'),'B3':robust_loss(res,d,c,sa['fb_px_m'],'confidence')}
  norms={}
  for name,loss in losses.items():
   grad=torch.autograd.grad(loss,p['means3D'],retain_graph=True)[0]
   norm=float(grad.norm());
   if not np.isfinite(norm) or norm<=0:raise RuntimeError('invalid gradient norm')
   norms[name]=norm
  results.append({'frame':frame,'mask_count':int(mask.sum()),'loss':{k:float(v.detach()) for k,v in losses.items()},'means3D_gradient_norm':norms})
  del ds,tr,res,d,c,losses,grad
 gamma={'B0':1.,'B4':1.}
 for name in ('B1','B2','B3'):
  vals=[r['means3D_gradient_norm']['B0']/r['means3D_gradient_norm'][name] for r in results]
  gamma[name]=float(np.median(vals))
  if not .1<=gamma[name]<=10:raise RuntimeError(f'gamma out of range {name}')
 if torch.cuda.max_memory_allocated()>4*(1<<30) or time.monotonic()-start>120:raise RuntimeError('budget')
 result={'status':'PASS_GRADIENT_SCALE_MATCH_ONLY','gamma':gamma,'calibration':results,'stereo_00':sa,'stereo_08':sb,'inputs_sha256':input_hashes,'snapshot_sha256':sha(snapshot),'metadata_sha256':sha(meta),'pose_sha256':sha(pose),'sigma_d_px_assumed':1.,'loss_scale_ref_m':.3,'scale_clip_m':[.1,3.],'confidence_mapping':'1+2(1-c)','scope':'gradient magnitude match only; no independent sensor-noise calibration','peak_cuda_bytes':torch.cuda.max_memory_allocated(),'elapsed_s':time.monotonic()-start,'quality_adoption':False}
 out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
