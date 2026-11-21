"""F1 on an isolated S23 mapper result with its own selected cameras."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
P=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116'); RASTER=Path('/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--cell',type=Path,required=True);a.add_argument('--steps',type=int,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();cell=args.cell.resolve();out=args.out
 if out.exists():raise FileExistsError(out)
 mapfile=cell/'runs/B/params.npz';trajectory=cell/'runs/B/estimated_trajectory.npz';runtime=json.loads((cell/'inputs/runtime/input_manifest.json').read_text());N=runtime['frame_count'];assert args.steps>0
 assert (cell/'runs/B/receipt.json').is_file() and json.loads((cell/'runs/B/receipt.json').read_text())['status']=='PASS_NATIVE_MAPPING_INTEGRATION'
 sys.path.insert(0,str(cell/'source'));sys.path.insert(0,str(RASTER))
 import diff_gaussian_rasterization as raster
 assert Path(raster.__file__).resolve().is_relative_to(RASTER.resolve())
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 from utils.recon_helpers import setup_camera
 import importlib.util
 spec=importlib.util.spec_from_file_location('mapping_core',cell/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 with np.load(mapfile,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 with np.load(trajectory,allow_pickle=False) as z:w2c=z['w2c'].copy()
 assert w2c.shape==(N,4,4) and np.isfinite(w2c).all();assert data['cam_trans'].shape==(1,3,N) and data['cam_unnorm_rots'].shape==(1,4,N)
 quat=data['cam_unnorm_rots'][0].T;quat=quat/np.linalg.norm(quat,axis=1)[:,None];mat=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();camera_error=max(float(np.abs(mat-w2c[:,:3,:3]).max()),float(np.abs(data['cam_trans'][0].T-w2c[:,:3,3]).max()));assert camera_error<1e-4,camera_error
 params={k:torch.from_numpy(data[k]).cuda() for k in ['means3D','rgb_colors','log_scales','unnorm_rotations','logit_opacities','cam_unnorm_rots','cam_trans']};H,W=runtime['H'],runtime['W'];K=data['intrinsics'][:3,:3]
 # Mapper renders transformed Gaussians with the first camera. F1 uses each
 # selected camera directly; compare both paths before optimization.
 first=setup_camera(W,H,K,w2c[0]);diff=[]
 with torch.no_grad():
  for i in sorted({0,N//2,N-1}):
   trans=core.transform_to_frame(params,i,gaussians_grad=False,camera_grad=False);rv=core.transformed_params2rendervar(params,trans);x,*_=Renderer(raster_settings=first)(**rv)
   direct=setup_camera(W,H,K,w2c[i]);y,*_=Renderer(raster_settings=direct)(means3D=params['means3D'],means2D=torch.zeros_like(params['means3D']),colors_precomp=params['rgb_colors'],rotations=F.normalize(params['unnorm_rotations'],dim=1),opacities=torch.sigmoid(params['logit_opacities']),scales=torch.exp(params['log_scales']).repeat(1,3) if params['log_scales'].shape[1]==1 else torch.exp(params['log_scales']))
   d=torch.abs(x-y);diff.append(dict(frame=i,mean=float(d.mean()),max=float(d.max())));del x,y,trans,rv,d
 assert max(z['mean'] for z in diff)<1e-3,('Mapper/F1 camera rendering disagrees',diff)
 colors=torch.nn.Parameter(params['rgb_colors']);logits=torch.nn.Parameter(params['logit_opacities']);optimizer=torch.optim.Adam([{'params':[colors],'lr':.001},{'params':[logits],'lr':.005}]);rot=F.normalize(params['unnorm_rotations'],dim=1);scales=torch.exp(params['log_scales']);scales=scales.repeat(1,3) if scales.shape[1]==1 else scales
 out.mkdir(parents=True,exist_ok=False);log=(out/'loss.jsonl').open('x',buffering=1);tic=time.time();torch.cuda.reset_peak_memory_stats()
 for step in range(args.steps):
  i=step%N;row=runtime['rows'][i];target=torch.from_numpy(np.asarray(Image.open(row['rgb']).convert('RGB'),dtype=np.float32)/255).cuda().permute(2,0,1);depth=np.load(row['depth'],allow_pickle=False);mask=torch.from_numpy((np.isfinite(depth)&(depth>0)).astype(np.float32)).cuda();assert mask.any()
  camera=setup_camera(W,H,K,w2c[i]);optimizer.zero_grad(set_to_none=True);rgb,*_=Renderer(raster_settings=camera)(means3D=params['means3D'],means2D=torch.zeros_like(params['means3D']),colors_precomp=colors,rotations=rot,opacities=torch.sigmoid(logits),scales=scales);loss=((rgb-target).abs()*mask[None]).sum()/(3*mask.sum());assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step%20==0 or step==args.steps-1:log.write(json.dumps(dict(step=step+1,frame=i,loss=float(loss.detach())))+'\n')
  del target,mask,rgb,loss
 log.close();data['rgb_colors']=colors.detach().cpu().numpy();data['logit_opacities']=logits.detach().cpu().numpy();np.savez(out/'params.npz',**data)
 with np.load(mapfile,allow_pickle=False) as old,np.load(out/'params.npz',allow_pickle=False) as new:
  assert old.files==new.files
  for k in old.files:
   assert np.isfinite(new[k]).all() if np.issubdtype(new[k].dtype,np.number) else True
   if k not in ('rgb_colors','logit_opacities'):assert np.array_equal(old[k],new[k]),k
 (out/'summary.json').write_text(json.dumps(dict(status='PASS',source_map_sha256=sha(mapfile),output_map_sha256=sha(out/'params.npz'),steps=args.steps,frames=N,train_frame_policy='sequential cycle all mapper frames',train_mask='Demo 0.60 confidence hard-depth mask; not ground truth static',pose_source=str(trajectory),max_camera_error=camera_error,mapper_direct_render_difference=diff,raster=str(raster.__file__),background='black, mapper setup_camera',fixed_fields_verified=True,seconds=time.time()-tic,peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))
if __name__=='__main__':main()
