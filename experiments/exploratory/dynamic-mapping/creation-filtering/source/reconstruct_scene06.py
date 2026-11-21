"""Reconstruct pre-filter candidates from prior post-frame snapshot; validate original scalar audits."""
from pathlib import Path
import csv,json,time,hashlib,numpy as np,torch,cv2
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from datasets.gradslam_datasets import load_dataset_config,ReplicaDataset
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame,transformed_params2depthplussilhouette
r=Path('/root/autodl-tmp/stage2_round14_scene06_20261031_v2');out=Path('/root/autodl-tmp/stage2_round16_creation_audit_v1');out.mkdir(exist_ok=False);w=r/'bridge';seq='vkitti2_Scene06_clone_130_137';ds=ReplicaDataset(load_dataset_config(str(w/'camera_config.yaml')),str(w/'replica_layout'),seq,start=0,end=-1,stride=1,desired_height=192,desired_width=640,device='cuda:0',relative_pose=True,ignore_bad=False,use_train_split=True);color,depth,K,pose=ds[0];first_w2c=torch.linalg.inv(pose);cam=setup_camera(640,192,K[:3,:3].cpu().numpy(),first_w2c.cpu().numpy());old=[json.loads(x) for x in (r/'D3/creation_audit.jsonl').read_text().splitlines()];t=time.perf_counter();torch.cuda.reset_peak_memory_stats();result=[]
with torch.no_grad():
 for i in range(8):
  color,dep,K,pose=ds[i];depth=dep.permute(2,0,1)[0];valid=depth>0
  if i==0:candidate=torch.ones_like(valid,dtype=torch.bool)
  else:
   previous=np.load(r/'maps/D3'/f'params{i-1}.npz');current=np.load(r/'maps/D3'/f'params{i}.npz');keys=['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales'];p={k:torch.as_tensor(previous[k],device='cuda',dtype=torch.float32) for k in keys};p.update({k:torch.as_tensor(current[k],device='cuda',dtype=torch.float32) for k in ['cam_unnorm_rots','cam_trans']});trans=transform_to_frame(p,i,gaussians_grad=False,camera_grad=False);dv=transformed_params2depthplussilhouette(p,first_w2c,trans);raster,_,_=Renderer(raster_settings=cam)(**dv);error=(depth-raster[0]).abs()*(depth>0);candidate=(raster[1]<.5)|((raster[0]>depth)&(error>50*error.median()))
  state=np.load(w/'sidecars'/f'{i:06d}_state.npy');cand=candidate.cpu().numpy();v=valid.cpu().numpy();count=int(cand.sum());effective=cand&v;expected=old[i];assert count==expected['candidate_before_state'],(i,count,expected['candidate_before_state']);assert int((effective&(state==0)).sum())==expected['actual_created'];np.savez_compressed(out/f'frame{i:02}.npz',candidate=cand,valid_depth=v,depth_m=depth.cpu().numpy(),state=state);result.append({'frame':130+i,'candidate':count,'valid_candidates':int(effective.sum()),'accepted':int((effective&(state==0)).sum()),'rejected':int((effective&(state!=0)).sum()),'original_count_matches':True,'state_sha256':hashlib.sha256((w/'sidecars'/f'{i:06d}_state.npy').read_bytes()).hexdigest()})
(out/'reconstruction.json').write_text(json.dumps(result,indent=2));(out/'metrics.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'scope':'posthoc candidate reconstruction; prior-map params and current pose; counts checked, pixel identity not historically logged'},indent=2));print(json.dumps(result,indent=2))
