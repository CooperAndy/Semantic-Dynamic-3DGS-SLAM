"""Post-run common-camera image scoring, no GT or training."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
import lpips
from PIL import Image
R=Path('/root/autodl-tmp/chy_b_budget_20261027/window200');S=R/'source';sys.path.insert(0,str(S))
from external_tracking import render
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.time(); variants=['N0','Grad','B'];frames=list(range(19,200,20));m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());records={};runs={}; maps={}
 torch.set_num_threads(4);model=lpips.LPIPS(net='alex',version='0.1',spatial=False,verbose=False).cpu().eval()
 sp=importlib.util.spec_from_file_location('score_core',S/'scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(sp);sys.modules[sp.name]=core;sp.loader.exec_module(core)
 for name in variants:
  p=R/'runs'/name;receipt=json.loads((p/'smoke_receipt.json').read_text());assert receipt['status']=='PASS_ENGINEERING' and receipt['frames']==200
  hist=[json.loads(s) for s in (p/'addition_history.jsonl').read_text().splitlines()];assert len(hist)==189 and all(x['selected_count']==3000 for x in hist)
  with np.load(p/'params.npz',allow_pickle=False) as z:
   assert not np.isin(z['timestep'],frames).any() and not np.isin(z['keyframe_time_indices'],frames).any()
   maps[name]={k:z[k].copy() for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans','intrinsics','w2c']}
  runs[name]={'receipt':receipt,'map_sha256':sha(p/'params.npz'),'history_sha256':sha(p/'addition_history.jsonl'),'selection_mean_seconds':float(np.mean([x['selection_info']['seconds'] for x in hist])),'candidate_min':min(x['candidate_count'] for x in hist),'candidate_median':float(np.median([x['candidate_count'] for x in hist]))}
 assert len({runs[n]['receipt']['initial_gaussian_sha256'] for n in variants})==1
 for name in variants[1:]:
  for k in ['cam_unnorm_rots','cam_trans','intrinsics','w2c']:assert np.array_equal(maps['N0'][k],maps[name][k]),(name,k)
 D=R/'evaluation';D.mkdir(exist_ok=False);data={'cam':setup_camera(m['W'],m['H'],maps['N0']['intrinsics'],maps['N0']['w2c']),'w2c':torch.tensor(maps['N0']['w2c'],device='cuda')}
 for frame in frames:
  row=m['rows'][frame];rgbpath=Path(row['rgb']);dpath=Path(row['depth']);assert sha(rgbpath)==row['hashes']['rgb'] and sha(dpath)==row['hashes']['depth']
  with Image.open(rgbpath) as im: target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  depth=np.load(dpath,allow_pickle=False);valid=np.isfinite(depth)&(depth>0)
  outputs={}
  for name in variants:
   z=maps[name];params={k:torch.as_tensor(z[k],device='cuda') for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
   rgb,ds=render(params,data,frame,core);assert np.isfinite(rgb).all() and np.isfinite(ds).all()
   outputs[name]=(rgb,ds);del params
  common=valid.copy()
  for rgb,ds in outputs.values():common &= (ds[1]>.99)&(ds[0]>0)&np.isfinite(ds[0])
  rr={}
  for name,(rgb,ds) in outputs.items():
   err=rgb.astype(np.float64)-target.astype(np.float64); ce=np.clip(rgb,0,1).astype(np.float64)-target.astype(np.float64)
   mse=float(np.mean(err**2)); cmse=float(np.mean(ce**2)); cov=ds[1]>.99
   aa=torch.from_numpy(rgb).unsqueeze(0);bb=torch.from_numpy(target).unsqueeze(0)
   with torch.no_grad():ssim=float(calc_ssim(aa,bb)); percept=float(model(aa.clamp(0,1),bb,normalize=True).item())
   rowout={'psnr_raw_db':float(-10*np.log10(mse)),'psnr_clipped_db':float(-10*np.log10(cmse)),'ssim':ssim,'lpips':percept,'coverage099':float(cov.mean()),'depth_proxy_raw_valid_mae_m':float(np.mean(np.abs(ds[0][valid]-depth[valid]))),'depth_proxy_common_norm_mae_m':float(np.mean(np.abs(ds[0][common]/ds[1][common]-depth[common]))) if common.any() else None,'common_pixels':int(common.sum())}
   p=D/f'{name}_{frame:06d}.npz';np.savez_compressed(p,rgb=rgb,depth_silhouette=ds);rowout['render_sha256']=sha(p);rr[name]=rowout
  records[str(frame)]=rr;print('scored',frame,flush=True)
 summaries={n:{k:float(np.mean([records[str(f)][n][k] for f in frames])) for k in ['psnr_raw_db','psnr_clipped_db','ssim','lpips','coverage099','depth_proxy_raw_valid_mae_m','depth_proxy_common_norm_mae_m']} for n in variants}
 result={'status':'COMPLETED','scope':'fixed ORB same camera, mapping-heldout exposed DEV, per-image means, no GT or tracking evaluation','frames':frames,'runs':runs,'records':records,'means':summaries,'seconds':time.time()-start,'peak_torch_bytes':torch.cuda.max_memory_allocated(),'quality_adoption':False}
 (D/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
 print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
