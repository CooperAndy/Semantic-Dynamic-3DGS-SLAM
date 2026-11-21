"""Common-camera, mapper-holdout-only A1 scoring after sealed map runs."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path('/root/autodl-tmp/A2_stereo_robust_geometry/factorial_v1');BASE=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/C200');R=Path('/root/autodl-tmp/stage3_g15_20261028_p47149')
COUNT=int(sys.argv[1]);assert COUNT in (25,200)
arms=('C0','C1','C2','C3','C4','C5');frames=[i for i in range(19,COUNT,20)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();torch.set_num_threads(2);out=ROOT/f'evaluation_{COUNT}.json';assert not out.exists()
 m=json.loads((BASE/'inputs/runtime/input_manifest.json').read_text());rows={r['frame_id']:r for r in m['rows']};maps={};checks={}
 sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(R/'native'))
 spec=importlib.util.spec_from_file_location('a2_eval_core',ROOT/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 from external_tracking import render
 from utils.recon_helpers import setup_camera
 from utils.slam_external import calc_ssim
 reference=None
 for name in arms:
  path=ROOT/'runs'/f'{name}_{COUNT}'; receipt=json.loads((path/'a2_receipt.json').read_text());p=path/'params.npz'
  if receipt['status']!='PASS_ENGINEERING' or sha(p)!=receipt['map_sha256']:raise RuntimeError('unsealed/incomplete map')
  with np.load(p) as z:
   assert not np.isin(z['timestep'],frames).any() and not np.isin(z['keyframe_time_indices'],frames).any()
   meta={k:z[k].copy() for k in ('cam_unnorm_rots','cam_trans','intrinsics','w2c')}
   if reference is None:reference=meta
   else:
    for k in meta:
     if not np.array_equal(meta[k],reference[k]):raise RuntimeError('camera mismatch '+k)
   maps[name]={k:torch.tensor(z[k],device='cuda') for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales')}
   maps[name].update({k:torch.tensor(reference[k],device='cuda') for k in ('cam_unnorm_rots','cam_trans')})
  checks[name]={'sha256':receipt['map_sha256'],'births_total':receipt['births_total'],'survivors':receipt['survivors'],'seconds':receipt['seconds'],'peak_cuda_bytes':receipt['peak_cuda_bytes'],'initial_hash':receipt['initial_hash']}
 if len({checks[k]['initial_hash'] for k in arms})!=1:raise RuntimeError('initialization mismatch')
 data={'cam':setup_camera(m['W'],m['H'],reference['intrinsics'],reference['w2c']),'w2c':torch.tensor(reference['w2c'],device='cuda')};result=[]
 for f in frames:
  row=rows[f]
  for k in ('rgb','depth','confidence'):
   if sha(row[k])!=row['hashes'][k]:raise RuntimeError('heldout input hash mismatch')
  with Image.open(row['rgb']) as im:target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  d=np.load(row['depth']);cf=np.load(row['confidence']);renders={};common=np.isfinite(d)&(d>.5)&(d<80)&np.isfinite(cf)&(cf>=.8)
  for name in arms:
   rgb,ds=render(maps[name],data,f,core)
   if not np.isfinite(rgb).all() or not np.isfinite(ds).all():raise RuntimeError('nonfinite')
   depth=ds[0]/np.maximum(ds[1],1e-6);common&=(ds[1]>.99)&np.isfinite(depth)&(depth>.5)&(depth<80);renders[name]=(rgb,ds,depth)
  frame={'frame':f,'common_depth_pixels':int(common.sum()),'arms':{}}
  for name,(rgb,ds,depth) in renders.items():
   pred=np.clip(rgb,0,1); mse=np.mean((pred.astype(np.float64)-target.astype(np.float64))**2)
   with torch.no_grad():ssim=float(calc_ssim(torch.tensor(pred).unsqueeze(0),torch.tensor(target).unsqueeze(0)))
   frame['arms'][name]={'clipped_psnr_db':float(-10*np.log10(mse)),'ssim':ssim,'alpha099_coverage':float((ds[1]>.99).mean()),'depth_proxy_common_mae_m':float(np.mean(abs(depth[common]-d[common]))) if common.any() else None}
  result.append(frame)
 for name in arms:
  if sha(ROOT/'runs'/f'{name}_{COUNT}'/'params.npz')!=checks[name]['sha256']:raise RuntimeError('map changed during score')
 means={}
 for name in arms:
  means[name]={metric:float(np.mean([r['arms'][name][metric] for r in result])) for metric in ('clipped_psnr_db','ssim','alpha099_coverage')}
  vals=[r['arms'][name]['depth_proxy_common_mae_m'] for r in result if r['arms'][name]['depth_proxy_common_mae_m'] is not None]
  means[name]['depth_proxy_common_mae_m']=float(np.mean(vals)) if vals else None
 report={'status':'COMPLETE','scope':'exposed KITTI00 C200 DEV, mapping-only holdout with E6a proxy; ORB/E6a upstream have temporal dependencies','frames':result,'means':means,'checks':checks,'elapsed_s':time.monotonic()-start,'quality_adoption':False}
 out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
