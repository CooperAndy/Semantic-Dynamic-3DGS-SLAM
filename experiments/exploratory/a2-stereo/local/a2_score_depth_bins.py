"""Distance-stratified pooled metrics on five-map common support only."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');ROOT=O/'mapper_v1';BASE=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200');RT=Path('/root/autodl-tmp/stage3_g15_20261028_p47149');ARMS=('B0','B1','B2','B3','B4')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 resultfile=ROOT/'evaluation_depth_bins_200.json';assert not resultfile.exists();torch.set_num_threads(2)
 audit=json.loads((ROOT/'independent_audit_v2_200.json').read_text());assert audit['status']=='PASS'
 m=json.loads((BASE/'inputs/runtime/input_manifest.json').read_text());sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(RT/'native'))
 spec=importlib.util.spec_from_file_location('a2_bin_eval_core',ROOT/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 from external_tracking import render
 from utils.recon_helpers import setup_camera
 maps={};ref=None
 for name in ARMS:
  path=ROOT/'runs'/f'{name}_200'/'params.npz';assert sha(path)==audit['arms'][name]['map_sha256']
  with np.load(path) as z:
   meta={k:z[k].copy() for k in ('cam_unnorm_rots','cam_trans','intrinsics','w2c')}
   if ref is None:ref=meta
   else:assert all(np.array_equal(meta[k],ref[k]) for k in meta)
   maps[name]={k:torch.tensor(z[k],device='cuda') for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales')}
   maps[name].update({k:torch.tensor(ref[k],device='cuda') for k in ('cam_unnorm_rots','cam_trans')})
 data={'cam':setup_camera(m['W'],m['H'],ref['intrinsics'],ref['w2c']),'w2c':torch.tensor(ref['w2c'],device='cuda')}
 bins=[(.5,10.),(10.,30.),(30.,80.)];totals={f'{lo:g}-{hi:g}':{'pixels':0,'arms':{a:{'depth_absolute_sum':0.,'rgb_sse':0.} for a in ARMS}} for lo,hi in bins};frames=[]
 for frame in range(19,200,20):
  row=m['rows'][frame]
  for key in ('rgb','depth','confidence'):assert sha(row[key])==row['hashes'][key]
  with Image.open(row['rgb']) as im:target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  depth=np.load(row['depth']);conf=np.load(row['confidence']);common=np.isfinite(depth)&(depth>.5)&(depth<80)&np.isfinite(conf)&(conf>=.8);rends={}
  for arm in ARMS:
   rgb,ds=render(maps[arm],data,frame,core);z=ds[0]/np.maximum(ds[1],1e-6);common&=(ds[1]>.99)&np.isfinite(z)&(z>.5)&(z<80);rends[arm]=(np.clip(rgb,0,1),z)
  frame_result={'frame':frame,'bins':{}}
  for lo,hi in bins:
   key=f'{lo:g}-{hi:g}';mask=common&(depth>=lo)&(depth<hi);n=int(mask.sum());cell={'pixels':n,'arms':{}};totals[key]['pixels']+=n
   for arm,(rgb,z) in rends.items():
    ae=float(np.abs(z[mask]-depth[mask]).sum(dtype=np.float64));sse=float(np.square(rgb[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)).sum());totals[key]['arms'][arm]['depth_absolute_sum']+=ae;totals[key]['arms'][arm]['rgb_sse']+=sse
    cell['arms'][arm]={'depth_proxy_mae':ae/n if n else None,'rgb_psnr':-10*np.log10(sse/(3*n)) if n and sse>0 else None}
   frame_result['bins'][key]=cell
  frames.append(frame_result)
 for key,t in totals.items():
  for arm,x in t['arms'].items():
   n=t['pixels'];x['pooled_depth_proxy_mae']=x['depth_absolute_sum']/n if n else None;x['pooled_rgb_psnr']=-10*np.log10(x['rgb_sse']/(3*n)) if n and x['rgb_sse']>0 else None
 for arm in ARMS:assert sha(ROOT/'runs'/f'{arm}_200'/'params.npz')==audit['arms'][arm]['map_sha256']
 result={'status':'COMPLETE','scope':'E6a proxy distances and five-map common support only; not independent geometry','bins':totals,'frames':frames,'quality_adoption':False}
 resultfile.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'status':'COMPLETE','bins':totals},indent=2))
if __name__=='__main__':main()
