"""Posthoc paired B0/B3 regional and proxy loss-shape diagnostic."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200');RT=Path('/root/autodl-tmp/stage3_g15_20261028_p47149');DEST=O/'diagnostic_20261028';ARMS=('B0','B1','B2','B3','B4')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();DEST.mkdir(exist_ok=True);out=DEST/'region_diagnostic.json';assert not out.exists();torch.set_num_threads(2)
 seal=json.loads((R/'A2_DEV_evidence_seal_20261028.json').read_text());locks={x['path']:x['sha256'] for x in seal['files']}
 calibration=O/'gradient_calibration_20261028.json'; assert sha(calibration)==locks[str(calibration)];cal=json.loads(calibration.read_text());fb=cal['stereo_08']['fb_px_m'];gamma=cal['gamma']['B3']
 manifest=json.loads((B/'inputs/runtime/input_manifest.json').read_text());rows={r['frame_id']:r for r in manifest['rows']};sys.path.insert(0,str(R/'source'));sys.path.insert(0,str(RT/'native'))
 spec=importlib.util.spec_from_file_location('a2_region_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 from external_tracking import render
 from utils.recon_helpers import setup_camera
 maps={};ref=None
 for name in ARMS:
  path=R/'runs'/f'{name}_200'/'params.npz';assert sha(path)==locks[str(path)]
  with np.load(path) as z:
   meta={k:z[k].copy() for k in ('cam_unnorm_rots','cam_trans','intrinsics','w2c')}
   if ref is None:ref=meta
   else:assert all(np.array_equal(meta[k],ref[k]) for k in meta)
   maps[name]={k:torch.tensor(z[k],device='cuda') for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales')}
   maps[name].update({k:torch.tensor(ref[k],device='cuda') for k in ('cam_unnorm_rots','cam_trans')})
 H,W=manifest['H'],manifest['W'];data={'cam':setup_camera(W,H,ref['intrinsics'],ref['w2c']),'w2c':torch.tensor(ref['w2c'],device='cuda')}
 original=json.loads((R/'evaluation_200.json').read_text());assert sha(R/'evaluation_200.json')==locks[str(R/'evaluation_200.json')]
 frame_results=[];group_totals={}
 for frame in range(19,200,20):
  if time.monotonic()-start>180:raise RuntimeError('time budget')
  row=rows[frame]
  for key in ('rgb','depth','confidence'):assert sha(row[key])==row['hashes'][key]
  with Image.open(row['rgb']) as im:target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  depth=np.load(row['depth']);conf=np.load(row['confidence']);common=np.isfinite(depth)&(depth>.5)&(depth<80)&np.isfinite(conf)&(conf>=.8);rends={}
  for name in ARMS:
   rgb,ds=render(maps[name],data,frame,core);z=ds[0]/np.maximum(ds[1],1e-6);common&=(ds[1]>.99)&np.isfinite(z)&(z>.5)&(z<80);rends[name]={'rgb':np.clip(rgb,0,1),'depth':z,'raw':ds[0],'alpha':ds[1]}
  ncommon=int(common.sum());old=next(x for x in original['frames'] if x['frame']==frame);assert old['common_depth_pixels']==ncommon
  err0=np.mean((rends['B0']['rgb']-target)**2,axis=0);err3=np.mean((rends['B3']['rgb']-target)**2,axis=0)
  if abs(float(-10*np.log10(err0.mean()))-old['arms']['B0']['clipped_psnr_db'])>1e-4 or abs(float(-10*np.log10(err3.mean()))-old['arms']['B3']['clipped_psnr_db'])>1e-4:raise RuntimeError('render does not reproduce sealed evaluation')
  z0=rends['B0']['depth'];z3=rends['B3']['depth'];delta_ae=np.abs(z3-depth)-np.abs(z0-depth)
  base_s=np.clip(depth*depth/fb,.1,3.);raw_s=base_s*(1+2*(1-conf));s=np.clip(raw_s,.1,3.)
  resid=rends['B0']['raw']-depth;grad=gamma*.3/s*np.minimum(np.abs(resid)/s,1.)
  frame_result={'frame':frame,'common_pixels':ncommon,'rgb_mse_B0':float(err0.mean()),'rgb_mse_B3':float(err3.mean()),'rgb_mse_change':float((err3-err0).mean()),'tiles':[], 'groups':{}}
  for yy in range(4):
   for xx in range(4):
    sl=(slice(yy*H//4,(yy+1)*H//4),slice(xx*W//4,(xx+1)*W//4));cm=common[sl];improve=err0[sl]-err3[sl]
    frame_result['tiles'].append({'row':yy,'col':xx,'image_pixels':int(improve.size),'rgb_sse_improvement':float(improve.sum(dtype=np.float64)*3),'common_pixels':int(cm.sum()),'proxy_abs_error_change_sum_m':float(delta_ae[sl][cm].sum(dtype=np.float64))})
  for lo,hi in ((.5,10.),(10.,30.),(30.,80.)):
   mask=common&(depth>=lo)&(depth<hi);key=f'{lo:g}-{hi:g}';n=int(mask.sum());ss=s[mask];gg=grad[mask]
   cell={'pixels':n,'B0_abs_sum':float(np.abs(z0[mask]-depth[mask]).sum(dtype=np.float64)),
         'B3_abs_sum':float(np.abs(z3[mask]-depth[mask]).sum(dtype=np.float64)),
         'B0_signed_sum':float((z0[mask]-depth[mask]).sum(dtype=np.float64)),
         'B3_signed_sum':float((z3[mask]-depth[mask]).sum(dtype=np.float64)),
         'b3_pixel_grad_sum':float(gg.sum(dtype=np.float64)),'b3_scale_sum':float(ss.sum(dtype=np.float64)),
         'b3_scale_clip_min_count':int((ss<=.1000001).sum()),'b3_scale_clip_max_count':int((ss>=2.999999).sum()),
         'huber_quadratic_count':int((np.abs(resid[mask])<=ss).sum())}
   frame_result['groups'][key]=cell;tot=group_totals.setdefault(key,{k:0 for k in cell})
   for k,v in cell.items():tot[k]+=v
  if frame in (39,99,159):
   # Rendering comparisons are descriptive: target, B0, B3 and RGB squared-error change.
   imdiff=np.clip((err0-err3)*10,-1,1);heat=np.zeros((H,W,3),np.uint8);heat[:,:,1]=np.uint8(np.maximum(imdiff,0)*255);heat[:,:,0]=np.uint8(np.maximum(-imdiff,0)*255)
   panels=[np.uint8(np.clip(target.transpose(1,2,0),0,1)*255),np.uint8(rends['B0']['rgb'].transpose(1,2,0)*255),np.uint8(rends['B3']['rgb'].transpose(1,2,0)*255),heat]
   canvas=Image.new('RGB',(W*2,(H+30)*2),'black');draw=ImageDraw.Draw(canvas)
   for j,(arr,label) in enumerate(zip(panels,['Observed RGB','B0 rendering','B3 rendering','RGB error change: green improves, red worsens'])):
    x=(j%2)*W;y=(j//2)*(H+30);canvas.paste(Image.fromarray(arr),(x,y+30));draw.text((x+8,y+8),label,fill='white')
   canvas.save(DEST/f'frame_{frame:03d}_comparison.png')
  frame_results.append(frame_result)
 for key,c in group_totals.items():
  n=c['pixels']
  if n:
   for k in ('B0_abs_sum','B3_abs_sum','B0_signed_sum','B3_signed_sum','b3_pixel_grad_sum','b3_scale_sum'):c[k+'_mean']=c[k]/n
   for k in ('b3_scale_clip_min_count','b3_scale_clip_max_count','huber_quadratic_count'):c[k+'_fraction']=c[k]/n
 for name in ARMS:
  path=R/'runs'/f'{name}_200'/'params.npz';assert sha(path)==locks[str(path)]
 result={'status':'COMPLETE_POSTHOC','frames':frame_results,'distance_groups_pooled':group_totals,'calibration_sha256':sha(calibration),'score_sha256':sha(R/'evaluation_200.json'),'elapsed_s':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'scope':'five-map common E6a proxy, mixed pixel loss coefficients; no independent GT or causal proof','quality_adoption':False}
 out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'status':result['status'],'distance_groups_pooled':group_totals,'elapsed_s':result['elapsed_s']},indent=2))
if __name__=='__main__':main()
