"""Frozen five methods on project holdout. No GT label file is opened."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,factor_weights,native_render_variables
from native_semantic_render import render_mass,semantic_features,decode_mass
GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--sequence',required=True);a=ap.parse_args();root=Path(a.root)
 mp=root/'inputs/kitti360_isolated_v1/manifest.json';man=json.loads(mp.read_text());seq=next(s for s in man['sequences'] if s['name']==a.sequence);d=Path(seq['path'])
 out=root/'03_方法冻结与隔离评价/kitti360_isolated_v1'/a.sequence;assert not out.exists();out.mkdir(parents=True)
 for n in ['states','renders','observations']:(out/n).mkdir()
 protocol=root/'protocol/KITTI360隔离评价预声明_20261007_v1.json';assert sha(protocol)==man['protocol_sha256']
 dump(out/'config.json',{'status':'FROZEN_HOLDOUT','protocol_sha256':sha(protocol),'manifest_sha256':sha(mp),'groups':GROUPS,'fusion_frames':seq['fusion'],'evaluation_frames':seq['evaluation'],'geometry_frames':seq['geometry_frames'],'GT_in_main':False,'history':'OFF','S0_selected_on_DEV':True})
 sys.path.insert(0,'/root/autodl-tmp/SplaTAM')
 import torch
 import torch.nn.functional as F
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 from utils.recon_helpers import setup_camera
 from utils.slam_helpers import transform_to_frame
 bp=d/'base_map.npz';basehash=sha(bp)
 with np.load(bp,allow_pickle=False) as p:base={k:p[k] for k in p.files}
 ids=base['source_gaussian_index'];n=len(ids);params={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']};camera=setup_camera(seq['width'],seq['height'],base['intrinsics'],base['w2c'])
 def forward(t,c):return Renderer(raster_settings=camera)(**native_render_variables(params,t,c,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid))[0]
 def adjoint(t,m):
  cols=torch.zeros((n,3),device='cuda',requires_grad=True);return torch.autograd.grad(forward(t,cols),cols,grad_outputs=m)[0].detach()
 states={g:Evidence(ids,19) for g in GROUPS};torch.cuda.reset_peak_memory_stats();start=time.time();stats=[];locks=[]
 for pos,f in enumerate(seq['fusion']):
  rec=seq['records'][f];pi=rec['prediction_frame'];lp=d/'prediction/labels'/f'{pi:06d}_semantic.npy';cp=d/'prediction/confidence'/f'{pi:06d}_confidence.npy';dp=rec['files']['depth_sgbm']['path'];assert sha(dp)==rec['files']['depth_sgbm']['sha256']
  labels=np.load(lp);conf=np.load(cp).astype(np.float32);dep=np.load(dp);locks.append({'frame':f,'original_frame':rec['original_frame'],'labels':sha(lp),'confidence':sha(cp),'depth':sha(dp),'generation_sha256':sha(d/'prediction/generation_summary.json')})
  with torch.no_grad():
   t=transform_to_frame(params,f,False,False);z=t['means3D'][:,2];mom=forward(t,torch.stack([z,torch.ones_like(z),z*z],1));alpha=mom[1];mapdep=mom[0]/alpha.clamp_min(1e-8);dep=torch.as_tensor(dep,device='cuda');geo=1/(1+(mapdep-dep).abs()**2);basic=torch.isfinite(dep)&(dep>=.5)&(dep<=80)&(alpha>=.05)&torch.isfinite(mapdep)&(mapdep>0)
  labels=torch.as_tensor(labels.astype(np.int64),device='cuda');conf=torch.as_tensor(conf,device='cuda');valid=basic&(labels>=0)&(labels<19)&torch.isfinite(conf)&(conf>=0)&(conf<=1);grads=[]
  for offset in range(0,22,3):
   masks=torch.zeros((3,*labels.shape),device='cuda')
   for j,k in enumerate(range(offset,min(offset+3,22))):masks[j]=valid.float()*((labels==k).float() if k<19 else conf if k==19 else geo if k==20 else torch.ones_like(conf))
   grads.append(adjoint(t,masks).cpu().numpy())
  combined=np.concatenate(grads,1)[:,:22];assert np.isfinite(combined).all() and combined.min()>=-1e-5;combined=np.maximum(combined,0);den=combined[:,:19].sum(1);total=float((alpha*valid).sum(dtype=torch.float64).cpu());err=abs(float(den.sum(dtype=np.float64))-total)/max(total,1);assert err<1e-4 and np.allclose(den,combined[:,21],atol=.002,rtol=1e-4)
  active=np.flatnonzero(den>=.001);dist=combined[active,:19]/den[active,None];sem=np.clip(combined[active,19]/den[active],0,1);geometric=np.clip(combined[active,20]/den[active],0,1);version=hashlib.sha256(json.dumps([locks[-1],basehash],sort_keys=True).encode()).hexdigest()
  np.savez_compressed(out/'observations'/f'{f:06d}_prediction.npz',gaussian_id=ids[active],gaussian_index=active,distribution=dist,semantic_reliability=sem,geometry_reliability=geometric,input_version=np.array(version))
  distribution=np.zeros((n,19),np.float32);distribution[active]=dist
  for g,s in states.items():
   weights=np.zeros(n,np.float64);weights[active]=factor_weights(g,np.ones(len(active),bool),sem,geometric,.5);obs={'dataset':'KITTI360','sequence':seq['name'],'camera':'image_00','original_frame':rec['original_frame']};assert s.update_frame(obs,version,distribution,weights);assert not s.update_frame(obs,version,distribution,weights)
  stats.append({'frame':f,'valid_pixels':int(valid.sum()),'gaussians':len(active),'adjoint_relative_error':err});del combined,grads,distribution,masks
  print(json.dumps({'phase':'fusion','sequence':a.sequence,'frames':pos+1,'total':21}),flush=True)
 rgb_reference={};rgb_checks=[]
 for g,s in states.items():
  p=out/'states'/f'{g}_final.npz';s.save(p);loaded=Evidence.load(p,ids);assert all(np.array_equal(getattr(s,k),getattr(loaded,k)) for k in ['evidence','support','total_weight']);folder=out/'renders'/g;folder.mkdir();features=semantic_features(s)
  for f in seq['evaluation']:
   with torch.no_grad():
    t=transform_to_frame(params,f,False,False);shared=forward(t,torch.ones_like(params['rgb_colors']))[0].cpu().numpy();rgb=forward(t,params['rgb_colors']).cpu().numpy();decoded=decode_mass(render_mass(params,t,features,camera),reference_alpha=shared)
   if f not in rgb_reference:rgb_reference[f]=rgb
   difference=float(np.max(np.abs(rgb-rgb_reference[f])));assert np.allclose(rgb,rgb_reference[f],atol=1e-6,rtol=1e-6);rgb_checks.append({'frame':f,'group':g,'max_abs_difference':difference});np.savez_compressed(folder/f'{f:06d}.npz',**{k:v for k,v in decoded.items() if k!='class_mass'})
  print(json.dumps({'phase':'render','group':g,'views':5}),flush=True)
 assert sha(bp)==basehash
 for k,tensor in params.items():assert np.array_equal(tensor.cpu().numpy(),base[k].astype(np.float32)),k
 dump(out/'frame_stats.json',stats);dump(out/'input_locks.json',locks);dump(out/'RGB_equality.json',rgb_checks)
 dump(out/'result.json',{'status':'COMPLETE_ISOLATED_FUSION_AWAIT_GT_SCORING','groups':GROUPS,'base_unchanged':True,'RGB_equality_max':max(x['max_abs_difference'] for x in rgb_checks),'source_frame_replay_noop':True,'state_identity_roundtrip':True,'frames':21,'views':5,'wall_seconds':time.time()-start,'peak_torch_GPU_bytes':torch.cuda.max_memory_allocated(),'base_sha256':basehash,'GT_opened_by_fusion':False})
if __name__=='__main__':main()
