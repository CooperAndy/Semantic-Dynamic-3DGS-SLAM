"""Manifest-driven native five-factor fusion plus DEV-only O-Sem (S0 GT)."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,factor_weights,native_render_variables
from native_semantic_render import render_mass,semantic_features,decode_mass
GROUPS=['S0','S1','S-G','S2','S2-H','O-Sem']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--sequence',required=True);ap.add_argument('--mode',choices=['smoke','full'],required=True);a=ap.parse_args()
 root=Path(a.root);manpath=root/'inputs/vkitti2_gt_dev_v2/manifest.json';man=json.loads(manpath.read_text());seq=next(s for s in man['sequences'] if s['name']==a.sequence)
 d=Path(seq['path']);out=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{a.sequence}_{a.mode}'
 if out.exists():raise FileExistsError('preserve old run')
 out.mkdir(parents=True);(out/'states').mkdir();(out/'renders').mkdir();(out/'observations').mkdir()
 basepath=d/'base_map.npz';basehash=sha(basepath)
 fusion=seq['fusion'][:3] if a.mode=='smoke' else seq['fusion'];evaluation=seq['evaluation'][:1] if a.mode=='smoke' else seq['evaluation']
 config={'status':'FROZEN_BEFORE_GT_DEV_RUN','manifest_sha256':sha(manpath),'base_sha256':basehash,'fusion_frames':fusion,'evaluation_frames':evaluation,'groups':GROUPS,'map_geometry_exposure':'all26 views; semantic excludes eval5','geo_abs_m':1.,'alpha_min':.05,'frame_alpha_mass_min':.001,'hard_threshold':.5,'unknown_fraction_max':.5,'history':'OFF','main_input':'predicted labels/confidence/SGBM; no GT validity masks','Oracle':'O-Sem S0 unit hard votes on fusion GT; DEV only','frame_budget':'unit normalized hard-label distribution per distinct source frame','depth_model':'SGBM predicted, fixed absolute tolerance; not calibrated variance','quality_winner':None}
 dump(out/'config.json',config)
 for name in ['run_vkitti_gt_fusion.py','semantic_cpu.py','native_semantic_render.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
 sys.path.insert(0,'/root/autodl-tmp/SplaTAM')
 import torch
 import torch.nn.functional as F
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 from utils.recon_helpers import setup_camera
 from utils.slam_helpers import transform_to_frame
 depmanifest=json.loads((root/'01_语义表示_初始化与渲染审计/native_audit_20261007_v1/source_manifest.json').read_text())[:4]
 assert all(sha(r['source'])==r['sha256'] for r in depmanifest);dump(out/'native_dependencies.json',depmanifest)
 with np.load(basepath,allow_pickle=False) as p:base={k:p[k] for k in p.files}
 ids=base['source_gaussian_index'];n=len(ids)
 params={k:torch.as_tensor(base[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
 camera=setup_camera(int(base['org_width']),int(base['org_height']),base['intrinsics'],base['w2c'])
 def forward(t,c):return Renderer(raster_settings=camera)(**native_render_variables(params,t,c,torch.zeros_like,lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid))[0]
 def adjoint(t,m):
  cols=torch.zeros((n,3),device='cuda',requires_grad=True);return torch.autograd.grad(forward(t,cols),cols,grad_outputs=m)[0].detach()
 states={g:Evidence(ids,19) for g in GROUPS};torch.cuda.reset_peak_memory_stats();start=time.time();stats=[];inputlocks=[]
 for pos,f in enumerate(fusion):
  record=seq['records'][f]
  for kind in ['depth_sgbm','gt_labels']:assert sha(record['files'][kind]['path'])==record['files'][kind]['sha256']
  lp=d/'prediction/labels'/f'{f:06d}_semantic.npy';cp=d/'prediction/confidence'/f'{f:06d}_confidence.npy'
  rawlabels=np.load(lp);rawconf=np.load(cp).astype(np.float32);gt=np.load(record['files']['gt_labels']['path']);dep=np.load(record['files']['depth_sgbm']['path'])
  inputlocks.append({'frame':f,'prediction_labels':sha(lp),'confidence':sha(cp),'depth':record['files']['depth_sgbm']['sha256'],'oracle_GT':record['files']['gt_labels']['sha256']})
  inputlocks[-1]['prediction_generation_sha256']=sha(d/'prediction/generation_summary.json')
  with torch.no_grad():
   t=transform_to_frame(params,f,False,False);z=t['means3D'][:,2];moments=forward(t,torch.stack([z,torch.ones_like(z),z*z],dim=1));alpha=moments[1];mapdep=moments[0]/alpha.clamp_min(1e-8)
   dep=torch.as_tensor(dep,device='cuda');geometric=1/(1+(mapdep-dep).abs()**2);basic=torch.isfinite(dep)&(dep>=.5)&(dep<=80)&(alpha>=.05)&torch.isfinite(mapdep)&(mapdep>0)
  for oracle in [False,True]:
   labels=torch.as_tensor((gt if oracle else rawlabels).astype(np.int64),device='cuda');conf=torch.as_tensor(np.ones_like(rawconf) if oracle else rawconf,device='cuda')
   valid=basic&(labels>=0)&(labels<19)&torch.isfinite(conf)&(conf>=0)&(conf<=1);grads=[]
   for offset in range(0,22,3):
    masks=torch.zeros((3,*labels.shape),device='cuda')
    for j,k in enumerate(range(offset,min(offset+3,22))):masks[j]=valid.float()*((labels==k).float() if k<19 else conf if k==19 else geometric if k==20 else torch.ones_like(conf))
    grads.append(adjoint(t,masks).cpu().numpy())
   combined=np.concatenate(grads,axis=1)[:,:22];assert np.isfinite(combined).all() and combined.min()>=-1e-5;combined=np.maximum(combined,0)
   den=combined[:,:19].sum(1);total=float((alpha*valid).sum(dtype=torch.float64).cpu());err=abs(float(den.sum(dtype=np.float64))-total)/max(total,1);assert err<1e-4
   assert np.allclose(den,combined[:,21],atol=.002,rtol=1e-4)
   active=np.flatnonzero(den>=.001);dist=combined[active,:19]/den[active,None];sem=np.clip(combined[active,19]/den[active],0,1);geo=np.clip(combined[active,20]/den[active],0,1)
   version=hashlib.sha256(json.dumps([inputlocks[-1],basehash,oracle],sort_keys=True).encode()).hexdigest()
   np.savez_compressed(out/'observations'/f'{f:06d}_{"oracle" if oracle else "prediction"}.npz',gaussian_id=ids[active],gaussian_index=active,distribution=dist,semantic_reliability=sem,geometry_reliability=geo,input_version=np.array(version))
   distribution=np.zeros((n,19),np.float32);distribution[active]=dist
   for g in (['O-Sem'] if oracle else GROUPS[:-1]):
    weights=np.zeros(n,np.float64);weights[active]=factor_weights('S0' if oracle else g,np.ones(len(active),bool),sem,geo,.5)
    obs={'dataset':'VKITTI2','sequence':seq['scene']+'/clone','camera':'Camera_0','original_frame':record['original_frame']}
    assert states[g].update_frame(obs,version,distribution,weights);assert not states[g].update_frame(obs,version,distribution,weights)
   stats.append({'frame':f,'oracle':oracle,'valid_pixels':int(valid.sum()),'gaussians':len(active),'adjoint_relative_error':err})
   del distribution,combined,grads,masks
  print(json.dumps({'phase':'fusion','sequence':a.sequence,'frames':pos+1,'total':len(fusion)}),flush=True)
 for g,s in states.items():
  statepath=out/'states'/f'{g}_final.npz';s.save(statepath);loaded=Evidence.load(statepath,ids)
  assert all(np.array_equal(getattr(s,k),getattr(loaded,k)) for k in ['evidence','support','total_weight']) and s.processed==loaded.processed
  folder=out/'renders'/g;folder.mkdir();features=semantic_features(s)
  for f in evaluation:
   with torch.no_grad():
    t=transform_to_frame(params,f,False,False);shared=forward(t,torch.ones_like(params['rgb_colors']))[0].cpu().numpy();decoded=decode_mass(render_mass(params,t,features,camera),reference_alpha=shared)
   np.savez_compressed(folder/f'{f:06d}.npz',**{k:v for k,v in decoded.items() if k!='class_mass'})
  print(json.dumps({'phase':'render','group':g,'views':len(evaluation)}),flush=True)
 assert sha(basepath)==basehash
 for k,tensor in params.items():assert np.array_equal(tensor.cpu().numpy(),base[k].astype(np.float32)),k
 dump(out/'frame_stats.json',stats);dump(out/'input_locks.json',inputlocks)
 dump(out/'result.json',{'status':'COMPLETE_GT_DEV_FUSION_AWAIT_CPU_EVALUATION','groups':GROUPS,'base_unchanged':True,'geometry_and_RGB_and_camera_unchanged':True,'source_frame_replay_noop':True,'state_identity_roundtrip':True,'frames':len(fusion),'views':len(evaluation),'wall_seconds':time.time()-start,'peak_torch_GPU_bytes':torch.cuda.max_memory_allocated(),'base_sha256':basehash,'GT_role':'O-Sem fusion on DEV only; evaluation labels not used in fusion','final_test':False})
if __name__=='__main__':main()
