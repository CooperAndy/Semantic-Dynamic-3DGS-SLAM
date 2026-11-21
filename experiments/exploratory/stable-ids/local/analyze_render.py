import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

R=Path('/root/autodl-tmp/chy_c_stableid_20261027');OUT=R/'controlled';REF=Path('/root/autodl-tmp/chy_b_budget_20261027/window200')
KEYS=['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def st(x):
 x=np.asarray(x).ravel().astype(np.float64);assert x.size and np.isfinite(x).all()
 return {'n':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(x.max())}
def main():
 started=time.time();p=R/'runs/N0_ID';rec=json.loads((p/'smoke_receipt.json').read_text());assert rec['status']=='PASS_ENGINEERING'
 hist=[json.loads(x) for x in (p/'addition_history.jsonl').read_text().splitlines()];pr=[json.loads(x) for x in (p/'pruning.jsonl').read_text().splitlines()]
 assert len(hist)==189 and all(x['selected_count']==3000 for x in hist)
 early=dict(np.load(p/'params99.npz',allow_pickle=False));final=dict(np.load(p/'params.npz',allow_pickle=False))
 # Independently reconstruct ID order and birth labels from append and deletion events.
 ninit=pr[0]['before'];ids=np.arange(ninit,dtype=np.int64);birth=np.zeros(ninit,dtype=np.int64);nnext=ninit;deleted=[];snap_checks=[]
 additions={x['frame']:x['selected_count'] for x in hist};events={}
 for e in pr:events.setdefault(e['frame'],[]).append(e)
 for frame in range(200):
  if frame in additions:
   n=additions[frame];ids=np.concatenate((ids,np.arange(nnext,nnext+n)));birth=np.concatenate((birth,np.full(n,frame)));nnext+=n
  for e in events.get(frame,[]):
   assert len(ids)==e['before']; drop=np.empty(0,dtype=np.int64)
   if e['removed']:
    f=p/e['file'];assert sha(f)==e['sha256'];drop=np.load(f)['deleted_ids'];assert len(drop)==e['removed'] and np.isin(drop,ids).all();deleted.append(drop)
   keep=~np.isin(ids,drop);ids=ids[keep];birth=birth[keep]
  if frame in (0,99,198):
   s=dict(np.load(p/f'params{frame}.npz',allow_pickle=False));assert np.array_equal(ids,s['stable_id']) and np.array_equal(birth,s['timestep']);snap_checks.append(frame)
 assert np.array_equal(ids,final['stable_id']) and np.array_equal(birth,final['timestep'])
 deleted=np.concatenate(deleted);assert len(np.unique(deleted))==len(deleted) and len(deleted)+len(ids)==nnext
 assert np.unique(ids).size==len(ids) and np.all(ids[1:]>ids[:-1])
 ei=early['stable_id'];fi=final['stable_id'];match=np.isin(fi,ei);fp=np.flatnonzero(match);ep=np.searchsorted(ei,fi[fp]);assert np.array_equal(ei[ep],fi[fp]) and np.array_equal(early['timestep'][ep],final['timestep'][fp])
 delei=~np.isin(ei,fi);sig=lambda x:1/(1+np.exp(-np.clip(x.astype(np.float64),-700,700)))
 deltas={'position_l2_m':st(np.linalg.norm(final['means3D'][fp]-early['means3D'][ep],axis=1)),
         'rgb_absolute_channels':st(abs(final['rgb_colors'][fp]-early['rgb_colors'][ep])),
         'scale_relative_abs':st(abs(np.exp(final['log_scales'][fp]-early['log_scales'][ep])-1)),
         'opacity_abs':st(abs(sig(final['logit_opacities'][fp])-sig(early['logit_opacities'][ep])))}
 ref=dict(np.load(REF/'runs/N0/params.npz',allow_pickle=False));comparison={}
 for k in KEYS+['cam_unnorm_rots','cam_trans']:
  same_shape=final[k].shape==ref[k].shape;comparison[k]={'same_shape':same_shape,'exact':bool(same_shape and np.array_equal(final[k],ref[k])),'max_abs':float(np.max(abs(final[k]-ref[k]))) if same_shape else None}
 audit={'status':'PASS_ID_CONSERVATION','frame_count':200,'initial_points':ninit,'total_created':nnext,'total_deleted':len(deleted),'final_points':len(ids),'snapshots_reconstructed':snap_checks,
        'early99_points':len(ei),'matched_final_survivors':len(fp),'early99_deleted':int(delei.sum()),'matched_deltas':deltas,'reference_N0_comparison':comparison,
        'source_sha256':{str(f):sha(f) for f in [p/'params99.npz',p/'params.npz',p/'pruning.jsonl',p/'addition_history.jsonl']},
        'note':'Shared GPU with G13; runtime not comparable. Within-run identity verified; reference exactness reported separately.'}
 (R/'identity_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False));print('IDENTITY_PASS',json.dumps(deltas),flush=True)
 # A strict implementation-only replay check before using this trace for causal rendering.
 if not all(x['exact'] for x in comparison.values()):
  raise RuntimeError('Recorded control differs from prior N0: preserve identity audit and investigate before rendering')
 OUT.mkdir(exist_ok=False);sys.path.insert(0,str(R/'source'))
 from external_tracking import render
 from utils.recon_helpers import setup_camera
 sp=importlib.util.spec_from_file_location('c_render_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(sp);sys.modules[sp.name]=core;sp.loader.exec_module(core)
 manifest=json.loads((R/'inputs/runtime/input_manifest.json').read_text());data={'cam':setup_camera(manifest['W'],manifest['H'],final['intrinsics'],final['w2c']),'w2c':torch.tensor(final['w2c'],device='cuda')}
 conditions={'final':[],'reset_position':['means3D'],'reset_colour':['rgb_colors'],'reset_scale':['log_scales'],'reset_opacity':['logit_opacities'],'reset_all_early':KEYS}
 frames=list(range(19,200,20));byframe={str(i):{} for i in frames};replays=[];records=[]
 for cond,keys in conditions.items():
  arr={k:final[k].copy() for k in KEYS}
  for k in keys:arr[k][fp]=early[k][ep]
  params={k:torch.tensor(arr[k],device='cuda') for k in KEYS};params.update({k:torch.tensor(final[k],device='cuda') for k in ['cam_unnorm_rots','cam_trans']})
  for i in frames:
   rgb,ds=render(params,data,i,core);assert np.isfinite(rgb).all() and np.isfinite(ds).all()
   f=OUT/f'{cond}_{i:06d}.npz';np.savez_compressed(f,rgb=rgb,depth_silhouette=ds);records.append({'file':f.name,'sha256':sha(f)})
   if cond=='final':
    with np.load(REF/'evaluation'/f'N0_{i:06d}.npz',allow_pickle=False) as z:
     dr=float(np.max(abs(rgb-z['rgb'])));dd=float(np.max(abs(ds-z['depth_silhouette'])));bound=float(1e-4+1e-5*np.max(abs(z['depth_silhouette'])));assert dr<=1e-5 and dd<=bound;replays.append({'frame':i,'rgb_max_abs':dr,'ds_max_abs':dd,'ds_bound':bound})
   print('render',cond,i,flush=True)
  del params;torch.cuda.empty_cache()
 for i in frames:
  row=manifest['rows'][i];assert sha(Path(row['rgb']))==row['hashes']['rgb'] and sha(Path(row['depth']))==row['hashes']['depth']
  with Image.open(row['rgb']) as im:target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  dep=np.load(row['depth']);valid=np.isfinite(dep)&(dep>0);outputs={};common=valid.copy()
  for c in conditions:
   z=dict(np.load(OUT/f'{c}_{i:06d}.npz'));rgb=z['rgb'];ds=z['depth_silhouette'];depth=ds[0]/np.maximum(ds[1],1e-6);outputs[c]=(rgb,ds,depth);common &= (ds[1]>.99)&np.isfinite(depth)&(depth>.5)&(depth<60)
  base=outputs['final']
  for c,(rgb,ds,z) in outputs.items():
   e=rgb.astype(np.float64)-target;ce=np.clip(rgb,0,1).astype(np.float64)-target
   byframe[str(i)][c]={'psnr_raw_db':float(-10*np.log10(np.mean(e*e))),'psnr_clipped_db':float(-10*np.log10(np.mean(ce*ce))),'coverage099':float(np.mean(ds[1]>.99)),
       'common_pixels':int(common.sum()),'depth_proxy_common_mae_m':float(np.mean(abs(z[common]-dep[common]))) if common.any() else None,
       'depth_change_vs_final_m':st(abs(z[common]-base[2][common])) if common.any() else None}
 means={c:{k:float(np.mean([byframe[str(i)][c][k] for i in frames])) for k in ['psnr_raw_db','psnr_clipped_db','coverage099','depth_proxy_common_mae_m']} for c in conditions}
 result={'status':'COMPLETED','scope':'survivor-only one-parameter rollback rendering, exposed DEV; not online anchors or GT localization','audit_sha256':sha(R/'identity_audit.json'),'means':means,'frames':byframe,'replays':replays,'files':records,'seconds':time.time()-started,'quality_adoption':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(means,indent=2),flush=True)
if __name__=='__main__':main()
