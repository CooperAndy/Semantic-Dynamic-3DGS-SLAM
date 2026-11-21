"""Controlled donor inputs; native losses and initialization remain unchanged."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F

def install(core,cfg,out,backend):
 R=out.parents[1];N=cfg['data']['num_frames'];budget=cfg['factorial']['history_budget'];seed=cfg['seed'];donor=R/'runs'/cfg['donor_variant'];is_donor=out==donor
 base=np.full((N,60),-1,np.int32);hist=base.copy();flag=np.zeros_like(base,bool)
 if not is_donor:
  seal=json.loads((R/'receipts/donor.json').read_text());assert seal['status']=='PASS_SEALED_DONOR_INPUTS'
  for x in seal['files']:
   p=R/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
  with np.load(donor/'target_schedule.npz') as z:base=z['base'].copy();hist=z['history_03'].copy();flag=z['flag_03'].copy()
 masks=out/'addition_masks';masks.mkdir();birthdir=out/'birth_geometry';birthdir.mkdir();updates=(out/'mapping_updates.jsonl').open('x',buffering=1);pools=(out/'base_pools.jsonl').open('x',buffering=1);state={};context={'birth':None};obs=dict(mapping_steps=0,mapping_backwards=0,history_slots=0,changed_history_targets=0,addition_calls=0,added=0,birth_geometry_files=0,camera_Adam=0,camera_gradients=0)
 original_loss=core.get_loss;original_optimizer=core.initialize_optimizer;original_cloud=core.get_pointcloud
 def cloud(color,depth,intrinsics,w2c,*a,**kw):
  result=original_cloud(color,depth,intrinsics,w2c,*a,**kw)
  if context['birth'] is not None:
   i=context['birth'];mask=kw['mask'];ids=torch.nonzero(mask.reshape(-1),as_tuple=False).reshape(-1);pc,dist=result;assert len(ids)==len(pc)
   np.savez_compressed(birthdir/f'{i:06}.npz',ids=ids.cpu().numpy(),pointcloud=pc.detach().cpu().numpy(),mean_sq_dist=dist.detach().cpu().numpy(),w2c=w2c.detach().cpu().numpy(),intrinsics=intrinsics.detach().cpu().numpy(),sampled_depth=depth.reshape(-1)[ids].detach().cpu().numpy(),image_shape=np.array(depth.shape[-2:]));obs['birth_geometry_files']+=1
  return result
 def loss(*a,**kw):
  assert not kw.get('tracking'),'Camera Adam must not run in feature tracker'
  if not kw.get('mapping'):return original_loss(*a,**kw)
  f=sys._getframe(1);t=int(f.f_locals['time_idx']);it=int(f.f_locals['iter']);kf={int(x['id']):x for x in f.f_locals['keyframe_list']};old=sorted(q for q in kf if q<=t-200)
  if is_donor:
   pool=[int(x) for x in f.f_locals['selected_time_idx']]
   if not it:pools.write(json.dumps(dict(frame=t,targets=pool))+'\n')
   base[t,it]=pool[np.random.default_rng(np.random.SeedSequence([seed,t,it,0])).integers(len(pool))];hist[t,it]=base[t,it]
   if old and it in [15,35,55]:hist[t,it]=old[np.random.default_rng(np.random.SeedSequence([seed,t,it,1])).integers(len(old))];flag[t,it]=True
  idx=int(hist[t,it] if budget else base[t,it]);assert idx<=t and idx%20!=19
  backend.check(a[0],[t,idx]);args=list(a);data=dict(args[1]);source=f.f_locals['curr_data'] if idx==t else kf[idx];data.update(im=source['im'] if idx==t else source['color'],depth=source['depth'],depth_confidence=source['depth_confidence'],id=idx,iter_gt_w2c_list=f.f_locals['gt_w2c_all_frames'][:idx+1]);args[1]=data;args[3]=idx
  if idx!=t:assert np.array_equal(kf[idx]['est_w2c'].detach().cpu().numpy(),backend.selected[idx])
  slot=bool(budget and flag[t,it]);assert not slot or idx<=t-200;state.update(frame=t,iteration=it,target=idx,base_target=int(base[t,it]),history_slot=slot,history_changed=bool(slot and idx!=base[t,it]));value,var,details=original_loss(*args,**kw);assert torch.isfinite(value)
  def backward(g):obs['mapping_backwards']+=1;assert torch.isfinite(g).all();return g
  value.register_hook(backward);return value,var,details
 def optimizer(params,lrs,tracking):
  assert not tracking;opt=original_optimizer(params,lrs,tracking);step=opt.step;q=params['cam_unnorm_rots'].detach().clone();t=params['cam_trans'].detach().clone()
  def wrapped(*a,**kw):
   assert params['cam_unnorm_rots'].grad is None and params['cam_trans'].grad is None
   ret=step(*a,**kw);assert torch.equal(params['cam_unnorm_rots'],q) and torch.equal(params['cam_trans'],t);obs['mapping_steps']+=1;obs['history_slots']+=int(state['history_slot']);obs['changed_history_targets']+=int(state['history_changed']);updates.write(json.dumps(state)+'\n');backend.check(params,[state['frame'],state['target']])
   if state['iteration']==59:assert len(params['means3D'])<=6000000 and shutil.disk_usage(R).free>5.15*2**30
   return ret
  opt.step=wrapped;return opt
 def addition(params,variables,data,sil_thres,index,mean_sq_dist_method,gaussian_distribution,confidence_seed_threshold=None):
  backend.check(params,[index]);context['birth']=index;before=len(params['means3D'])
  try:
   if is_donor:
    from c04n_addition import add_new_gaussians
    params,variables=add_new_gaussians(params,variables,data,sil_thres,index,mean_sq_dist_method,gaussian_distribution,confidence_seed_threshold,core=core,mode='native',cap=3000,seed=seed,log_path=out/'native_addition.jsonl',fb=cfg['mapping']['residual_fb'])
   else:
    src=donor/'addition_masks'/f'{index:06}.npz';p=masks/src.name;p.hardlink_to(src)
    with np.load(p) as z:ids=z['selected_ids'].copy();tail=bool(z['native_raw_any'])
    selected=torch.zeros(data['depth'].numel(),dtype=torch.bool,device='cuda');selected[torch.tensor(ids,device='cuda',dtype=torch.long)]=True;assert bool((data['depth'].reshape(-1)[selected]>0).all());g=core.transform_to_frame(params,index,gaussians_grad=False,camera_grad=False);core.Renderer(raster_settings=data['cam'])(**core.transformed_params2depthplussilhouette(params,data['w2c'],g))
    if tail:
     q=F.normalize(params['cam_unnorm_rots'][...,index].detach());t=params['cam_trans'][...,index].detach();w2c=torch.eye(4,device='cuda');w2c[:3,:3]=core.build_rotation(q);w2c[:3,3]=t;points,dist=core.get_pointcloud(data['im'],data['depth'],data['intrinsics'],w2c,mask=selected,compute_mean_sq_dist=True,mean_sq_dist_method=mean_sq_dist_method);new=core.initialize_new_params(points,dist,gaussian_distribution)
     for k,v in new.items():params[k]=torch.nn.Parameter(torch.cat((params[k],v),0).requires_grad_(True))
     for k in ['means2D_gradient_accum','denom','max_2D_radius']:variables[k]=torch.zeros(len(params['means3D']),device='cuda').float()
     variables['timestep']=torch.cat((variables['timestep'],index*torch.ones(len(ids),device='cuda')),0)
  finally:context['birth']=None
  n=len(params['means3D'])-before;assert n<=3000;obs['addition_calls']+=1;obs['added']+=n
  with (out/'addition_history.jsonl').open('a') as f:f.write(json.dumps(dict(frame=index,before=before,after=len(params['means3D']),added=n,selected_w2c=backend.selected[index].tolist()))+'\n')
  return params,variables
 def finish():
  np.savez_compressed(out/'target_schedule.npz',base=base,history_03=hist,flag_03=flag);updates.close();pools.close();return obs
 core.get_loss=loss;core.initialize_optimizer=optimizer;core.add_new_gaussians=addition;core.get_pointcloud=cloud
 return obs,finish
