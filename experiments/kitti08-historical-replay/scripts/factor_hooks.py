"""H0 supplies actual native births and counter targets; H3 reuses the sealed pair inputs."""
import sys,json,hashlib,shutil,numpy as np,torch
import torch.nn.functional as F
from pathlib import Path

def install(core,cfg,out):
 R=out.parents[1];b=cfg['factorial']['history_budget'];seed=cfg['seed'];H=b>0
 base=np.full((1000,60),-1,dtype=np.int32);hist=base.copy();flag=np.zeros_like(base,dtype=bool)
 masks=out/'addition_masks';masks.mkdir()
 donor=R/'runs'/f'B00C0_S{seed}'
 if H:
  with np.load(donor/'target_schedule.npz') as z:base=z['base'].copy();hist=z['history_03'].copy();flag=z['flag_03'].copy()
 state={};obs=dict(mapping_loss_calls={},mapping_target_frames=[],additions=[],camera_backward_loss_calls=0,completed_steps=0,history_slots=0,changed_history_targets=0,box_projection_steps=0)
 original_loss=core.get_loss;original_opt=core.initialize_optimizer
 updates=(out/'mapping_updates.jsonl').open('x',buffering=1)
 pools=(out/'base_pools.jsonl').open('x',buffering=1)
 def loss(*a,**kw):
  if not kw.get('mapping'):
   if kw.get('tracking'):obs['camera_backward_loss_calls']+=1
   return original_loss(*a,**kw)
  f=sys._getframe(1);t=int(f.f_locals['time_idx']);it=int(f.f_locals['iter']);kf={int(x['id']):x for x in f.f_locals['keyframe_list']}
  old=sorted(q for q in kf if q<=t-200)
  if not H:
   pool=[int(x) for x in f.f_locals['selected_time_idx']]
   if it==0:pools.write(json.dumps(dict(frame=t,targets=pool))+'\n')
   base[t,it]=pool[np.random.default_rng(np.random.SeedSequence([seed,t,it,0])).integers(len(pool))]
   hist[t,it]=base[t,it]
   if old and it in [15,35,55]:
    hist[t,it]=old[np.random.default_rng(np.random.SeedSequence([seed,t,it,1])).integers(len(old))];flag[t,it]=True
  idx=int(hist[t,it] if H else base[t,it]);args=list(a);d=dict(args[1])
  if idx==t:
   src=f.f_locals['curr_data'];d.update(im=src['im'],depth=src['depth'],depth_confidence=src['depth_confidence'])
  else:
   src=kf[idx];d.update(im=src['color'],depth=src['depth'],depth_confidence=src['depth_confidence'])
  d.update(id=idx,iter_gt_w2c_list=f.f_locals['gt_w2c_all_frames'][:idx+1]);args[1]=d;args[3]=idx
  assert idx<=t and idx%20!=19;slot=bool(H and flag[t,it]);assert not slot or idx<=t-200
  state.update(online=t,iteration=it,base_target=int(base[t,it]),target=idx,history_slot=slot,history_changed=bool(slot and idx!=base[t,it]))
  key=str(t);obs['mapping_loss_calls'][key]=obs['mapping_loss_calls'].get(key,0)+1
  if idx not in obs['mapping_target_frames']:obs['mapping_target_frames'].append(idx)
  return original_loss(*args,**kw)
 def optimizer(params,lrs,tracking):
  opt=original_opt(params,lrs,tracking)
  if tracking:return opt
  step=opt.step
  def wrapped(*a,**k):
   ret=step(*a,**k);row=dict(state);row['color_box']=False
   obs['completed_steps']+=1;obs['history_slots']+=int(row['history_slot']);obs['changed_history_targets']+=int(row['history_changed']);updates.write(json.dumps(row)+'\n')
   if row['iteration']==59:
    assert len(params['means3D'])<=6000000,'Registered point limit'
    if params['means3D'].is_cuda:assert torch.cuda.memory_allocated()<=.85*torch.cuda.get_device_properties(0).total_memory,'Registered GPU allocation limit'
    if shutil.disk_usage(R).free<5.15*2**30:raise RuntimeError('Registered5GiB disk floor plus snapshot reserve')
   return ret
  opt.step=wrapped;return opt
 def addition(params,variables,data,sil_thres,index,mean_sq_dist_method,gaussian_distribution,confidence_seed_threshold=None):
  if not H:
   from c04n_addition import add_new_gaussians
   params,variables=add_new_gaussians(params,variables,data,sil_thres,index,mean_sq_dist_method,gaussian_distribution,confidence_seed_threshold,
       core=core,mode='native',cap=3000,seed=seed,log_path=out/'native_addition.jsonl',fb=cfg['mapping']['residual_fb'])
   p=masks/f'{index:06d}.npz'
   with np.load(p) as z:n=len(z['selected_ids'])
   after=len(params['means3D']);before=after-n
  else:
   src=donor/'addition_masks'/f'{index:06d}.npz';p=masks/src.name
   p.hardlink_to(src)
   with np.load(p) as z:ids=z['selected_ids'].copy();native_tail=bool(z['native_raw_any'])
   n=len(ids);assert n<=3000 and len(np.unique(ids))==n
   mask=torch.zeros(data['depth'].numel(),dtype=torch.bool,device='cuda');mask[torch.tensor(ids,device='cuda',dtype=torch.long)]=True
   assert bool((data['depth'].reshape(-1)[mask]>0).all())
   if confidence_seed_threshold is not None:assert bool((data['depth_confidence'].reshape(-1)[mask]>=confidence_seed_threshold).all())
   # Preserve the native diagnostic depth/silhouette forward in the paired run.
   g=core.transform_to_frame(params,index,gaussians_grad=False,camera_grad=False);core.Renderer(raster_settings=data['cam'])(**core.transformed_params2depthplussilhouette(params,data['w2c'],g))
   before=len(params['means3D'])
   if native_tail:
    q=F.normalize(params['cam_unnorm_rots'][...,index].detach());t=params['cam_trans'][...,index].detach();w2c=torch.eye(4,device='cuda');w2c[:3,:3]=core.build_rotation(q);w2c[:3,3]=t
    cloud,dist=core.get_pointcloud(data['im'],data['depth'],data['intrinsics'],w2c,mask=mask,compute_mean_sq_dist=True,mean_sq_dist_method=mean_sq_dist_method);new=core.initialize_new_params(cloud,dist,gaussian_distribution)
    for k,v in new.items():params[k]=torch.nn.Parameter(torch.cat((params[k],v),0).requires_grad_(True))
    for k in ['means2D_gradient_accum','denom','max_2D_radius']:variables[k]=torch.zeros(len(params['means3D']),device='cuda').float()
    variables['timestep']=torch.cat((variables['timestep'],index*torch.ones(n,device='cuda').float()),0)
   after=len(params['means3D']);assert after-before==n
  row=dict(frame=index,added=n,before=before,after_add=after,mask_sha256=hashlib.sha256(p.read_bytes()).hexdigest());obs['additions'].append(row)
  with (out/'addition_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return params,variables
 def finish():
  np.savez_compressed(out/'target_schedule.npz',base=base,history_03=hist,flag_03=flag)
  updates.close();pools.close()
 core.get_loss=loss;core.initialize_optimizer=optimizer;core.add_new_gaussians=addition
 return obs,finish
