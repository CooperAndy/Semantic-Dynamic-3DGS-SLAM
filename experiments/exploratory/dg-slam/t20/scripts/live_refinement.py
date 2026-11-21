"""T12's unchanged local rule on real tracking inputs and a frozen past map."""
import hashlib,time
from pathlib import Path
import numpy as np,torch
from pose_math import exp_se3
from bounded_step_cpu import bounded_trial
from dg_slam.gaussian.common import get_tensor_from_camera,get_camera_from_tensor
from runtime_boot import P
def map_fingerprint(m):
 h=hashlib.sha256()
 for name in ['xyz','features_dc','features_rest','opacity','scaling','rotation']:
  a=getattr(m,'gaussians_'+name).detach().cpu().contiguous().numpy();h.update(name.encode());h.update(a.tobytes())
 return h.hexdigest()
def local_tracking(m,idx,color,depth,estimated,seg,mode,folder,counts):
 idx=int(idx);m.update_para_from_mapping()
 if idx<=1:
  T=torch.eye(4,device=estimated.device) if idx==0 else estimated
  m.estimate_c2w_list[idx]=T.detach().cpu();m.pre_c2w=T.detach().clone();m.idx[0]=idx
  return {'eligible':False,'reason':'initial_two_keyframes'}
 m.count_bound=(idx%5) if idx%5>0 else 5
 before=map_fingerprint(m);started=time.perf_counter()
 raw=get_tensor_from_camera(estimated.detach()).to(m.device)
 T=torch.cat([get_camera_from_tensor(raw),raw.new_tensor([[0,0,0,1]])],dim=0).float();initial=T.detach().clone()
 S=torch.tensor(P['candidate_recipe']['scale'],device=T.device,dtype=torch.float64)
 import importlib
 mod=importlib.import_module(type(m).__module__)
 def render(T):
  pkg=mod.render(m.gaussians_xyz,m.gaussians_features_dc,m.gaussians_features_rest,m.opacity_activation(m.gaussians_opacity),m.scaling_activation(m.gaussians_scaling),m.rotation_activation(m.gaussians_rotation),m.gaussians.get_active_sh_degree(),m.gaussians.get_max_sh_degree(),T[:3,3],torch.linalg.inv(T).T.contiguous(),m.projection_matrix,m.fovx,m.fovy,m.H,m.W)
  return {'image':pkg['render'].permute(1,2,0),'depth':pkg['depth'][0],'acc':pkg['acc'].squeeze()}
 with torch.no_grad():first=render(T)
 support=(depth>0)&(depth<8)&torch.isfinite(depth)&torch.isfinite(first['depth'])&seg&(first['acc']>.95)
 edge=torch.zeros_like(support);edge[10:-10,10:-10]=True;support&=edge
 N=int(support.sum());m.opacity_mask=first['acc']>.95
 if N<1000:
  m.estimate_c2w_list[idx]=T.detach().cpu();m.pre_c2w=T.detach().clone();m.idx[0]=idx
  counts['support_fallbacks']+=1;assert before==map_fingerprint(m)
  return {'eligible':True,'reason':'insufficient_support','support':N,'fallback':True,'map_unchanged':True}
 def loss(pkg):return .9*torch.abs(pkg['image']-color)[support].sum()+.1*torch.clamp(torch.abs(pkg['depth']-depth),max=1000)[support].sum()
 jc=[];jd=[]
 for axis in range(6):
  eps=.001 if axis<3 else np.pi/180*.01;pair=[]
  for sign in [-1,1]:
   xi=torch.zeros(6,device=T.device,dtype=torch.float64);xi[axis]=eps*sign
   with torch.no_grad():q=render((T.double()@exp_se3(xi)).float())
   pair.append(q)
  jc.append(((pair[1]['image'].double()-pair[0]['image'].double())/(2*eps)*S[axis])[support].cpu().numpy())
  jd.append(((pair[1]['depth'].double()-pair[0]['depth'].double())/(2*eps)*S[axis])[support].cpu().numpy())
 jc=np.stack(jc,axis=-1);jd=np.stack(jd,axis=-1)
 H=(.9*np.einsum('nci,ncj->ij',jc,jc)+.1*np.einsum('ni,nj->ij',jd,jd))/N;H=(H+H.T)/2
 ev,Q=np.linalg.eigh(H);norm=np.trace(H)/6
 assert np.isfinite(H).all() and norm>0 and ev[0]>=-1e-10*norm
 full=(Q/np.sqrt(ev/norm+.01))@Q.T;full*=np.sqrt(6)/np.linalg.norm(full)
 C=full if mode=='LOCAL_FULL' else np.eye(6)
 assert np.linalg.eigvalsh(C)[0]>0
 pre=[];post=[];grad=[];cost=[];accepted=[];steps=[];props=[];propsteps=[];proploss=[];propiter=[];bounds=[];reason='iteration_budget'
 for k in range(50):
  x=torch.zeros(6,device=T.device,dtype=torch.float64,requires_grad=True)
  cam=(T.double()@exp_se3(S*x)).float();assert torch.equal(cam,T)
  pkg=render(cam);value=loss(pkg);score=float(value.detach());assert np.isfinite(score)
  value.backward();counts['camera_backward']+=1;g=x.grad.detach().cpu().numpy().copy();assert np.isfinite(g).all()
  pre.append(T.detach().cpu().numpy());grad.append(g);cost.append(score);candidates=[]
  def evaluate(step):
   with torch.no_grad():candidate=(T.double()@exp_se3(S*torch.from_numpy(step).to(T.device))).float();v=float(loss(render(candidate)))
   candidates.append(candidate.detach().clone());props.append(candidate.cpu().numpy());propsteps.append(step.copy());proploss.append(v);propiter.append(k);bounds.append(score+1e-4*float(g@step));return v
  result=bounded_trial(g,C,score,evaluate)
  if result['accepted']:T=candidates[-1];counts['camera_accepted_updates']+=1
  accepted.append(result['accepted']);steps.append(result['step']);post.append(T.detach().cpu().numpy())
  if not result['accepted']:reason=result['reason'];break
 with torch.no_grad():final=render(T);final_loss=float(loss(final))
 assert abs(final_loss-result['loss'])<=1e-5*max(abs(result['loss']),1.) and final_loss<=cost[0]+1e-5*max(cost[0],1.)
 assert before==map_fingerprint(m)
 path=folder/f'camera_kf{idx:03d}.npz'
 np.savez_compressed(path,initial_c2w=initial.cpu().numpy(),candidate_c2w=T.cpu().numpy(),H=H,P_FULL=full,C=C,support=support.cpu().numpy(),pre_c2w=np.asarray(pre),post_c2w=np.asarray(post),gradients=np.asarray(grad),loss=np.asarray(cost),accepted=np.asarray(accepted),step=np.asarray(steps),proposal_c2w=np.asarray(props).reshape(-1,4,4),proposal_step=np.asarray(propsteps).reshape(-1,6),proposal_loss=np.asarray(proploss),proposal_iteration=np.asarray(propiter),proposal_armijo=np.asarray(bounds))
 m.estimate_c2w_list[idx]=T.detach().cpu();m.pre_c2w=T.detach().clone();m.idx[0]=idx
 if m.low_gpu_mem:torch.cuda.empty_cache()
 return {'eligible':True,'support':N,'fallback':False,'reason':reason,'iterations':len(cost),'accepted':sum(accepted),'initial_loss':cost[0],'final_loss':final_loss,'map_unchanged':True,'seconds':time.perf_counter()-started,'trace':str(path)}
