"""Real-observation numerical gate and conditional same-map DEV replays. No GT."""
from pathlib import Path
import argparse, hashlib, importlib, json, os, random, sys, time
sys.dont_write_bytecode=True
import numpy as np
F=Path(__file__).resolve().parents[1]
P=json.loads((F/'protocol.json').read_text());LOCK=json.loads((F/'execution_lock.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
assert sha(F/'protocol.json')==LOCK['protocol_sha256']
for name,h in LOCK['scripts'].items():assert sha(F/'scripts'/name)==h,name
for name,h in LOCK['inputs'].items():assert sha(F/name)==h,name
assert __import__('shutil').disk_usage(F).free>P['limits']['minimum_free_bytes']
T14=Path(P['t14_root']);sys.path.insert(0,str(T14/'scripts'))
from runtime_boot import bootstrap,guard_gt
native=bootstrap();GT_AUDIT=guard_gt()
sys.path.insert(0,str(F/'scripts'))
import torch
from pose_math import exp_se3
from dg_slam.gs_tracking_mapping import gs_tracking_mapping
from dg_slam.gaussian.common import get_tensor_from_camera,get_camera_from_tensor
from scipy.spatial.transform import Rotation
mod=importlib.import_module('dg_slam.gs_tracking_mapping')
S=torch.tensor(P['scale'],device='cuda',dtype=torch.float64)
def cos(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def seed():
 random.seed(P['seed']);np.random.seed(P['seed']);torch.manual_seed(P['seed']);torch.cuda.manual_seed_all(P['seed'])
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def fingerprint(values):
 h=hashlib.sha256()
 for k,a in values.items():
  if torch.is_tensor(a):a=a.detach().cpu().contiguous().numpy()
  h.update(k.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
def rnghash():
 state=np.random.get_state();v={'np':state[1],'cpu':torch.get_rng_state()}
 for k,a in enumerate(torch.cuda.get_rng_state_all()):v['cuda'+str(k)]=a
 return hashlib.sha256((fingerprint(v)+str(random.getstate())+str((state[0],state[2],state[3],state[4]))).encode()).hexdigest()
def logse3(T):
 w=Rotation.from_matrix(T[:3,:3]).as_rotvec();a=np.linalg.norm(w)
 W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
 c=1/12+a*a/720 if a<1e-3 else (1-a/2/np.tan(a/2))/(a*a)
 return np.r_[(np.eye(3)-W/2+c*W@W)@T[:3,3],w]
class Context:
 def __init__(self,frame):
  self.frame=frame;spec=next(c for c in P['cases'] if c['frame']==frame)
  with np.load(spec['map_path']) as z:self.v={k:torch.from_numpy(a.copy()).cuda() for k,a in z.items()}
  with np.load(spec['case_path']) as z:
   self.color=torch.from_numpy(z['observed_color'].copy()).cuda();self.depth=torch.from_numpy(z['observed_depth'].copy()).cuda()
   self.seg=torch.from_numpy(z['seg_mask'].copy()).cuda().bool();rawT=torch.from_numpy(z['coarse_c2w'].copy()).cuda()
  raw=get_tensor_from_camera(rawT).to(rawT.device);self.T0=torch.cat([get_camera_from_tensor(raw),raw.new_tensor([[0,0,0,1]])],dim=0).float()
  original=json.loads((F/'inputs/original_run_manifest.json').read_text())
  m=gs_tracking_mapping.__new__(gs_tracking_mapping);m.cfg=original['cfg'];m.scale_factor=.8
  m.H,m.W,m.fx,m.fy,m.cx,m.cy=[m.cfg['cam'][k] for k in ['H','W','fx','fy','cx','cy']]
  m.edge=None;m.update_scale();m.update_cam();m.fovx=mod.focal2fov(m.fx,m.W);m.fovy=mod.focal2fov(m.fy,m.H)
  m.projection_matrix=mod.getProjectionMatrix(znear=.01,zfar=100.,fovX=m.fovx,fovY=m.fovy).transpose(0,1).cuda()
  self.m=m;self.map_hash=fingerprint(self.v);self.render_calls=0
  with torch.no_grad():first=self.render(self.T0)
  self.forward_hash=fingerprint(first)
  support=(self.depth>0)&(self.depth<8)&torch.isfinite(self.depth)&torch.isfinite(first['depth'])&self.seg&(first['acc']>.95)
  edge=torch.zeros_like(support);edge[10:-10,10:-10]=True;self.support=support&edge;self.N=int(self.support.sum())
  assert self.N>=1000
  self.input_hash=fingerprint({'T0':self.T0,'color':self.color,'depth':self.depth,'seg':self.seg,'support':self.support})
 def render(self,T):
  m=self.m;v=self.v
  pkg=mod.render(v['xyz'],v['features_dc'],v['features_rest'],torch.sigmoid(v['opacity']),torch.exp(v['scaling']),torch.nn.functional.normalize(v['rotation']),3,3,T[:3,3],torch.linalg.inv(T).T.contiguous(),m.projection_matrix,m.fovx,m.fovy,m.H,m.W)
  self.render_calls+=1
  return {'image':pkg['render'].permute(1,2,0),'depth':pkg['depth'][0],'acc':pkg['acc'].squeeze()}
 def loss(self,q):return .9*torch.abs(q['image']-self.color)[self.support].sum()+.1*torch.clamp(torch.abs(q['depth']-self.depth),max=1000)[self.support].sum()
 def step(self,T,u):return (T.double()@exp_se3(S*torch.as_tensor(u,device='cuda',dtype=torch.float64))).float()
 def value(self,T):
  with torch.no_grad():return float(self.loss(self.render(T)))
 def gradient(self,T):
  x=torch.zeros(6,device='cuda',dtype=torch.float64,requires_grad=True);camera=(T.double()@exp_se3(S*x)).float();assert torch.equal(camera,T)
  q=self.render(camera);loss=self.loss(q);loss.backward();g=x.grad.detach().cpu().numpy().copy()
  assert np.isfinite(g).all();return float(loss.detach()),g
 def matrix(self,multiplier):
  jc=[];jd=[]
  for axis in range(6):
   eps=(.001 if axis<3 else np.pi/180*.01)*multiplier;u=np.zeros(6);u[axis]=eps/P['scale'][axis]
   with torch.no_grad():minus=self.render(self.step(self.T0,-u));plus=self.render(self.step(self.T0,u))
   jc.append(((plus['image'].double()-minus['image'].double())/(2*eps)*S[axis])[self.support].cpu().numpy())
   jd.append(((plus['depth'].double()-minus['depth'].double())/(2*eps)*S[axis])[self.support].cpu().numpy())
  jc=np.stack(jc,axis=-1);jd=np.stack(jd,axis=-1)
  H=(.9*np.einsum('nci,ncj->ij',jc,jc)+.1*np.einsum('ni,nj->ij',jd,jd))/self.N;H=(H+H.T)/2
  ev,Q=np.linalg.eigh(H);scale=np.trace(H)/6;assert ev[0]>=-1e-10*scale and scale>0
  C=(Q/np.sqrt(ev/scale+.01))@Q.T;C*=np.sqrt(6)/np.linalg.norm(C)
  return H,C
 def finish(self):assert fingerprint(self.v)==self.map_hash and GT_AUDIT['blocked_GT_opens']==0
def numeric():
 assert not (F/'receipts/numerical_gate.json').exists()
 rows=[];gates=P['numeric_gate'];started=time.time()
 for frame in [c['frame'] for c in P['cases']]:
  seed();ctx=Context(frame);rng=rnghash();score,g=ctx.gradient(ctx.T0)
  with torch.no_grad():repeat=ctx.render(ctx.T0)
  assert fingerprint(repeat)==ctx.forward_hash
  Hs={};Ps={}
  for multiplier in gates['matrix_eps_multipliers']:Hs[multiplier],Ps[multiplier]=ctx.matrix(multiplier)
  C=Ps[1.];fd=[];fd_details=[]
  for h in gates['L1_FD_normalized_h']:
   v=[]
   for axis in range(6):
    u=np.zeros(6);u[axis]=h;v.append((ctx.value(ctx.step(ctx.T0,u))-ctx.value(ctx.step(ctx.T0,-u)))/(2*h))
   v=np.array(v);fd.append(v);fd_details.append({'h':h,'gradient':v.tolist(),'relative_L2':float(np.linalg.norm(g-v)/max(np.linalg.norm(g),np.linalg.norm(v),1.)), 'cosine':cos(g,v)})
  primary=fd[1];scale_diff=max(float(np.linalg.norm(v-primary)/max(np.linalg.norm(primary),1.)) for v in fd)
  directions=[]
  for name,M in [('ID',np.eye(6)),('FULL',C)]:
   d=-M@g;d/=np.linalg.norm(d)
   for h in gates['L1_FD_normalized_h']:
    actual=(ctx.value(ctx.step(ctx.T0,h*d))-ctx.value(ctx.step(ctx.T0,-h*d)))/(2*h);pred=float(g@d)
    directions.append({'direction':name,'h':h,'analytic':pred,'FD':actual,'relative':abs(pred-actual)/max(abs(pred),abs(actual),1.),'same_sign':bool(pred*actual>0)})
  matrix_details=[]
  for mult in [.5,2.]:
   matrix_details.append({'eps_multiplier':mult,'P_relative_Frobenius':float(np.linalg.norm(Ps[mult]-C)/np.linalg.norm(C)), 'Pg_cosine':cos(Ps[mult]@g,C@g), 'H_relative_Frobenius':float(np.linalg.norm(Hs[mult]-Hs[1.])/np.linalg.norm(Hs[1.]))})
  tests={'gradient_relative':fd_details[1]['relative_L2']<=gates['gradient_relative_L2_max'],
         'gradient_cosine':fd_details[1]['cosine']>=gates['gradient_cosine_min'],
         'gradient_scale_stability':scale_diff<=gates['FD_gradient_scale_relative_max'],
         'directional_derivatives':all(r['same_sign'] and r['relative']<=gates['ID_FULL_directional_relative_max'] for r in directions if r['h']==gates['primary_h']),
         'P_scale_stability':all(r['P_relative_Frobenius']<=gates['P_relative_Frobenius_max'] for r in matrix_details),
         'Pg_scale_stability':all(r['Pg_cosine']>=gates['Pg_direction_cosine_min'] for r in matrix_details)}
  np.savez_compressed(F/f'receipts/numeric_frame{frame}.npz',T0=ctx.T0.cpu().numpy(),support=ctx.support.cpu().numpy(),g=g,FD_gradients=np.asarray(fd),H=Hs[1.],P=C,H_half=Hs[.5],H_double=Hs[2.],P_half=Ps[.5],P_double=Ps[2.])
  assert rng==rnghash();ctx.finish()
  row={'frame':frame,'status':'PASS' if all(tests.values()) else 'FAIL','tests':tests,'initial_loss':score,'support':ctx.N,'map_hash':ctx.map_hash,'input_hash':ctx.input_hash,'first_forward_hash':ctx.forward_hash,'gradient':g.tolist(),'FD_gradient_checks':fd_details,'FD_gradient_scale_relative_max':scale_diff,'directional_checks':directions,'matrix_checks':matrix_details,'renders':ctx.render_calls,'rng_unchanged':True}
  rows.append(row);print(json.dumps(row),flush=True)
 result={'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','cases':rows,'seconds':time.time()-started,'protocol_sha256':sha(F/'protocol.json'),'lock_sha256':sha(F/'execution_lock.json'),'GT_audit':GT_AUDIT,'natives':native}
 dump(F/'receipts/numerical_gate.json',result);print('NUMERICAL_GATE',result['status'],flush=True)
def replay(frame,arm):
 gate=json.loads((F/'receipts/numerical_gate.json').read_text());assert gate['status']=='PASS'
 output=F/f'runs/f{frame}_{arm}';output.mkdir(exist_ok=False);seed();ctx=Context(frame);rng=rnghash()
 bind=next(r for r in gate['cases'] if r['frame']==frame)
 assert all(getattr(ctx,k)==bind[k] for k in ['map_hash','input_hash']) and ctx.forward_hash==bind['first_forward_hash']
 with np.load(F/f'receipts/numeric_frame{frame}.npz') as z:
  assert np.array_equal(z['T0'],ctx.T0.cpu().numpy()) and np.array_equal(z['support'],ctx.support.cpu().numpy());H=z['H'].copy();full=z['P'].copy()
 C=np.eye(6) if arm=='LOCAL_ID' else np.diag(np.diag(full)) if arm=='LOCAL_DIAG_P' else full
 T=ctx.T0.clone();pre=[];post=[];grad=[];cost=[];accepted=[];steps=[];proposals=[];reason='iteration_budget';started=time.time()
 for k in range(P['iteration_max']):
  score,g=ctx.gradient(T)
  if k==0:assert score==bind['initial_loss'] and np.array_equal(g,np.asarray(bind['gradient']))
  pre.append(T.cpu().numpy());grad.append(g);cost.append(score);ok=False;step=np.zeros(6)
  if np.linalg.norm(g)<=1e-12:reason='zero_gradient';break
  d=-C@g;d/=np.linalg.norm(d);assert g@d<0
  for j in range(P['backtracks']):
   u=P['step_radius']*P['backtrack_factor']**j*d;candidate=ctx.step(T,u)
   distance=float(np.linalg.norm(logse3(np.linalg.inv(ctx.T0.cpu().numpy().astype(float))@candidate.cpu().numpy())/np.asarray(P['scale'])))
   within=arm!='LOCAL_FULL_CAP' or distance<=P['total_anchor_radius']+1e-12
   value=ctx.value(candidate) if within else None;bound=score+P['Armijo_c1']*float(g@u)
   proposals.append({'iteration':k,'backtrack':j,'step':u.tolist(),'anchor_distance':distance,'within_cap':within,'loss':value,'armijo_bound':bound})
   if within and value<=bound:T=candidate;ok=True;step=u;break
  accepted.append(ok);steps.append(step);post.append(T.cpu().numpy())
  if not ok:reason='six_rejections_stop_case';break
 with torch.no_grad():
  q=ctx.render(T);final=float(ctx.loss(q));rgb=float(torch.abs(q['image']-ctx.color)[ctx.support].mean());dep=float(torch.abs(q['depth']-ctx.depth)[ctx.support].mean());coverage=float((q['acc'][ctx.support]>.95).float().mean())
 assert final<=cost[0] and rng==rnghash();ctx.finish()
 np.savez_compressed(output/'trace.npz',initial_c2w=ctx.T0.cpu().numpy(),candidate_c2w=T.cpu().numpy(),pre_c2w=np.asarray(pre),post_c2w=np.asarray(post),gradients=np.asarray(grad),loss=np.asarray(cost),accepted=np.asarray(accepted),step=np.asarray(steps),H=H,P_FULL=full,C=C)
 dump(output/'proposals.json',proposals)
 dump(output/'run_manifest.json',{'status':'COMPLETED','frame':frame,'arm':arm,'initial_loss':cost[0],'final_loss':final,'rgb_l1':rgb,'depth_l1':dep,'coverage':coverage,'iterations':len(cost),'accepted':sum(accepted),'reason':reason,'map_hash':ctx.map_hash,'input_hash':ctx.input_hash,'forward_hash':ctx.forward_hash,'rng_unchanged':True,'map_unchanged':True,'GT_audit':GT_AUDIT,'seconds':time.time()-started,'protocol_sha256':sha(F/'protocol.json'),'lock_sha256':sha(F/'execution_lock.json'),'artifacts':{'trace.npz':sha(output/'trace.npz'),'proposals.json':sha(output/'proposals.json')}})
 print('REPLAY_COMPLETED',frame,arm,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['numeric','replay']);a.add_argument('--frame',type=int);a.add_argument('--arm',choices=P['arms']);args=a.parse_args()
 if args.stage=='numeric':numeric()
 else:assert args.frame in [47,95] and args.arm;replay(args.frame,args.arm)
