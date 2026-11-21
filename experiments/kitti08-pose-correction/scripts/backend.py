"""Gaussian-depth photographic tracker; camera writes precede native mapping."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,torch
from scipy.spatial.transform import Rotation
from feature_pose import Matcher
from pose_chain import predict
from visibility import opacity_retention
import external_tracking as ext
from tracking_policy import support_info,residuals,decide
from utils.recon_helpers import setup_camera

class Backend:
 def __init__(self,core,cfg,out,manifest):
  self.core=core;self.cfg=cfg;self.out=Path(out);self.rows=manifest['rows'];self.policy=cfg['feedback_policy'];self.images={};self.nominal={};self.selected={};self.qt={};self.feedback=False;self.current=-1;self.records=[];self.pose_checks=0;self.render_calls=0;self.native_set=ext.set_external
  self.stream=(self.out/'backend_events.jsonl').open('x',buffering=1)
  self.proposals=self.out/'proposals';self.proposals.mkdir();self.floats=self.out/'tracking_float';self.floats.mkdir()
 def W(self,p,i):return ext.camera(p,i,self.core).cpu().numpy()
 def qt_at(self,p,i):return tuple(p[k][...,i].detach().cpu().numpy().copy() for k in ['cam_unnorm_rots','cam_trans'])
 def check(self,p,indices):
  for i in indices:
   q,t=self.qt_at(p,i);a,b=self.qt[i];assert np.array_equal(q,a) and np.array_equal(t,b),('Mapping changed selected camera',i);self.pose_checks+=1
 def observe(self,p,T,i,core):
  self.native_set(p,T,i,core);f=sys._getframe(1)
  if f.f_code.co_name!='rgbd_slam':return
  assert i==self.current+1
  if i:self.check(p,[i-1])
  self.current=i;data=f.f_locals['tracking_curr_data'];self.images[i]=data['im'].detach().cpu().numpy().copy();self.nominal[i]=self.W(p,i);self.selected[i]=self.nominal[i].copy();self.qt[i]=self.qt_at(p,i)
  if i==0 or self.policy=='ORB':self.write(dict(frame=i,policy=self.policy,accepted=False,reason='first_frame' if not i else 'external_fixed',had_feedback=False,nominal_w2c=self.nominal[i].tolist(),prediction_w2c=self.nominal[i].tolist(),selected_w2c=self.nominal[i].tolist(),selected_q=self.qt[i][0].tolist(),selected_t=self.qt[i][1].tolist()))
 def write(self,row):self.records.append(row);self.stream.write(json.dumps(row,allow_nan=False)+'\n')
 def render(self,p,data,i):self.render_calls+=2;return ext.render(p,data,i,self.core)
 def conditional(self,ds):return (torch.tensor(ds[0],device='cuda')/torch.tensor(ds[1],device='cuda').clamp_min(1e-4)).cpu().numpy()
 def refine(self,p,variables,data,i,config,core):
  assert i==self.current and i>0 and self.policy in ['HARD','MASS'];before=ext.map_hash(p);versions={k:p[k]._version for k in ext.MAP_KEYS};had=self.feedback;nominal=self.nominal[i];qnom,tnom=self.qt_at(p,i)
  prediction=predict(self.nominal[i-1],nominal,self.selected[i-1],had)
  if had:self.native_set(p,torch.tensor(prediction,dtype=torch.float32,device='cuda'),i,core)
  T0=self.W(p,i);q0,t0=(p[k][...,i].detach().clone() for k in ['cam_unnorm_rots','cam_trans']);r0,ds0=self.render(p,data,i);d0=self.conditional(ds0);target=data['im'].detach().cpu().numpy();depth=data['depth'].detach().cpu().numpy()[0];confidence=data.get('depth_confidence')
  if confidence is None:
   source=self.rows[i];cp=Path(source['confidence']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['hashes']['confidence'];confidence=np.load(cp)
  else:confidence=confidence.detach().cpu().numpy().squeeze()
  mask=(depth>0)&(confidence>=.8)&(ds0[1]>.99)&np.isfinite(ds0[0])&np.isfinite(ds0[2]-ds0[0]**2);info=support_info(mask,depth>0)
  anchor=i-1
  if anchor in set(config['eval_only_frames']):anchor-=1
  assert 0<=anchor<i and anchor in self.selected and anchor in self.images;self.check(p,[anchor]);K=data['intrinsics'].detach().cpu().numpy();Kr=K.copy();Kr[:2,2]+=.5;H,W=depth.shape;rd=dict(data,cam=setup_camera(W,H,Kr,data['w2c'].detach().cpu().numpy()));unused,refds=self.render(p,rd,anchor);refdepth=self.conditional(refds);refmask=(refds[1]>.99)&(refdepth>0)&np.isfinite(refdepth);refmask[:4]=False;refmask[-4:]=False;refmask[:,:4]=False;refmask[:,-4:]=False;A=self.W(p,anchor);initial=np.asarray(T0,np.float64)@np.linalg.inv(np.asarray(A,np.float64));proposal=Matcher(target).fit(self.images[anchor],refdepth,refds[1],refmask,K.astype(np.float64),initial,180028+i)
  propfile=self.proposals/f'{i:06}.json';propfile.write_text(json.dumps(proposal,allow_nan=False));r1=r0;ds1=ds0;d1=d0;T1=T0
  if proposal['ok']:
   self.native_set(p,torch.tensor(np.array(proposal['relative_w2c'])@np.asarray(A,np.float64),device='cuda',dtype=torch.float32),i,core);T1=self.W(p,i);r1,ds1=self.render(p,data,i);d1=self.conditional(ds1)
  qcandidate,tcandidate=self.qt_at(p,i);D=np.asarray(T1,np.float64)@np.linalg.inv(np.asarray(T0,np.float64));translation=float(np.linalg.norm(D[:3,3]));rotation=float(np.degrees(Rotation.from_matrix(D[:3,:3]).magnitude()));hard=dict(accepted=False,reason='insufficient_support');mass=dict(hard);metrics=None;retention=None
  if info['pixels']:
   ini=residuals(r0,d0,target,depth,mask,ds0[1],{'im':.5,'depth':1.});candidate=residuals(r1,d1,target,depth,mask,ds1[1],{'im':.5,'depth':1.});metrics=dict(initial=ini,candidate=candidate);hard=decide(ini,candidate,translation,rotation) if info['sufficient'] else dict(accepted=False,reason='insufficient_support');retention=opacity_retention(ds0[1],ds1[1],mask);mass=dict(hard)
   if 'gates' in hard:
    gates=dict(hard['gates']);gates['coverage_guard']=retention>=.98;mass=dict(accepted=all(gates.values()),reason='accepted' if all(gates.values()) else ','.join(k for k,v in gates.items() if not v),gates=gates)
  if not proposal['ok']:hard=dict(accepted=False,reason=proposal['reason']);mass=dict(hard)
  decision=mass if self.policy=='MASS' else hard
  if not decision['accepted']:
   with torch.no_grad():p['cam_unnorm_rots'][...,i]=q0;p['cam_trans'][...,i]=t0
  q,t=self.qt_at(p,i);selected=self.W(p,i)
  if not had and not decision['accepted']:assert np.array_equal(q,qnom) and np.array_equal(t,tnom)
  assert ext.map_hash(p)==before and all(p[k]._version==versions[k] for k in ext.MAP_KEYS)
  for k in ['cam_unnorm_rots','cam_trans']:p[k].grad=None
  row=dict(frame=i,policy=self.policy,anchor=anchor,had_feedback=had,accepted=decision['accepted'],reason=decision['reason'],nominal_w2c=nominal.tolist(),prediction_w2c=T0.tolist(),candidate_w2c=np.asarray(T1).tolist(),selected_w2c=selected.tolist(),prediction_q=q0.cpu().tolist(),prediction_t=t0.cpu().tolist(),candidate_q=qcandidate.tolist(),candidate_t=tcandidate.tolist(),selected_q=q.tolist(),selected_t=t.tolist(),geometry_ok=proposal['ok'],geometry_reason=proposal['reason'],support=info,support_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),hard=hard,mass=mass,opacity_retention=retention,metrics=metrics,translation_m=translation,rotation_deg=rotation,proposal_path=propfile.relative_to(self.out).as_posix(),proposal_sha256=hashlib.sha256(propfile.read_bytes()).hexdigest(),map_before_sha256=before,map_after_sha256=before,steps=[],map_write_allowed=True,validation_renders=6 if proposal['ok'] else 4)
  if i in [1,config['data']['num_frames']//2,config['data']['num_frames']-1]:
   f=self.floats/f'{i:06}.npz';np.savez_compressed(f,initial_rgb=r0,candidate_rgb=r1,initial_ds=ds0,candidate_ds=ds1,initial_depth=d0,candidate_depth=d1,target_rgb=target,target_depth=depth,confidence=confidence,support=mask);row.update(float_path=f.relative_to(self.out).as_posix(),float_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
  self.selected[i]=selected;self.qt[i]=(q,t);self.feedback=had or decision['accepted'];self.write(row);return p,variables,row
 def install(self):
  import backtracking_refine
  ext.set_external=self.observe;backtracking_refine.refine=self.refine
 def finish(self):self.stream.close()
