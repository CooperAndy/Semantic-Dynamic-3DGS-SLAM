"""Branch-aware DEV entry gate, then unchanged actual-L1 four-arm replays."""
from pathlib import Path
import argparse,json,sys,time,hashlib
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1];P=json.loads((F/'repair_protocol.json').read_text())
LOCK=json.loads((F/'repair_execution_lock.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(F/'repair_protocol.json')==LOCK['protocol_sha256']
for name,h in LOCK['scripts'].items():assert sha(F/'scripts'/name)==h,name
for name,h in LOCK['receipts'].items():assert sha(F/'receipts'/name)==h,name
sys.path.insert(0,P['T15_root']+'/scripts')
import replay_runtime as rt
import numpy as np,torch
O=rt.P;gates=P['gate'];T15=Path(P['T15_root'])
old_gate=json.loads((T15/'receipts/numerical_gate.json').read_text());assert old_gate['status']=='FAIL'
assert rt.native==old_gate['natives']
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def numeric():
 assert not (F/'receipts/repaired_numerical_gate.json').exists()
 regional=json.loads((F/'receipts/conditional_derivatives.json').read_text());rows=[]
 for frame in P['frames']:
  rt.seed();ctx=rt.Context(frame);rng=rt.rnghash();score,g=ctx.gradient(ctx.T0);score2,g2=ctx.gradient(ctx.T0)
  prior=next(r for r in old_gate['cases'] if r['frame']==frame)
  assert ctx.map_hash==prior['map_hash'] and ctx.input_hash==prior['input_hash'] and ctx.forward_hash==prior['first_forward_hash']
  assert score==score2==prior['initial_loss']
  repeat=float(np.linalg.norm(g-g2)/max(np.linalg.norm(g),1.))
  h=P['primary_h'];fd=[]
  for axis in range(6):
   u=np.zeros(6);u[axis]=h;fd.append((ctx.value(ctx.step(ctx.T0,u))-ctx.value(ctx.step(ctx.T0,-u)))/(2*h))
  fd=np.asarray(fd);global_rel=float(np.linalg.norm(g-fd)/max(np.linalg.norm(g),np.linalg.norm(fd),1.));global_cos=rt.cos(g,fd)
  with np.load(T15/f'receipts/numeric_frame{frame}.npz') as z:
   full=z['P'].copy();half=z['P_half'].copy();double=z['P_double'].copy()
   assert np.array_equal(z['T0'],ctx.T0.cpu().numpy()) and np.array_equal(z['support'],ctx.support.cpu().numpy())
  directional=[]
  for name,C in [('ID',np.eye(6)),('FULL',full)]:
   d=-C@g;d/=np.linalg.norm(d);actual=(ctx.value(ctx.step(ctx.T0,h*d))-ctx.value(ctx.step(ctx.T0,-h*d)))/(2*h);pred=float(g@d)
   directional.append({'name':name,'analytic':pred,'FD':actual,'same_sign':bool(actual*pred>0),'relative':abs(actual-pred)/max(abs(actual),abs(pred),1.)})
  region=torch.zeros_like(ctx.support)
  for tile in P['diagnostic_tiles']:
   yy,xx=divmod(tile,(ctx.m.W+15)//16);region[yy*16:min((yy+1)*16,ctx.m.H),xx*16:min((xx+1)*16,ctx.m.W)]=True
  original=ctx.support;ctx.support=original&region;region_pixels=int(ctx.support.sum());_,rg=ctx.gradient(ctx.T0);ctx.support=original
  conditional=next(r for r in regional['cases'] if r['frame']==frame);assert region_pixels==conditional['regional_support_pixels'] and region_pixels>0
  fds=np.asarray(conditional['FD_gradients']);target=rg[P['diagnostic_axes']]
  cr=float(np.linalg.norm(target-fds[1])/max(np.linalg.norm(target),np.linalg.norm(fds[1]),1.));cc=rt.cos(target,fds[1]);cs=max(float(np.linalg.norm(v-fds[1])/max(np.linalg.norm(fds[1]),1.)) for v in fds)
  pm=max(float(np.linalg.norm(q-full)/np.linalg.norm(full)) for q in [half,double]);pc=min(rt.cos(q@g,full@g) for q in [half,double])
  tests={'conditional_relative':cr<=gates['conditional_gradient_relative_L2_max'],'conditional_cosine':cc>=gates['conditional_gradient_cosine_min'],
    'conditional_scale_stability':cs<=gates['conditional_FD_scale_relative_max'],
    'global_actual_relative':global_rel<=gates['global_actual_gradient_relative_L2_max'],'global_actual_cosine':global_cos>=gates['global_actual_gradient_cosine_min'],
    'global_actual_directions':all(r['same_sign'] and r['relative']<=gates['global_actual_ID_FULL_direction_relative_max'] for r in directional),
    'P_scale_stability':pm<=gates['P_scale_relative_max'],'Pg_scale_stability':pc>=gates['Pg_scale_cosine_min'],
    'backward_repeat':repeat<=gates['backward_repeat_relative_max'],
    'CPU_reconstruction':all(r['RGB_max']<=gates['native_CPU_RGB_max'] and r['depth_max']<=gates['native_CPU_depth_max'] for r in conditional['CPU_reconstruction'])}
  row={'frame':frame,'status':'PASS' if all(tests.values()) else 'FAIL','tests':tests,'conditional_gradient':target.tolist(),'conditional_FD':fds.tolist(),
       'conditional_relative_L2':cr,'conditional_cosine':cc,'conditional_FD_scale_relative':cs,'global_relative_L2':global_rel,'global_cosine':global_cos,
       'P_scale_relative':pm,'Pg_scale_cosine':pc,'backward_repeat_relative':repeat,'directional_checks':directional,
       'initial_loss':score,'initial_gradient':g.tolist(),'map_hash':ctx.map_hash,'input_hash':ctx.input_hash,'forward_hash':ctx.forward_hash,'regional_support_pixels':region_pixels}
  ctx.finish();assert rt.rnghash()==rng;rows.append(row);print(json.dumps(row),flush=True)
 out={'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','scope':'DEV_REPLAY_ENTRY_ONLY','cases':rows,'protocol_sha256':sha(F/'repair_protocol.json'),'lock_sha256':sha(F/'repair_execution_lock.json'),'original_T15':'FAIL_UNCHANGED','GT_audit':rt.GT_AUDIT}
 dump(F/'receipts/repaired_numerical_gate.json',out);print('REPAIRED_NUMERICAL_GATE',out['status'],flush=True)
def replay(frame,arm):
 gate=json.loads((F/'receipts/repaired_numerical_gate.json').read_text());assert gate['status']=='PASS'
 out=F/f'replays/f{frame}_{arm}';out.mkdir(parents=True,exist_ok=False)
 rt.seed();ctx=rt.Context(frame);rng=rt.rnghash();bind=next(r for r in gate['cases'] if r['frame']==frame)
 assert ctx.map_hash==bind['map_hash'] and ctx.input_hash==bind['input_hash'] and ctx.forward_hash==bind['forward_hash']
 with np.load(T15/f'receipts/numeric_frame{frame}.npz') as z:H=z['H'].copy();full=z['P'].copy()
 C=np.eye(6) if arm=='LOCAL_ID' else np.diag(np.diag(full)) if arm=='LOCAL_DIAG_P' else full
 T=ctx.T0.clone();pre=[];post=[];grad=[];cost=[];accepted=[];steps=[];proposals=[];reason='iteration_budget';started=time.time()
 for k in range(O['iteration_max']):
  score,g=ctx.gradient(T)
  if k==0:
   assert score==bind['initial_loss'];assert np.linalg.norm(g-np.asarray(bind['initial_gradient']))/max(np.linalg.norm(g),1.)<=gates['backward_repeat_relative_max']
  pre.append(T.cpu().numpy());grad.append(g);cost.append(score);ok=False;step=np.zeros(6)
  if np.linalg.norm(g)<=1e-12:reason='zero_gradient';accepted.append(False);steps.append(step);post.append(T.cpu().numpy());break
  d=-C@g;d/=np.linalg.norm(d);assert g@d<0
  for j in range(O['backtracks']):
   u=O['step_radius']*O['backtrack_factor']**j*d;candidate=ctx.step(T,u)
   distance=float(np.linalg.norm(rt.logse3(np.linalg.inv(ctx.T0.cpu().numpy().astype(float))@candidate.cpu().numpy())/np.asarray(O['scale'])))
   within=arm!='LOCAL_FULL_CAP' or distance<=O['total_anchor_radius']+1e-12
   value=ctx.value(candidate) if within else None;bound=score+O['Armijo_c1']*float(g@u)
   proposals.append({'iteration':k,'backtrack':j,'step':u.tolist(),'anchor_distance':distance,'within_cap':within,'loss':value,'armijo_bound':bound})
   if within and value<=bound:T=candidate;ok=True;step=u;break
  accepted.append(ok);steps.append(step);post.append(T.cpu().numpy())
  if not ok:reason='six_rejections_stop_case';break
 with torch.no_grad():
  q=ctx.render(T);final=float(ctx.loss(q));rgb=float(torch.abs(q['image']-ctx.color)[ctx.support].mean());dep=float(torch.abs(q['depth']-ctx.depth)[ctx.support].mean());coverage=float((q['acc'][ctx.support]>.95).float().mean())
 assert final<=cost[0] and rt.rnghash()==rng;ctx.finish()
 np.savez_compressed(out/'trace.npz',initial_c2w=ctx.T0.cpu().numpy(),candidate_c2w=T.cpu().numpy(),pre_c2w=np.asarray(pre),post_c2w=np.asarray(post),gradients=np.asarray(grad),loss=np.asarray(cost),accepted=np.asarray(accepted),step=np.asarray(steps),H=H,P_FULL=full,C=C)
 dump(out/'proposals.json',proposals)
 dump(out/'run_manifest.json',{'status':'COMPLETED','frame':frame,'arm':arm,'initial_loss':cost[0],'final_loss':final,'rgb_l1':rgb,'depth_l1':dep,'coverage':coverage,
   'iterations':len(cost),'accepted':sum(accepted),'reason':reason,'map_hash':ctx.map_hash,'input_hash':ctx.input_hash,'forward_hash':ctx.forward_hash,'rng_unchanged':True,'map_unchanged':True,
   'GT_audit':rt.GT_AUDIT,'seconds':time.time()-started,'protocol_sha256':sha(F/'repair_protocol.json'),'lock_sha256':sha(F/'repair_execution_lock.json'),
   'artifacts':{'trace.npz':sha(out/'trace.npz'),'proposals.json':sha(out/'proposals.json')}})
 print('REPLAY_COMPLETED',frame,arm,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['numeric','replay']);a.add_argument('--frame',type=int);a.add_argument('--arm',choices=O['arms']);v=a.parse_args()
 if v.stage=='numeric':numeric()
 else:assert v.frame in P['frames'] and v.arm;replay(v.frame,v.arm)
