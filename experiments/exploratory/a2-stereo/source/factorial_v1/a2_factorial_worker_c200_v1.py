"""A2 mapper DEV run; isolated source; actual 3000 births per append frame."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');ROOT=O/'factorial_v1';S=ROOT/'source';BASE=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/C200')
ARM=sys.argv[1];COUNT=int(sys.argv[2]);assert ARM in ('C0','C1','C2','C3','C4','C5') and COUNT in (25,200)
OUT=ROOT/'runs'/f'{ARM}_{COUNT}';start=time.monotonic();sys.path.insert(0,str(O));sys.path.insert(0,str(S));os.chdir(S);torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
registration=json.loads((ROOT/'run_registration.json').read_text())
if registration.get('state') != 'APPROVED_FOR_EXECUTION_AFTER_AUDIT':raise RuntimeError('A2 factorial draft blocked: sequence/calibration audit incomplete')
for entry in registration['file_locks']:
 if sha(entry['path'])!=entry['sha256']:raise RuntimeError('registered file changed: '+entry['path'])
cal_path=O/'gradient_calibration_20261028.json';cal=json.loads(cal_path.read_text());assert cal['status']=='PASS_GRADIENT_SCALE_MATCH_ONLY'
manifest=json.loads((BASE/'inputs/runtime/input_manifest.json').read_text());rows={r['frame_id']:r for r in manifest['rows']}
if sha(manifest['pose_file'])!=manifest['pose_sha256']:raise RuntimeError('pose mismatch')
calib_path=Path('/root/autodl-tmp/stage3_g05C_20261025/input_view/calib.txt')
if sha(calib_path)!='1bb615b4535ac73053be176ac1650fcfe7678ae8225d8999e5a822aa71655682':raise RuntimeError('KITTI00 calibration changed')
projections={}
for line in calib_path.read_text().splitlines():
 if line.startswith(('P2:','P3:')):
  key,values=line.split(':',1);projections[key]=np.array([float(v) for v in values.split()]).reshape(3,4)
P2,P3=projections['P2'],projections['P3'];C2=-np.linalg.solve(P2[:,:3],P2[:,3]);C3=-np.linalg.solve(P3[:,:3],P3[:,3]);fb=float(P2[0,0]*abs(C3[0]-C2[0]))
if not np.allclose(P2[:,:3],np.array(manifest['K']),atol=1e-10) or not np.isclose(fb,383.3593894935312,rtol=1e-12):raise RuntimeError('wrong KITTI00 K or fb')
for i in range(COUNT):
 for k in ('rgb','depth','confidence'):
  if sha(rows[i][k])!=rows[i]['hashes'][k]:raise RuntimeError(f'input mismatch {i} {k}')
OUT.mkdir(parents=True,exist_ok=False)
cfg=load('a2_config',BASE/'source/configs/kitti/G00_S0.py').config;cfg['workdir']=str(ROOT/'runs');cfg['run_name']=OUT.name;cfg['data']['num_frames']=COUNT;cfg['eval_only_frames']=[i for i in cfg['eval_only_frames'] if i<COUNT]
assert cfg['data']['sequence']=='00' and manifest['H']==376 and manifest['W']==1241
held=set(cfg['eval_only_frames']);core=load('a2_mapper_core',S/'scripts/splatam_external_tm.py');core.seed_everything(0)
from c04n_addition import add_new_gaussians as add
from external_tracking import map_hash
state={'births':0,'initial_hash':None,'additions':[],'loss_frames':{},'targets':[],'confidence_frames':[]}
init=core.initialize_first_timestep
def init_hook(*a,**kw):
 result=init(*a,**kw);state['births']=len(result[0]['means3D']);state['initial_hash']=map_hash(result[0]);return result
core.initialize_first_timestep=init_hook
def add_hook(*a,**kw):
 p,v=a[:2];i=int(a[4]);assert i not in held;before=len(p['means3D'])
 p,v=add(*a,**kw,core=core,mode='native',cap=3000,seed=0,log_path=OUT/'addition_history.jsonl',fb=cfg['mapping']['residual_fb'])
 n=len(p['means3D'])-before
 if n!=3000:raise RuntimeError(f'actual quota mismatch frame={i} n={n}')
 state['births']+=n;state['additions'].append({'frame':i,'births':n});return p,v
core.add_new_gaussians=add_hook;orig=core.get_loss;cache={}
def loss_hook(*a,**kw):
 p,data,v,frame=a[:4]
 if kw.get('mapping'):
  online=int(sys._getframe(1).f_locals['time_idx']);assert online not in held and int(frame) not in held
  state['loss_frames'][str(online)]=state['loss_frames'].get(str(online),0)+1
  if int(frame) not in state['targets']:state['targets'].append(int(frame))
  if ARM in ('C1','C2','C3','C4','C5'):
   data['a2_mode']=ARM;data['a2_fb']=fb
   if ARM in ('C3','C5'):
    if int(frame) not in cache:
     arr=np.load(rows[int(frame)]['confidence']);cache[int(frame)]=torch.tensor(arr,device='cuda').unsqueeze(0);state['confidence_frames'].append(int(frame))
    data['a2_confidence']=cache[int(frame)]
   else:data['a2_confidence']=data['depth']
 result=orig(*a,**kw)
 if not torch.isfinite(result[0]):raise RuntimeError('nonfinite loss')
 if time.monotonic()-start>900:raise RuntimeError('time budget')
 if torch.cuda.memory_allocated()>4*2**30:raise RuntimeError('GPU memory budget')
 return result
core.get_loss=loss_hook
try:
 core.rgbd_slam(cfg);torch.cuda.synchronize()
 with np.load(OUT/'params.npz') as z:
  n=len(z['means3D']); assert all(np.isfinite(z[k]).all() for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales'))
  assert not np.isin(z['timestep'],list(held)).any() and not np.isin(z['keyframe_time_indices'],list(held)).any()
 assert len(state['additions'])==COUNT-len(held)-1 and sum(state['loss_frames'].values())==60*(COUNT-len(held)) and all(x==60 for x in state['loss_frames'].values())
 assert not set(state['targets'])&held and not set(state['confidence_frames'])&held
 receipt={'status':'PASS_ENGINEERING','arm':ARM,'frames':COUNT,'initial_hash':state['initial_hash'],'births_total':state['births'],'survivors':n,'mapping_loss_frames':state['loss_frames'],'mapping_targets':state['targets'],'confidence_frames':state['confidence_frames'],'holdout':sorted(held),'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'map_sha256':sha(OUT/'params.npz'),'gradient_calibration_sha256':sha(cal_path),'gamma':{'C0':1.0,'C1':1.0010791602334679,'C2':0.7633236491505211,'C3':0.7633236491505211,'C4':0.7633236491505211,'C5':0.7633236491505211}[ARM],'fb':fb,'quality_adoption':False}
 (OUT/'a2_receipt.json').write_text(json.dumps(receipt,indent=2));print('A2_COMPLETE',json.dumps(receipt))
except BaseException as e:
 (OUT/'a2_receipt.json').write_text(json.dumps({'status':'FAILED_PRESERVED','arm':ARM,'error':repr(e),'traceback':traceback.format_exc(),'state':state},indent=2));raise

