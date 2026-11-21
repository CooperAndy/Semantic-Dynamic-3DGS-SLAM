"""A2 mapper DEV run; isolated source; actual 3000 births per append frame."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');ROOT=O/'mapper_v1';S=ROOT/'source';BASE=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200')
ARM=sys.argv[1];COUNT=int(sys.argv[2]);assert ARM in ('B0','B1','B2','B3','B4') and COUNT in (25,200)
OUT=ROOT/'runs'/f'{ARM}_{COUNT}';OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();sys.path.insert(0,str(O));sys.path.insert(0,str(S));os.chdir(S);torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
registration=json.loads((O/'A2_run_registration_20261028.json').read_text())
for entry in registration['file_locks']:
 if sha(entry['path'])!=entry['sha256']:raise RuntimeError('registered file changed: '+entry['path'])
cal_path=O/'gradient_calibration_20261028.json';cal=json.loads(cal_path.read_text());assert cal['status']=='PASS_GRADIENT_SCALE_MATCH_ONLY'
manifest=json.loads((BASE/'inputs/runtime/input_manifest.json').read_text());rows={r['frame_id']:r for r in manifest['rows']}
if sha(manifest['pose_file'])!=manifest['pose_sha256']:raise RuntimeError('pose mismatch')
if sha('/root/autodl-tmp/stage3_g05B_20261025/input_view/calib.txt')!=cal['stereo_08']['calib_sha256']:raise RuntimeError('calibration changed')
for i in range(COUNT):
 for k in ('rgb','depth','confidence'):
  if sha(rows[i][k])!=rows[i]['hashes'][k]:raise RuntimeError(f'input mismatch {i} {k}')
cfg=load('a2_config',BASE/'source/configs/kitti/G00_S0.py').config;cfg['workdir']=str(ROOT/'runs');cfg['run_name']=OUT.name;cfg['data']['num_frames']=COUNT;cfg['eval_only_frames']=[i for i in cfg['eval_only_frames'] if i<COUNT]
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
 p,v=add(*a,**kw,core=core,mode='disp_valid' if ARM=='B4' else 'native',cap=3000,seed=0,log_path=OUT/'addition_history.jsonl',fb=cfg['mapping']['residual_fb'])
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
  if ARM in ('B1','B2','B3'):
   data['a2_mode']={'B1':'fixed','B2':'distance','B3':'confidence'}[ARM];data['a2_fb']=cal['stereo_08']['fb_px_m'];data['a2_gamma']=cal['gamma'][ARM]
   if ARM=='B3':
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
 receipt={'status':'PASS_ENGINEERING','arm':ARM,'frames':COUNT,'initial_hash':state['initial_hash'],'births_total':state['births'],'survivors':n,'mapping_loss_frames':state['loss_frames'],'mapping_targets':state['targets'],'confidence_frames':state['confidence_frames'],'holdout':sorted(held),'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'map_sha256':sha(OUT/'params.npz'),'gradient_calibration_sha256':sha(cal_path),'gamma':cal['gamma'][ARM],'fb':cal['stereo_08']['fb_px_m'],'quality_adoption':False}
 (OUT/'a2_receipt.json').write_text(json.dumps(receipt,indent=2));print('A2_COMPLETE',json.dumps(receipt))
except BaseException as e:
 (OUT/'a2_receipt.json').write_text(json.dumps({'status':'FAILED_PRESERVED','arm':ARM,'error':repr(e),'traceback':traceback.format_exc(),'state':state},indent=2));raise
