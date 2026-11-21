import protect_parent;protect_parent.install()
from common import *
import sys,os,time,traceback,importlib.util,signal,subprocess
import numpy as np,torch
started=time.monotonic();variant=sys.argv[1];reg=json.loads((R/'registration.json').read_text());assert variant in reg['variants'];out=R/'runs'/variant;out.mkdir();__import__('source_snapshot').capture(R,out);S=R/'source';sys.path.insert(0,str(S));os.chdir(S);blocked=[];receipt=dict(variant=variant,status='RUNNING',quality_adoption=False)
def guard(e,a):
 if e=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(a[0])
  if '/poses/' in p or p.endswith('groundtruth.txt') or '/evaluation_only/' in p:blocked.append(p);raise RuntimeError('GT blocked in prediction')
sys.addaudithook(guard)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def stop(signum,frame):raise RuntimeError('Registered per-run budget stop')
signal.signal(signal.SIGTERM,stop)
try:
 torch.set_num_threads(4);cfg=load('g18d_config',S/'configs/kitti'/f'{variant}.py').config;N=cfg['data']['num_frames'];core=load('g18d_core',S/'scripts/splatam_external_tm.py');import stable_birth;stable_birth.install(core);receipt['stable_birth_sha256']=sha(R/'scripts/stable_birth.py');core.seed_everything(cfg['seed']);manifest=json.loads((R/'inputs/runtime/input_manifest.json').read_text());import semantic_runtime;from semantic_constraints import Flags
 sf=cfg['semantic'];provider=__import__('semantic_bridge').Bridge(sf['manifest'],manifest,'dynamic_filter' if sf['D'] else 'semantic_only')
 semantic=semantic_runtime.install(core,Flags(sf['T'],sf['M'],sf['D']),provider,out,cfg['eval_only_frames'],labels=sf.get('labels',True),shuffle=sf.get('shuffle',False),shuffle_seed=cfg['seed'])
 from runtime_timing import RenderTimer;render_timer=RenderTimer();calls=[0];Native=core.Renderer
 class Renderer:
  def __init__(self,*a,**k):self.native=Native(*a,**k)
  def __call__(self,*a,**k):calls[0]+=1;return render_timer.call(self.native,*a,**k)
  def __getattr__(self,n):return getattr(self.native,n)
 core.Renderer=Renderer;import external_tracking as ext;import stable_pose;stable_pose.install(ext);receipt['stable_pose_sha256']=sha(R/'scripts/stable_pose.py');from external_tracking import map_hash,camera
 init=core.initialize_first_timestep
 def initialize(*a,**k):
  result=init(*a,**k);receipt['initial_gaussian_sha256']=map_hash(result[0]);receipt['initial_points']=len(result[0]['means3D']);return result
 core.initialize_first_timestep=initialize
 import backend as backend_module;import stereo_proposal;stereo_proposal.install(backend_module,manifest);from backend import Backend
 backend=Backend(core,cfg,out,manifest);backend.install();from factor_hooks import install
 observation,finish=install(core,cfg,out,backend);dataset_fn=core.get_dataset;access=[]
 class Dataset:
  def __init__(self,ds):self.ds=ds;self.maximum=-1
  def __len__(self):return len(self.ds)
  def __getattr__(self,n):return getattr(self.ds,n)
  def __getitem__(self,index):
   assert 0<=index<N and index<=self.maximum+1,('Noncausal frame access',index,self.maximum);self.maximum=max(self.maximum,index);access.append(index);return self.ds[index]
 core.get_dataset=lambda *a,**k:Dataset(dataset_fn(*a,**k))
 shared=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15);dump(out/'shared_load_start.json',dict(at=now(),pid=os.getpid(),compute_apps=shared.stdout,stderr=shared.stderr));algorithm=time.monotonic();core.rgbd_slam(cfg);receipt['algorithm_seconds']=time.monotonic()-algorithm;finish();backend.finish();semantic.close();torch.cuda.synchronize();receipt['render_timing']=render_timer.flush()
 with np.load(out/'estimated_trajectory.npz') as z:poses=z['w2c'];assert poses.shape==(N,4,4) and np.isfinite(poses).all()
 assert len(backend.records)==N and [x['frame'] for x in backend.records]==list(range(N));assert observation['mapping_steps']==(N-len(cfg['eval_only_frames']))*60==observation['mapping_backwards'];assert observation['camera_Adam']==0
 with np.load(out/'params.npz') as z:
  assert not np.isin(z['timestep'],cfg['eval_only_frames']).any() and not np.isin(z['keyframe_time_indices'],cfg['eval_only_frames']).any();receipt['final_gaussians']=len(z['means3D']);assert np.array_equal(z['cam_unnorm_rots'][...,:N].shape,np.array([1,4,N]))
 for row,T in zip(backend.records,poses):assert np.array_equal(T,np.asarray(row['selected_w2c'],np.float32))
 receipt.update(status='PASS_NATIVE_MAPPING_INTEGRATION',frames=N,accepted=sum(x['accepted'] for x in backend.records),mapping_observation=observation,pose_checks=backend.pose_checks,dataset_access=access,tracking_native_forwards=backend.render_calls,native_forwards=calls[0],GT_blocked_opens=len(blocked),camera_backward=0,mapping_backward=observation['mapping_backwards'])
except BaseException as error:
 receipt.update(status='FAILED_PRESERVED',error=repr(error),traceback=traceback.format_exc())
 try:
  tb=error.__traceback__
  while tb:
   f=tb.tb_frame
   if f.f_code.co_name=='rgbd_slam' and 'params' in f.f_locals:
    core.save_params_ckpt(f.f_locals['params'],str(out),max(int(f.f_locals.get('time_idx',0)),0));receipt['partial_frame']=int(f.f_locals.get('time_idx',0));break
   tb=tb.tb_next
 except Exception as err:receipt['preserve_error']=repr(err)
 raise
finally:
 receipt.update(at=now(),wall_seconds=time.monotonic()-started,GT_blocked_opens=len(blocked));receipt['peak_torch_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0;dump(out/'receipt.json',receipt)
