from pathlib import Path
import shutil, json, hashlib, subprocess, os, sys
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012'); B=Path('/root/autodl-tmp/dg_slam_chy_20261011/01_source_env'); W=R/'stage_a/cuda_sm120';W.mkdir(exist_ok=False)
env=os.environ.copy();env.update(TORCH_CUDA_ARCH_LIST='12.0',CUDA_HOME='/usr/local/cuda',MAX_JOBS='2',OMP_NUM_THREADS='4',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1');env['PATH']='/usr/local/cuda/bin:'+env['PATH']
import torch
assert torch.cuda.get_device_capability()==(12,0)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
assert shutil.disk_usage(R).free>4*1024**3
lib=W/'lib';lib.mkdir();site=B/'venv_dg/lib/python3.12/site-packages'
for package in ['lietorch','diff_gaussian_rasterization_depth_pose']:
 shutil.copytree(site/package,lib/package,ignore=shutil.ignore_patterns('__pycache__','*.so'))
parents=[('droid',B/'dependencies/DROID-backend_dg_api'),('lie',B/'dependencies/DROID-SLAM_runtime/thirdparty/lietorch'),('raster',B/'DG-SLAM/submodules/diff-gaussian-rasterization_pose')]
receipt={'status':'BUILDING','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'arch':'12.0','shared_environment_modified':False,'parents':{},'steps':[]}
def save(): (W/'receipt.json').write_text(json.dumps(receipt,indent=2))
save()
try:
 for name,parent in parents:
  target=W/name
  shutil.copytree(parent,target,symlinks=True,ignore=shutil.ignore_patterns('.git','build','dist','*.egg-info','__pycache__','*.so'))
  receipt['parents'][name]={'path':str(parent),'setup_sha256':hashlib.sha256((parent/'setup.py').read_bytes()).hexdigest()};save()
  with (W/(name+'_build.log')).open('w') as log:
   subprocess.run([sys.executable,'setup.py','build_ext','--build-lib',str(lib),'--build-temp',str(W/(name+'_temp'))],cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  receipt['steps'].append(name+'_compiled');save()
 env['PYTHONPATH']=str(lib)+':'+str(R/'scripts')
 subprocess.run([sys.executable,'-c',"import torch,droid_backends,lietorch,diff_gaussian_rasterization_depth_pose;print(droid_backends.__file__);print(lietorch.SE3.Identity(1,device='cuda').matrix().cpu())"],env=env,check=True)
 receipt['extensions']={str(f.relative_to(lib)):hashlib.sha256(f.read_bytes()).hexdigest() for f in lib.rglob('*.so')};receipt['status']='BUILT';save()
 for phase,frames,name in [('smoke40',40,'KITTI00_200_A_smoke40_20261012_02'),('full200',200,'KITTI00_200_A_on_20261012_01')]:
  if phase=='full200':
   output=R/'04_reproduction/KITTI00_200_A_smoke40_20261012_02'
   meta=json.loads((output/'run_manifest.json').read_text());assert meta['status']=='COMPLETED' and meta['counters']['input_frames']==40
   hist=[json.loads(x) for x in (output/'map_range_history.jsonl').read_text().splitlines()]
   assert hist and all(x['xyz_finite'] and x['scale_finite'] and x['max_distance_current_camera_m']<1000 and x['max_activated_scale_m']<100 for x in hist), 'SMOKE_MAP_RANGE_REQUIRES_REVIEW'
  receipt['status']=phase.upper()+'_RUNNING';receipt['run_name']=name;save()
  with (R/'stage_a'/(phase+'_sm120.log')).open('w') as log:
   done=subprocess.run([sys.executable,str(R/'scripts/run_stage_a.py'),'--run-name',name,'--frames',str(frames),'--gs-refine','on'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  (R/'stage_a'/(phase+'_sm120.exit')).write_text(str(done.returncode)+'\n');assert done.returncode==0,(phase,done.returncode)
  receipt['steps'].append(phase+'_completed');save()
 receipt['status']='COMPLETED';save()
except Exception as e:
 receipt.update(status='FAILED',error=repr(e));save();raise
