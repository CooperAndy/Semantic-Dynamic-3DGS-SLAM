from pathlib import Path
import os,subprocess,shutil,json,hashlib,sys
R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012');W=R/'stage_a/cuda_sm120';T=Path('/root/autodl-tmp/dg_slam_chy_20261011/07_formal_tracking_T14_20261012/native')
cache=(T/'faiss/build_sm120/CMakeCache.txt').read_text();assert 'CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120' in cache
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lib=W/'lib'
assert sha(lib/'faiss/_swigfaiss.so')==sha(T/'faiss_python/faiss/_swigfaiss.so')
assert sha(W/'runtime_libs/libstdc++.so.6')==sha(T/'runtime_libs/libstdc++.so.6')
env=os.environ.copy();env.update(LD_PRELOAD=str(W/'runtime_libs/libstdc++.so.6'),PYTHONPATH=str(lib)+':'+str(R/'scripts'),LD_LIBRARY_PATH=str(W/'runtime_libs')+':'+env.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='4',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',CUDA_HOME='/usr/local/cuda')
receipt={'status':'FAISS_PROBE','scope':'T04 own experiment; only compatibility binary reused from existing T14 build; no research results reused','faiss_origin':str(T/'faiss_python/faiss'),'faiss_sha256':sha(lib/'faiss/_swigfaiss.so'),'cmake_cache_sha256':sha(T/'faiss/build_sm120/CMakeCache.txt'),'steps':[]}
def save(): (W/'resume_receipt.json').write_text(json.dumps(receipt,indent=2))
save()
try:
 probe="import torch,faiss,numpy as np;assert faiss.__version__=='1.10.0';r=faiss.StandardGpuResources();i=faiss.index_cpu_to_gpu(r,0,faiss.IndexFlatL2(3));x=np.array([[0,0,0],[1,0,0],[5,0,0]],dtype='float32');i.add(x);d,j=i.search(x,1);assert (j[:,0]==np.arange(3)).all();assert (d==0).all();print('FAISS_SM120_PROBE_PASS',faiss.__file__)"
 subprocess.run([sys.executable,'-c',probe],env=env,check=True);receipt['steps'].append('faiss_gpu_probe_pass');save()
 for phase,frames,name in [('smoke40',40,'KITTI00_200_A_smoke40_20261012_03'),('full200',200,'KITTI00_200_A_on_20261012_01')]:
  if phase=='full200':
   out=R/'04_reproduction/KITTI00_200_A_smoke40_20261012_03';meta=json.loads((out/'run_manifest.json').read_text());assert meta['status']=='COMPLETED' and meta['counters']['input_frames']==40
   hist=[json.loads(x) for x in (out/'map_range_history.jsonl').read_text().splitlines()]
   assert hist and all(x['xyz_finite'] and x['scale_finite'] and x['max_distance_current_camera_m']<1000 and x['max_activated_scale_m']<100 for x in hist), 'SMOKE_MAP_RANGE_REQUIRES_REVIEW'
  receipt.update(status=phase.upper()+'_RUNNING',run_name=name);save()
  with (R/'stage_a'/(phase+'_ready.log')).open('w') as log:
   done=subprocess.run([sys.executable,str(R/'scripts/run_stage_a.py'),'--run-name',name,'--frames',str(frames),'--gs-refine','on'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  (R/'stage_a'/(phase+'_ready.exit')).write_text(str(done.returncode)+'\n');assert done.returncode==0,(phase,done.returncode)
  receipt['steps'].append(phase+'_completed');save()
 receipt['status']='COMPLETED';save()
except Exception as e:
 receipt.update(status='FAILED',error=repr(e));save();raise
