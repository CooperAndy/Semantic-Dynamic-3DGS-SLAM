"""Prepared, not executed: sequential C0–C5 smoke with GPU/disk gates."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/root/autodl-tmp/A2_stereo_robust_geometry/factorial_v1');out=R/'batch200_parallel.json';assert not out.exists();env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native')
def gpu_pids():
 t=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);return [int(x.strip()) for x in t.splitlines() if x.strip().isdigit()]
def free_mib():
 t=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True);return int(t.strip().splitlines()[0])
audit=json.loads((R/'independent_audit_v2_25.json').read_text());assert audit['status']=='PASS' and len(audit['arms'])==6
evaluation=json.loads((R/'evaluation_25.json').read_text());assert evaluation['status']=='COMPLETE' and len(evaluation['checks'])==6 and len(evaluation['frames'])==1
state={'state':'RUNNING','runs':[],'quality_adoption':False}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(out)
save()
try:
 for arm in ('C0','C1','C2','C3','C4','C5'):
  free=free_mib()
  if free<6144:state.update(state='STOP_LOW_GPU_MEMORY',free_mib=free);break
  if shutil.disk_usage(R).free<5*2**30:state['state']='STOP_DISK';break
  logfile=R/f'{arm}_200_parallel.log';assert not logfile.exists() and not (R/'runs'/f'{arm}_200').exists();start=time.monotonic();rec={'arm':arm,'gpu_overlap_samples':[],'initial_other_gpu_pids':gpu_pids(),'initial_free_mib':free,'sharing_authorized':True}
  with logfile.open('xb') as f:
   p=subprocess.Popen([sys.executable,str(R/'a2_factorial_worker_c200_v1.py'),arm,'200'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);next_sample=0.
   while p.poll() is None:
    if time.monotonic()-start>900 or shutil.disk_usage(R).free<5*2**30:
     rec['stop_reason']='TIME_OR_DISK_LIMIT';p.terminate();p.wait(timeout=20);break
    if time.monotonic()>=next_sample:
     others=[i for i in gpu_pids() if i!=p.pid]
     if others:rec['gpu_overlap_samples'].append(others)
     next_sample=time.monotonic()+10
    time.sleep(2)
  receipt=R/'runs'/f'{arm}_200'/'a2_receipt.json';rec.update(exit_code=p.returncode,wall_s=time.monotonic()-start,receipt_status=json.loads(receipt.read_text()).get('status') if receipt.is_file() else 'MISSING');state['runs'].append(rec);save()
  if rec['exit_code']!=0 or rec['receipt_status']!='PASS_ENGINEERING':state['state']='STOP_FAILED_PRESERVED';break
 else:state['state']='RUN200_SIX_RUNS_COMPLETE_AUDIT_PENDING'
except BaseException as e:state.update(state='STOP_EXCEPTION',error=repr(e));raise
finally:save();print(json.dumps(state,indent=2))

if state['state'] != 'RUN200_SIX_RUNS_COMPLETE_AUDIT_PENDING':raise SystemExit(1)
