"""Sequential A2 DEV runner with GPU sharing and disk/time monitoring."""
import datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';OUT=R/'batch_200.json'
assert not OUT.exists()
assert json.loads((R/'independent_audit_v2_25.json').read_text())['status']=='PASS'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def dump(x):
 t=OUT.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(OUT)
def pids():
 output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 return [int(x.strip()) for x in output.splitlines() if x.strip().isdigit()]
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native')
status={'state':'RUNNING','started':now(),'runs':[],'quality_adoption':False};dump(status);p=None
try:
 for arm in ('B0','B1','B2','B3','B4'):
  if shutil.disk_usage(R).free<5*2**30:status['state']='STOP_DISK';break
  occupied=pids()
  if occupied:status.update(state='STOP_GPU_BUSY',busy_pids=occupied);break
  out=R/f'{arm}_200.log';assert not out.exists() and not (R/'runs'/f'{arm}_200').exists()
  start=time.monotonic();record={'arm':arm,'started':now(),'gpu_overlap_samples':[]};status['active']=arm;dump(status)
  with out.open('xb') as stream:
   p=subprocess.Popen([sys.executable,str(O/'a2_mapper_worker.py'),arm,'200'],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   record['pid']=p.pid;next_sample=0.
   while p.poll() is None:
    if time.monotonic()-start>900 or shutil.disk_usage(R).free<5*2**30:
     record['stop_reason']='TIME_OR_DISK_LIMIT';p.terminate();p.wait(timeout=20);break
    if time.monotonic()>=next_sample:
     others=[pid for pid in pids() if pid!=p.pid]
     if others:record['gpu_overlap_samples'].append({'at':now(),'pids':others})
     next_sample=time.monotonic()+10
    time.sleep(2)
  record.update(exit_code=p.returncode,finished=now(),wall_s=time.monotonic()-start)
  receipt=R/'runs'/f'{arm}_200'/'a2_receipt.json'
  if receipt.is_file():record['receipt_status']=json.loads(receipt.read_text()).get('status')
  else:record['receipt_status']='MISSING'
  status['runs'].append(record);status.pop('active',None);dump(status)
  if p.returncode!=0 or record['receipt_status']!='PASS_ENGINEERING':status['state']='STOP_FAILED_PRESERVED';break
 else:status['state']='ALL_5_COMPLETE'
except BaseException as e:
 if p is not None and p.poll() is None:
  p.terminate();p.wait(timeout=20)
 status.update(state='STOP_EXCEPTION',error=repr(e));raise
finally:
 status['finished']=now();dump(status);print(json.dumps(status,indent=2))
