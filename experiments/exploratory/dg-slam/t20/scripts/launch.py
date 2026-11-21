"""Detached finite schedule; stop on engineering failure, no quality-based retries."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
F=Path(__file__).resolve().parents[1];P=json.loads((F/'protocol.json').read_text())
BASE=F.parent/'07_formal_tracking_T14_20261012'
PY=str(F.parent/'01_source_env/venv_dg/bin/python')
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
env['LD_PRELOAD']=str(BASE/'native/runtime_libs/libstdc++.so.6')
env['WANDB_MODE']='disabled'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
dest=F/'receipts/launch.json';assert not dest.exists()
receipt={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':[],'formal':False}
dump(dest,receipt)
schedule=[('native_gpu_preflight',['native_gpu_check.py'],180),
          ('candidate_gpu_smoke',['candidate_gpu_smoke.py'],180),
          ('DEV200_s2718_NO_GS_REFINE',['run_tracking.py','--window','DEV200','--seed','2718','--arm','NO_GS_REFINE'],1200),
          ('DEV200_s2718_LOCAL_ID',['run_tracking.py','--window','DEV200','--seed','2718','--arm','LOCAL_ID'],1200)]
try:
    lock=json.loads((F/'execution_lock.json').read_text())
    assert sha(F/'protocol.json')==lock['protocol_sha256']
    for path,h in lock['runtime_immutable_files'].items():assert sha(Path(path))==h,path
    for name,argv,limit in schedule:
        assert shutil.disk_usage(F).free>P['budgets']['minimum_free_bytes']
        receipt['active_stage']=name;dump(dest,receipt)
        with (F/f'receipts/{name}.log').open('xb') as log:
            try:code=subprocess.run([PY,str(F/'scripts'/argv[0]),*argv[1:]],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=limit).returncode
            except subprocess.TimeoutExpired:code=124
        receipt['stages'].append({'stage':name,'exit_code':code,'completed_unix':time.time()});dump(dest,receipt)
        if code:raise RuntimeError(f'Engineering stage {name} failed: {code}; no automatic retry')
    assert json.loads((F/'receipts/candidate_gpu_smoke.json').read_text())['status']=='PASS'
    all_cases=[]
    for arm in P['arms']:
        gate=json.loads((F/f'runs/DEV200_s2718_{arm}/diagnostic_gate.json').read_text())
        assert gate['status']=='PASS'
        all_cases.extend(c['case'] for c in gate['cases'])
    assert sorted(all_cases)==sorted(c['case'] for c in P['future_filler_cases'])
    receipt.update(status='ALL_FIVE_DIAGNOSTICS_COMPLETE',cases=all_cases,completed_unix=time.time())
    dump(dest,receipt)
    sealed={}
    for part in ['runs','smoke']:
        for p in (F/part).rglob('*'):
            if p.is_file():sealed[p.relative_to(F).as_posix()]=sha(p)
    dump(F/'receipts/raw_seal.json',{'status':'SEALED_BEFORE_CPU_GT','files':sealed,'sealed_unix':time.time(),
         'protocol_sha256':sha(F/'protocol.json'),'execution_lock_sha256':sha(F/'execution_lock.json')})
except Exception as e:
    receipt.update(status='ENGINEERING_STOP',error=repr(e),stopped_unix=time.time());dump(dest,receipt)
    raise
