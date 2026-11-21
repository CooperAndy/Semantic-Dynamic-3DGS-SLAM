from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
F=Path(__file__).resolve().parents[1];P=json.loads((F/'protocol.json').read_text());BASE=F.parent/'07_formal_tracking_T14_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
a=argparse.ArgumentParser();a.add_argument('phase',choices=['smoke','registered']);v=a.parse_args();smoke=v.phase=='smoke'
out=F/f'receipts/{v.phase}_launch.json';assert not out.exists()
if not smoke:
 assert (F/'execution_lock.json').exists()
 assert json.loads((F/'receipts/smoke_audit.json').read_text())['status']=='PASS'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['LD_PRELOAD']=str(BASE/'native/runtime_libs/libstdc++.so.6')
PY=str(F.parent/'01_source_env/venv_dg/bin/python')
schedule=[]
for k,seed in enumerate([1219] if smoke else P['seeds']):
 arms=P['arms'][k%2:]+P['arms'][:k%2]
 for arm in arms:schedule.append(('SMOKE200' if smoke else 'DEV200',seed,arm))
receipt={'status':'RUNNING','runs':[],'started_unix':time.time()};dump(out,receipt)
for window,seed,arm in schedule:
 assert shutil.disk_usage(F).free>P['budgets']['minimum_free_bytes']
 name=f'{window}_s{seed}_{arm}'
 with (F/f'receipts/{name}.log').open('w') as log:
  try:code=subprocess.run([PY,str(F/'scripts/run_tracking.py'),'--window',window,'--seed',str(seed),'--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
  except subprocess.TimeoutExpired:code=124
 receipt['runs'].append({'window':window,'seed':seed,'arm':arm,'exit_code':code,'folder':str(F/('smoke' if smoke else 'runs')/name),'completed_unix':time.time()});dump(out,receipt)
 if smoke and code:receipt['status']='SMOKE_FAILED';dump(out,receipt);raise SystemExit(code)
receipt['status']='ALL_SMOKES_COMPLETE' if smoke else 'ALL_SIX_TERMINAL';receipt['completed_unix']=time.time();dump(out,receipt)
if not smoke:
 files={p.relative_to(F).as_posix():sha(p) for p in (F/'runs').rglob('*') if p.is_file()}
 dump(F/'receipts/raw_seal.json',{'status':'SEALED_BEFORE_CPU_GT','sealed_unix':time.time(),'files':files,'protocol_sha256':sha(F/'protocol.json'),'execution_lock_sha256':sha(F/'execution_lock.json')})
