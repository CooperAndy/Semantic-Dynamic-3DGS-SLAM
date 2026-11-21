"""Run numerical gate, then exactly eight conditional replays, never formal tracking."""
from pathlib import Path
import hashlib,json,os,subprocess,time
F=Path(__file__).resolve().parents[1]
P=json.loads((F/'protocol.json').read_text());T14=Path(P['t14_root'])
PY='/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/venv_dg/bin/python'
env=os.environ.copy();env['LD_PRELOAD']=str(T14/'native/runtime_libs/libstdc++.so.6');env['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
assert not (F/'receipts/launch.json').exists()
r={'status':'NUMERIC_RUNNING','started_unix':time.time(),'workers':[]};dump(F/'receipts/launch.json',r)
def worker(stage,frame=None,arm=None):
 name='numeric' if stage=='numeric' else f'f{frame}_{arm}'
 cmd=[PY,str(F/'scripts/replay_runtime.py'),stage]
 if frame is not None:cmd+=['--frame',str(frame),'--arm',arm]
 with (F/'receipts'/f'{name}.log').open('w') as log:
  try:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
  except subprocess.TimeoutExpired:code=124
 r['workers'].append({'stage':stage,'frame':frame,'arm':arm,'exit_code':code,'ended_unix':time.time()});dump(F/'receipts/launch.json',r)
 return code
code=worker('numeric')
if code:r.update(status='NUMERIC_EXECUTION_FAILED');dump(F/'receipts/launch.json',r);raise SystemExit(code)
gate=json.loads((F/'receipts/numerical_gate.json').read_text())
if gate['status']!='PASS':
 r.update(status='NUMERIC_GATE_FAILED_REPLAYS_NOT_STARTED',completed_unix=time.time());dump(F/'receipts/launch.json',r);raise SystemExit(0)
r['status']='REPLAYS_RUNNING';dump(F/'receipts/launch.json',r)
for case in P['cases']:
 for arm in P['arms']:
  code=worker('replay',case['frame'],arm)
  if code:r['status']='REPLAY_EXECUTION_FAILED';dump(F/'receipts/launch.json',r);raise SystemExit(code)
r.update(status='EIGHT_REPLAYS_COMPLETED',completed_unix=time.time());dump(F/'receipts/launch.json',r)
files={str(p.relative_to(F)):sha(p) for p in (F/'runs').rglob('*') if p.is_file()}
dump(F/'receipts/raw_replay_seal.json',{'status':'SEALED_BEFORE_CPU_GT_SCORING','files':files,'sealed_unix':time.time(),'protocol_sha256':sha(F/'protocol.json')})
