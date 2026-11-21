from pathlib import Path
import hashlib,json,os,subprocess,time
F=Path(__file__).resolve().parents[1];P=json.loads((F/'repair_protocol.json').read_text());O=P['optimizer_recipe']
PY='/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/venv_dg/bin/python';env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1';env['LD_PRELOAD']='/root/autodl-tmp/dg_slam_chy_20261011/07_formal_tracking_T14_20261012/native/runtime_libs/libstdc++.so.6'
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=F/'receipts/repair_launch.json';assert not receipt.exists();r={'status':'NUMERIC_RUNNING','started_unix':time.time(),'workers':[]};dump(receipt,r)
def worker(stage,frame=None,arm=None):
 name='repair_numerical' if stage=='numeric' else f'replay_f{frame}_{arm}'
 cmd=[PY,str(F/'scripts/repaired_runtime.py'),stage]
 if frame is not None:cmd+=['--frame',str(frame),'--arm',arm]
 with (F/'receipts'/f'{name}.log').open('w') as log:
  try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:code=124
 r['workers'].append({'stage':stage,'frame':frame,'arm':arm,'exit_code':code,'ended_unix':time.time()});dump(receipt,r);return code
code=worker('numeric')
if code:r['status']='NUMERIC_EXECUTION_FAILED';dump(receipt,r);raise SystemExit(code)
gate=json.loads((F/'receipts/repaired_numerical_gate.json').read_text())
if gate['status']!='PASS':r['status']='NEW_NUMERICAL_GATE_FAIL_NO_REPLAYS';dump(receipt,r);raise SystemExit(0)
r['status']='REPLAYS_RUNNING';dump(receipt,r)
for frame in P['frames']:
 for arm in O['arms']:
  code=worker('replay',frame,arm)
  if code:r['status']='REPLAY_EXECUTION_FAILED';dump(receipt,r);raise SystemExit(code)
r.update(status='EIGHT_REPLAYS_COMPLETED',completed_unix=time.time());dump(receipt,r)
files={str(p.relative_to(F)):sha(p) for p in (F/'replays').rglob('*') if p.is_file()}
dump(F/'receipts/replay_raw_seal.json',{'status':'SEALED_BEFORE_CPU_GT','files':files,'sealed_unix':time.time(),'protocol_sha256':sha(F/'repair_protocol.json')})
