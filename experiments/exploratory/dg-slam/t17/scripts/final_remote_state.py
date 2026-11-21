from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1];R=json.loads((F/'protocol.json').read_text(encoding='utf-8'))['remote_root']
sys.path.insert(0,str(F.parent/'CHY_T16_渲染分支诊断与修复_20261012/receipts'))
from resume import connect
payload='''from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
f=Path(ROOT)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
launch=json.loads((f/'receipts/registered_launch.json').read_text());assert launch['status']=='ALL_SIX_TERMINAL' and len(launch['runs'])==6
L=json.loads((f/'execution_lock.json').read_text());assert sha(f/'protocol.json')==L['protocol_sha256']
for path,h in L['runtime_immutable_files'].items():assert sha(path)==h,path
for rel,h in json.loads((f/'receipts/raw_seal.json').read_text())['files'].items():assert sha(f/rel)==h,rel
workers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not workers,workers
out={'status':'COMPLETED_IDLE','checked_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registered_runs':6,'exit_codes':[r['exit_code'] for r in launch['runs']],'decision':json.loads((f/'receipts/acceptance.json').read_text())['decision'],'runtime_hashes_verified':len(L['runtime_immutable_files']),'GPU_workers':workers,'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'disk_free_bytes':shutil.disk_usage(f).free,'server_shutdown':False,'formal_runs':0}
(f/'receipts/server_final_state.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''.replace('ROOT',repr(R))
with connect() as c:
 i,o,e=c.exec_command('/root/miniconda3/bin/python -',timeout=60);i.write(payload);i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err;print(out)
