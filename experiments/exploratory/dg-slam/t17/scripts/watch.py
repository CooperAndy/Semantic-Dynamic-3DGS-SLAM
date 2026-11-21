from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1];P=json.loads((F/'protocol.json').read_text(encoding='utf-8'));R=P['remote_root']
sys.path.insert(0,str(F.parent/'CHY_T16_渲染分支诊断与修复_20261012/receipts'))
from resume import connect
payload='''from pathlib import Path
import json,subprocess
f=Path(ROOT);r=json.loads((f/'receipts/registered_launch.json').read_text())
done=[{'seed':x['seed'],'arm':x['arm'],'exit_code':x['exit_code']} for x in r['runs']]
logs=sorted((f/'receipts').glob('DEV200*.log'),key=lambda p:p.stat().st_mtime)
active=None
if logs:
 name=logs[-1].stem;p=f/'runs'/name/'run_manifest.json'
 if p.exists():
  try:
   m=json.loads(p.read_text());active={k:m.get(k) for k in ['seed','arm','status','seconds','counts','error']}
  except json.JSONDecodeError:active={'manifest':'being updated'}
print(json.dumps({'status':r['status'],'completed':done,'active':active,'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()}))
'''.replace('ROOT',repr(R))
with connect() as c:
 i,o,e=c.exec_command('/root/miniconda3/bin/python -',timeout=30);i.write(payload);i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err;print(out)
