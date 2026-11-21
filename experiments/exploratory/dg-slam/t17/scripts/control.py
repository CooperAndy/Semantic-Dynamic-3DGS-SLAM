from pathlib import Path
import argparse,csv,json,stat,sys
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1];P=json.loads((F/'protocol.json').read_text(encoding='utf-8'));R=P['remote_root']
sys.path.insert(0,str(F.parent/'CHY_T16_渲染分支诊断与修复_20261012/receipts'))
from resume import connect
a=argparse.ArgumentParser();a.add_argument('mode',choices=['deploy','smoke','status','collect','freeze','run','evaluate']);v=a.parse_args()
with connect() as c:
 def remote(payload):
  i,o,e=c.exec_command('/root/miniconda3/bin/python -',timeout=60);i.write(payload);i.channel.shutdown_write();out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status();print(out);assert code==0,err
 if v.mode=='deploy':
  rows=list(csv.DictReader((F/'inputs/DEV200_runtime.csv').open(encoding='utf-8')))
  with (F/'inputs/SMOKE200_runtime.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows[:60])
  remote("from pathlib import Path\nimport subprocess,shutil\nf=Path("+repr(R)+")\nassert not f.exists()\nassert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()\nassert shutil.disk_usage(f.parent).free>5*1024**3\nfor name in ['scripts','inputs','receipts']: (f/name).mkdir(parents=True,exist_ok=True)\nprint('GPU_IDLE_NEW_ROOT_CREATED')")
  with c.open_sftp() as s:
   for p in [F/'protocol.json',F/'T17_完整轨迹对照预登记.md',*list((F/'scripts').glob('*.py')),*list((F/'inputs').glob('*.csv'))]:s.put(str(p),R+'/'+p.relative_to(F).as_posix())
  remote("from pathlib import Path\nimport hashlib,json,datetime\nf=Path("+repr(R)+")\nout={'registered_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':hashlib.sha256((f/'protocol.json').read_bytes()).hexdigest(),'before_any_T17_GPU_worker':True}\n(f/'receipts/design_registration.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))")
 elif v.mode in ['smoke','run']:
  phase='smoke' if v.mode=='smoke' else 'registered'
  remote("from pathlib import Path\nimport subprocess,os\nf=Path("+repr(R)+")\nenv=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'\nassert not (f/'receipts/"+phase+"_launch.json').exists()\nwith (f/'receipts/"+phase+"_launcher.log').open('w') as log:\n p=subprocess.Popen(['/root/miniconda3/bin/python',str(f/'scripts/launch.py'),"+repr(phase)+"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)\n(f/'receipts/"+phase+".pid').write_text(str(p.pid));print('LAUNCHED',p.pid)")
 elif v.mode in ['freeze','evaluate']:
  script='freeze.py' if v.mode=='freeze' else 'evaluate.py'
  remote("import subprocess,os\nenv=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''\nr=subprocess.run(['/root/miniconda3/bin/python',"+repr(R+'/scripts/'+script)+"],env=env);assert r.returncode==0")
 elif v.mode=='status':
  remote("from pathlib import Path\nimport json,subprocess\nf=Path("+repr(R)+")\nfor phase in ['smoke','registered']:\n p=f/f'receipts/{phase}_launch.json'\n if p.exists():\n  r=json.loads(p.read_text());print(phase,r['status'],len(r['runs']),[(x['arm'],x['exit_code']) for x in r['runs']])\n  logs=sorted((f/'receipts').glob(('SMOKE200' if phase=='smoke' else 'DEV200')+'*.log'),key=lambda p:p.stat().st_mtime)\n  if logs: print(logs[-1].name,logs[-1].read_text()[-1200:])\nprint(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True))")
 else:
  with c.open_sftp() as s:
   def walk(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    for x in s.listdir_attr(src):
     if stat.S_ISDIR(x.st_mode):walk(src+'/'+x.filename,dst/x.filename)
     else:s.get(src+'/'+x.filename,str(dst/x.filename))
   for name in ['receipts','smoke','runs']:
    try:walk(R+'/'+name,F/name)
    except FileNotFoundError:pass
   for name in ['execution_lock.json']:
    try:s.get(R+'/'+name,str(F/name))
    except FileNotFoundError:pass
  print('COLLECTED')
