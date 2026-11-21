"""Register and transport an isolated trace, then inspect/download its outputs."""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,ast
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F.parent/'CHY_接手DGSLAM_T11起_20261012/scripts'))
from server_io import connect
R='/root/autodl-tmp/dg_slam_chy_20261011/09_T16_branch_repair_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(c,cmd,timeout=40):
 i,o,e=c.exec_command(cmd,timeout=timeout);text=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
 if text:print(text,end='')
 if err:print(err,file=sys.stderr,end='')
 assert code==0,(code,cmd)
 return text
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','deploy','launch','status','collect']);v=a.parse_args()
 if v.stage=='register':
  for d in ['receipts','traces']:(F/d).mkdir(exist_ok=True)
  # traces is created by the worker to enforce one-shot execution; local empty directory is harmless.
  p={'scope':'T16 DEV-only forward branch tracing; preserve sealed T15 FAIL and T14 negative result',
   'T15_root':'/root/autodl-tmp/dg_slam_chy_20261011/08_T15_DEV_replay_20261012','remote_root':R,
   'frames':[47,95],'axes':[3,4],
   'frame47_positions':[-.004,-.002,-.001,0,.001,.002,.0035,.00375,.004],
   'control_positions':[-.004,-.002,-.001,0,.001,.002,.004],
   'maximum_captures':32,'algorithm_changes':False,'capture':'Native geometry/binning/image buffers plus actual rendered RGB/depth/opacity and residual signs; no native recompilation, no optimization, no GT.',
   'followup':'Attribute known small-angle RGB slope change using independent CPU compositor and fixed-order counterfactual. Register repaired numerical validation before any eight-arm optimization.'}
  assert not (F/'trace_protocol.json').exists()
  (F/'trace_protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
  for x in (F/'scripts').glob('*.py'):ast.parse(x.read_text(encoding='utf-8'))
  lock={'protocol_sha256':sha(F/'trace_protocol.json'),'script_sha256':sha(F/'scripts/trace_runtime.py'),
    'T15_protocol_sha256':sha(F.parent/'CHY_T15_DEV同图回放_20261012/protocol.json'),
    'T15_delivery_manifest_sha256':sha(F.parent/'CHY_T15_DEV同图回放_20261012/delivery_manifest.json')}
  (F/'trace_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8');print('REGISTERED',lock['protocol_sha256']);return
 with connect() as c:
  if v.stage=='deploy':
   execute(c,'mkdir -p '+shlex.quote(R)+'/scripts '+shlex.quote(R)+'/receipts')
   with c.open_sftp() as s:
    for name in ['trace_protocol.json','trace_lock.json','scripts/trace_runtime.py']:
     try:
      with s.open(R+'/'+name,'rb') as h:assert hashlib.sha256(h.read()).hexdigest()==sha(F/name)
     except FileNotFoundError:s.put(str(F/name),R+'/'+name)
   print('DEPLOYMENT_FINISHED')
  elif v.stage=='launch':
   payload="from pathlib import Path;import os,subprocess,json,hashlib,shutil;f=Path("+repr(R)+");lock=json.loads((f/'trace_lock.json').read_text());assert hashlib.sha256((f/'trace_protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256'];assert hashlib.sha256((f/'scripts/trace_runtime.py').read_bytes()).hexdigest()==lock['script_sha256'];assert shutil.disk_usage(f).free>5*1024**3;assert not (f/'receipts/trace.pid').exists();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['LD_PRELOAD']='/root/autodl-tmp/dg_slam_chy_20261011/07_formal_tracking_T14_20261012/native/runtime_libs/libstdc++.so.6';log=(f/'receipts/trace.log').open('wb');p=subprocess.Popen(['/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/venv_dg/bin/python',str(f/'scripts/trace_runtime.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True);(f/'receipts/trace.pid').write_text(str(p.pid));print(p.pid)"
   execute(c,'/root/miniconda3/bin/python -c '+shlex.quote(payload))
  elif v.stage=='status':execute(c,'tail -n 6 '+shlex.quote(R+'/receipts/trace.log')+'; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader')
  elif v.stage=='collect':
   with c.open_sftp() as s:
    import stat
    def walk(remote,local):
     local.mkdir(exist_ok=True,parents=True)
     for i in s.listdir_attr(remote):
      if stat.S_ISDIR(i.st_mode):walk(remote+'/'+i.filename,local/i.filename)
      else:s.get(remote+'/'+i.filename,str(local/i.filename))
    walk(R+'/traces',F/'traces');walk(R+'/receipts',F/'receipts');s.get(R+'/trace_result.json',str(F/'trace_result.json'))
   print('COLLECTED')
if __name__=='__main__':main()
