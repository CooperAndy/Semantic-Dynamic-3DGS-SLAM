"""Pinned SSH: additive deployment, detached launch, bounded read/status and collection."""
from pathlib import Path
import argparse, hashlib, json, shlex, sys, time
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F.parent/'CHY_接手DGSLAM_T11起_20261012/scripts'))
from server_io import connect
P=json.loads((F/'protocol.json').read_text(encoding='utf-8'));R=P['remote_root']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(c,cmd):
 i,o,e=c.exec_command(cmd,timeout=40);out=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status()
 if out:print(out,end='')
 if err:print(err,file=sys.stderr,end='')
 assert rc==0,(rc,cmd)
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['deploy','launch','status','collect']);v=a.parse_args()
 with connect() as c:
  if v.stage=='deploy':
   execute(c,'mkdir -p '+shlex.quote(R)+'/scripts '+shlex.quote(R)+'/inputs '+shlex.quote(R)+'/receipts '+shlex.quote(R)+'/runs')
   with c.open_sftp() as s:
    for p in [F/'protocol.json',F/'execution_lock.json',*sorted((F/'scripts').glob('*.py')),*sorted((F/'inputs').glob('*'))]:
     dst=R+'/'+str(p.relative_to(F)).replace('\\','/')
     try:
      with s.open(dst,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(p),dst
     except FileNotFoundError:s.put(str(p),dst)
   print('DEPLOYED_IMMUTABLE')
  elif v.stage=='launch':
   cmd='test ! -e '+shlex.quote(R+'/receipts/launch.pid')+' && nohup /root/miniconda3/bin/python '+shlex.quote(R+'/scripts/launch.py')+' > '+shlex.quote(R+'/receipts/launcher.log')+' 2>&1 < /dev/null &'
   # Use a short remote Python launcher to record the exact detached child PID.
   payload="from pathlib import Path;import subprocess;f=Path("+repr(R)+");assert not (f/'receipts/launch.pid').exists();log=(f/'receipts/launcher.log').open('wb');p=subprocess.Popen(['/root/miniconda3/bin/python',str(f/'scripts/launch.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);(f/'receipts/launch.pid').write_text(str(p.pid));print(p.pid)"
   execute(c,'/root/miniconda3/bin/python -c '+shlex.quote(payload))
  elif v.stage=='status':
   execute(c,'cat '+shlex.quote(R+'/receipts/launch.json')+'; tail -n 5 '+shlex.quote(R+'/receipts/numeric.log')+'; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader')
  elif v.stage=='collect':
   with c.open_sftp() as s:
    def walk(remote,local):
     import stat
     local.mkdir(parents=True,exist_ok=True)
     for item in s.listdir_attr(remote):
      rp=remote+'/'+item.filename;lp=local/item.filename
      if stat.S_ISDIR(item.st_mode):walk(rp,lp)
      else:s.get(rp,str(lp))
    for part in ['receipts','runs']:walk(R+'/'+part,F/part)
   print('COLLECTED')
if __name__=='__main__':main()
