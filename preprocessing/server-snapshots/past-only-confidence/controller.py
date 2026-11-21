from common import *
import subprocess,time,traceback,shutil
start=time.monotonic();commands=[]
def run(name,cmd,timeout):
 commands.append(dict(name=name,argv=cmd,at=now()));dump(E/'commands.json',commands)
 with (E/(name+'.log')).open('x') as log:subprocess.run(cmd,env=env(),cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
try:
 assert shutil.disk_usage(R).free>15*2**30
 run('prepare',[str(PY),str(R/'prepare.py')],600);run('contract',[str(PY),str(R/'contract.py')],120)
 run('produce',['strace','-f','-ttt','-yy','-s','256','-e','trace=openat,write','-o',str(R/'file_activity.log'),str(PY),str(R/'producer.py')],1200)
 run('audit',[str(PY),str(R/'audit.py')],600)
 assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<=6*2**30
 dump(E/'controller.json',dict(status='COMPLETE_AUDITED_PAST_ONLY_INPUT_REPLAY',at=now(),seconds=time.monotonic()-start,quality_adoption=False))
except BaseException as e:
 dump(E/'controller.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
