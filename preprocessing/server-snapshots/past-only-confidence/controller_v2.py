from common import *
import subprocess,time,traceback,shutil
start=time.monotonic()
def run(name,args,limit):
 with (E/(name+'_v2.log')).open('x') as f:subprocess.run(args,cwd=R,env=env(),stdout=f,stderr=subprocess.STDOUT,timeout=limit,check=True)
try:
 assert json.loads((E/'controller.json').read_text())['status']=='FAILED_PRESERVED' and not (R/'cache').exists()
 a=json.loads((R/'amendment_v2.json').read_text())
 for n,h in a['source_files'].items():assert sha(R/n)==h
 run('contract',[str(PY),str(R/'contract_v2.py')],120)
 run('produce',['strace','-f','-ttt','-yy','-s','256','-e','trace=openat,write','-o',str(R/'file_activity.log'),str(PY),str(R/'producer.py')],1200)
 run('audit',[str(PY),str(R/'audit_v2.py')],600)
 assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<=6*2**30
 dump(E/'controller_v2.json',dict(status='COMPLETE_AUDITED_PAST_ONLY_INPUT_REPLAY',at=now(),seconds=time.monotonic()-start,quality_adoption=False))
except BaseException as e:
 dump(E/'controller_v2.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
