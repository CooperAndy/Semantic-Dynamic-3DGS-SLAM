from common import *
import time,subprocess,shutil,traceback
start=time.monotonic()
try:
 registration=json.loads((E/'pre_gpu_registration.json').read_text())
 for root,key in [(R,'root_files'),(E,'operator_files')]:
  for name,digest in registration[key].items():assert sha(root/name)==digest,name
 while time.monotonic()-start<86400:
  train=DE/'controller.json';analysis=DE/'analysis_supplement.json';free=shutil.disk_usage(R).free
  training=json.loads(train.read_text()) if train.exists() else {};evaluation=json.loads(analysis.read_text()) if analysis.exists() else {}
  parent_ready=training.get('status','').startswith('COMPLETE') and evaluation.get('status','').startswith('COMPLETE_QUALIFIED')
  dump(E/'parent_wait.json',dict(status='READY_STARTING' if parent_ready and free>=23*2**30 else 'WAITING_FOR_PARENT_COMPLETE_OR_SPACE',at=now(),parent_training=training.get('status'),parent_evaluation=evaluation.get('status'),free_bytes=free,required_free_bytes=23*2**30,seconds=time.monotonic()-start))
  if parent_ready and free>=23*2**30:break
  time.sleep(10)
 else:raise TimeoutError('Registered parent observation budget ended;inspect real parent before restart')
 with (E/'startup.log').open('x') as log:subprocess.run(['python3',str(R/'scripts/register_and_start.py')],env=env(),cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 dump(E/'parent_wait_terminal.json',dict(status='COMPLETE_PARENT_DEPENDENCIES_MET_AND_STARTER_RETURNED',at=now(),seconds=time.monotonic()-start))
except BaseException as err:
 dump(E/'parent_wait_terminal.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(err),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
