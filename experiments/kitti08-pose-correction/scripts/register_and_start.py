from common import *
import subprocess,time,traceback,shutil
start=time.monotonic()
try:
 assert not (R/'registration.json').exists();a=json.loads((E/'pre_gpu_registration.json').read_text())
 for folder,key in [(R,'root_files'),(E,'operator_files')]:
  for rel,digest in a[key].items():assert sha(folder/rel)==digest,rel
 assert json.loads((DE/'controller.json').read_text())['status'].startswith('COMPLETE') and json.loads((DE/'analysis_supplement.json').read_text())['status'].startswith('COMPLETE_QUALIFIED')
 assert shutil.disk_usage(R).free>=23*2**30
 with (E/'native_preflight.log').open('x') as log:subprocess.run([PY,str(R/'scripts/native_preflight.py')],env=env(),cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 birth=json.loads((R/'receipts/stable_birth_preflight.json').read_text());assert birth['status']=='PASS_1000_NEW_CAMERAS_96000_STABLE_BIRTH_POINTS' and birth['max_world_error_m']<.0003
 files={}
 for folder in ['scripts','source','native','inputs','provenance']:
  for p in (R/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[p.relative_to(R).as_posix()]=sha(p)
 for rel in ['plan.json','PROTOCOL.md']:files[rel]=sha(R/rel)
 plan=json.loads((R/'plan.json').read_text());dump(R/'registration.json',dict(at=now(),variants=plan['variants'],source_files=files,parent_inputs=plan['source_input_locks'],causal_inputs=True,stable_birth_sha256=plan['stable_birth_sha256'],pre_gpu_registration_sha256=sha(E/'pre_gpu_registration.json'),quality_adoption=False))
 operators={p.name:sha(p) for p in E.glob('*.py')};dump(E/'analysis_registration.json',dict(at=now(),operator_scripts=operators,scope='Full six1000 audit/EVO480/quality;strict original birth gate unchanged',quality_adoption=False));pids={}
 for name,py,script in [('archive','python3',E/'archive_completed.py'),('analysis','python3',E/'analysis_controller.py'),('training',PY,R/'scripts/controller.py')]:
  with (E/(name+'_controller.log')).open('x') as log:p=subprocess.Popen([py,str(script)],env=env(),cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  pids[name]=p.pid;dump(E/'operator_pids.json',dict(at=now(),pids=pids))
 dump(E/'startup.json',dict(status='COMPLETE_REGISTERED_AND_FULL_TRAINING_STARTED',at=now(),pids=pids,seconds=time.monotonic()-start,quality_adoption=False))
except BaseException as err:
 dump(E/'startup.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(err),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
