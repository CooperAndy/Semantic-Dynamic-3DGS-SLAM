"""Finalize immutable queue registration after CPU preparation, before any G20 GPU work."""
from common import *
import subprocess,time
assert not (R/'registration.json').exists() and not (E/'pre_gpu_registration.json').exists()
assert json.loads((R/'receipts/preparation_cpu.json').read_text())['status'].startswith('PASS_3000')
plan=json.loads((R/'plan.json').read_text());plan['budget'].update(combined_new_bytes=16*2**30,start_min_free_bytes=23*2**30);dump(R/'plan.json',plan)
timestamp=now();dump(R/'receipts/pre_gpu_budget_registration.json',dict(at=timestamp,status='ADDED_COMBINED_DISK_LIMIT_BEFORE_ANY_MODEL_FRAMES',model_frames=0,raw_limit=12*2**30,combined_limit=16*2**30,start_min_free=23*2**30))
root_files={}
for folder in ['scripts','source','native','inputs','provenance']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':root_files[p.relative_to(R).as_posix()]=sha(p)
for rel in ['plan.json','PROTOCOL.md']:root_files[rel]=sha(R/rel)
for name in ['backend.py','factor_hooks.py','feature_pose.py','pose_chain.py','visibility.py']:assert sha(R/'scripts'/name)==sha(D/'scripts'/name)
registration=dict(status='REGISTERED_CPU_READY_WAITING_FOR_D_AND_SPACE_NO_GPU_RUN_YET',at=timestamp,root_files=root_files,operator_files={p.name:sha(p) for p in E.glob('*.py')},GPU_calls=0,quality_adoption=False);dump(E/'pre_gpu_registration.json',registration)
with (E/'parent_wait.log').open('x') as log:p=subprocess.Popen(['python3',str(R/'scripts/wait_parent.py')],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
j=dict(at=timestamp,pid=p.pid,status='QUEUED_FULL_SIX1000_AFTER_D_AND_SPACE',root_files=len(root_files),operator_files=len(registration['operator_files']));dump(E/'parent_wait_pid.json',j);print(json.dumps(j))
