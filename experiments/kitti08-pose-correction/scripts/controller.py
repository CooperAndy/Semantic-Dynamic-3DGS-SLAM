import protect_parent;protect_parent.install()
from stereo_proposal import verify_donor
from common import *
import subprocess,time,signal,traceback,shutil

start=time.monotonic();plan=json.loads((R/'plan.json').read_text());batch=[]
def seal():
 out=R/'runs'/plan['donor'];receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION'
 files=[out/'target_schedule.npz',out/'receipt.json']+sorted((out/'addition_masks').glob('*.npz'))
 dump(R/'receipts/donor.json',dict(status='PASS_SEALED_DONOR_INPUTS',at=now(),files=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
def root_bytes():return sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
try:
 for variant in plan['variants']:
  reg=json.loads((R/'registration.json').read_text())
  for name,digest in reg['source_files'].items():assert sha(R/name)==digest,name
  assert time.monotonic()-start<plan['budget']['total_seconds']
  assert shutil.disk_usage(R).free>plan['budget']['free_disk_bytes']
  with (E/(variant+'.log')).open('x') as f:
   p=subprocess.Popen([PY,str(R/'scripts/worker.py'),variant],stdout=f,stderr=subprocess.STDOUT,env=env(),start_new_session=True)
   dump(E/'current_process.json',dict(variant=variant,pid=p.pid,at=now()));run_start=time.monotonic();stop_reason=None
   while p.poll() is None:
    usage=root_bytes();combined=usage+sum(x.stat().st_size for x in E.rglob('*') if x.is_file());free=shutil.disk_usage(R).free;elapsed=time.monotonic()-run_start
    dump(E/'resource_progress.json',dict(at=now(),variant=variant,pid=p.pid,seconds=elapsed,root_bytes=usage,combined_bytes=combined,free_bytes=free))
    if elapsed>plan['budget']['per_run_seconds']:stop_reason='PER_RUN_TIME'
    elif time.monotonic()-start>plan['budget']['total_seconds']:stop_reason='BATCH_TIME'
    elif usage>plan['budget']['new_bytes']:stop_reason='NEW_TREE_BYTES'
    elif combined>plan['budget']['combined_new_bytes']:stop_reason='COMBINED_TREE_BYTES'
    elif free<plan['budget']['free_disk_bytes']:stop_reason='FREE_DISK'
    if stop_reason:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     break
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
  receipt=R/'runs'/variant/'receipt.json';j=json.loads(receipt.read_text()) if receipt.exists() else {}
  batch.append(dict(variant=variant,exit_code=p.returncode,stop_reason=stop_reason,status=j.get('status'),frames=j.get('frames',0),accepted=j.get('accepted'),error=j.get('error')))
  dump(E/'batch.json',dict(at=now(),runs=batch));assert p.returncode==0 and j['status']=='PASS_NATIVE_MAPPING_INTEGRATION',(variant,j.get('error'),stop_reason)
  verify_donor(R,variant)
 dump(E/'controller.json',dict(status='COMPLETE_6_BY_1000_STEREO_NATIVE_PENDING_FULL_AUDIT',at=now(),runs=batch,seconds=time.monotonic()-start,quality_adoption=False))
except BaseException as error:
 dump(E/'controller.json',dict(status='FAILED_PRESERVED',at=now(),error=repr(error),traceback=traceback.format_exc(),runs=batch));raise
