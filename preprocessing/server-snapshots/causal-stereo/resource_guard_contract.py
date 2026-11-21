"""CPU integration: success, owned timeout, pre-launch disk rejection, unrelated child preserved."""
from common import *
from resource_guard import run_guarded
import subprocess,sys
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
limits=dict(free_bytes=2**30,raw_bytes=18*2**30,combined_bytes=27*2**30)
sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
proof=[]
try:
    proof.append(run_guarded('contract_owned_success',[sys.executable,'-c','print("synthetic child complete")'],5,env,limits,poll_seconds=.05))
    try:run_guarded('contract_owned_timeout',[sys.executable,'-c','import time;time.sleep(30)'],.15,env,limits,poll_seconds=.05)
    except RuntimeError as err:assert str(err)=='PHASE_TIME';proof.append(dict(owned_timeout_stopped=True))
    else:raise AssertionError('Owned timeout escaped')
    impossible=dict(limits,free_bytes=10**30)
    try:run_guarded('contract_disk_reject',[sys.executable,'-c','raise SystemExit(99)'],5,env,impossible)
    except RuntimeError as err:assert str(err)=='FREE_DISK';proof.append(dict(low_disk_rejected_before_launch=True))
    else:raise AssertionError('Disk guard escaped')
    assert not (E/'contract_disk_reject.log').exists() and sentinel.poll() is None
    dump(R/'resource_guard_contract_receipt.json',dict(status='PASS_REAL_CHILD_SUCCESS_OWNED_TIMEOUT_DISK_GATE_AND_UNRELATED_PROCESS_PRESERVATION',at=now(),proof=proof,unrelated_synthetic_child_preserved=True,GPU_calls=0,actual_dataset_GT_reads=0))
finally:
    if sentinel.poll() is None:sentinel.terminate();sentinel.wait(timeout=5)
print('PASS_OWNED_RESOURCE_GUARD',flush=True)
