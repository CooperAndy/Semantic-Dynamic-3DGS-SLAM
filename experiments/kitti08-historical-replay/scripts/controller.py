from common import *
import subprocess, os, time, signal, datetime, numpy as np

env=os.environ.copy();env.pop('PYTHONPATH',None)
env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',TZ='Asia/Shanghai',PYTHONDONTWRITEBYTECODE='1')
original=json.loads((R/'provenance/original_registration.json').read_text())
deadline=datetime.datetime.fromisoformat(original['registered_at']).timestamp()+33000

def state(s,**kw):
    dump(E/'state.json',dict(state=s,at=now(),**kw));print(s,kw,flush=True)

def task(name,limit,args=None):
    args=args or [];state('RUNNING',step=name,args=args)
    with (E/(name+('_'+args[0] if args else '')+'.log')).open('xb') as f:
        p=subprocess.run([PY,str(R/'scripts'/(name+'.py')),*args],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=limit)
    assert p.returncode==0,(name,p.returncode)

try:
    task('prepare',300)
    # All generated configurations must differ from the original only in the root path.
    oldroot=Path(original['root'])
    for v in VARIANTS:
        a=(R/'source/configs/kitti'/f'{v}.py').read_text()
        b=(oldroot/'source/configs/kitti'/f'{v}.py').read_text().replace(str(oldroot),str(R))
        assert a==b,('CONFIG_CHANGE_BEYOND_ROOT',v)
    task('hook_contract',120);task('register',240)
    with np.load(R/'inputs/runtime/orb_adapter_reference.npz') as a,np.load(oldroot/'inputs/runtime/orb_adapter_reference.npz') as b:
        assert a.files==b.files
        assert all(np.array_equal(a[k],b[k]) for k in a.files),'New-device adapter differs; stop before training'
    lineage=json.loads((R/'provenance/recovery_manifest.json').read_text())
    for x in lineage['inherited_run_files']:
        assert sha(R/x['path'])==x['sha256']
    dump(R/'receipts/recovery_bridge.json',dict(status='PASS_ORIGINAL4773_LOCKS_FOUR_INHERITED_RUNS_AND_IDENTICAL1000_CAMERAS',at=now(),same_core=True,same_native=True,original_camera_arrays_identical=True))
    batch=json.loads((R/'provenance/original_receipts/batch.json').read_text())['runs']
    for v in ['B00C0_S2','B03C0_S2']:
        assert time.time()+5400<deadline,'Original batch wall-clock budget cannot cover another full run'
        if definition(v)['history_budget']:check_donor(2)
        state('RUNNING',step='MATRIX',variant=v);st=time.monotonic();timeout=False
        with (R/'receipts'/f'{v}.log').open('xb') as f:
            p=subprocess.Popen([PY,str(R/'scripts/worker.py'),v],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:p.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        rp=R/'runs'/v/'smoke_receipt.json';w=json.loads(rp.read_text()) if rp.exists() else {}
        status='COMPLETE_WINDOW' if p.returncode==0 and w.get('status')=='PASS_ENGINEERING' else ('TIME_BUDGET_STOP' if timeout else 'FATAL_FAILURE_PRESERVED')
        batch.append(dict(variant=v,status=status,frames=w.get('frames',0),wall_seconds=time.monotonic()-st,exit_code=p.returncode,error=w.get('error'),execution_host='p10781'))
        dump(R/'receipts/batch.json',dict(at=now(),runs=batch,interrupted_attempt_in_provenance=True))
        assert status=='COMPLETE_WINDOW',(v,status,w.get('error'))
        if not definition(v)['history_budget']:task('seal_donor',180,['2'])
    task('audit',600);task('evaluate',1200);task('audit_scores',600);task('seal',600)
    state('READY_FOR_LOCAL_STREAM_ARCHIVE',quality_adoption=False)
except BaseException as error:
    state('FAILED_PRESERVED',error=repr(error));raise
