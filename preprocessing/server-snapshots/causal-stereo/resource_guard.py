"""Run one owned process group with registered time and disk limits."""
from common import *
import subprocess,time,signal

def tree_bytes(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
def check_resources(limits):
    raw=tree_bytes(R);combined=raw+tree_bytes(E);free=__import__('shutil').disk_usage(R).free
    if free<limits['free_bytes']:raise RuntimeError('FREE_DISK')
    if raw>limits['raw_bytes']:raise RuntimeError('RAW_TREE_BYTES')
    if combined>limits['combined_bytes']:raise RuntimeError('COMBINED_TREE_BYTES')
    return dict(raw_bytes=raw,combined_bytes=combined,free_bytes=free)

def run_guarded(name,argv,timeout,env,limits,poll_seconds=10):
    initial=check_resources(limits);started=time.monotonic();child=None
    try:
        with (E/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                resources=check_resources(limits)
                dump(E/'runtime_progress.json',dict(status='RUNNING',at=now(),step=name,pid=child.pid,seconds=time.monotonic()-started,**resources))
                if time.monotonic()-started>timeout:raise RuntimeError('PHASE_TIME')
                try:child.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:pass
            assert child.returncode==0,(name,child.returncode)
        return dict(step=name,pid=child.pid,exit_code=child.returncode,seconds=time.monotonic()-started,initial_resources=initial)
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
