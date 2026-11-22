"""CPU-only GPU-context sampling; never stops, pauses or changes another task."""
import argparse,datetime,json,subprocess,time
from pathlib import Path


def sample(remote):
    remote=Path(remote)
    state=json.loads((remote/'long_operator_progress.json').read_text())
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    gpu=[]
    for line in raw.splitlines():
        if not line.strip():continue
        pid,executable,memory=[v.strip() for v in line.split(',',2)]
        try:args=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
        except OSError:args=[]
        script=args[1].decode(errors='replace') if len(args)>1 else ''
        gpu.append(dict(pid=int(pid),executable=executable,script=script,memory_MiB=float(memory),
            task_owned=script.startswith(str(remote)+'/')))
    return dict(at_unix=time.time(),status=state['status'],cell=state.get('cell'),group=state.get('group'),GPU_contexts=gpu,
        interpretation='Allocated GPU contexts sampled, not per-process compute utilization')


def monitor(remote):
    import fcntl
    remote=Path(remote);directory=remote/'resources';directory.mkdir(exist_ok=True)
    lock=(directory/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.monotonic()
    with (directory/'GPU_context_samples.jsonl').open('a',buffering=1) as log:
        while time.monotonic()-begin<7*86400:
            try:
                row=sample(remote);log.write(json.dumps(row)+'\n')
                (directory/'monitor_progress.json').write_text(json.dumps(dict(status='SAMPLING',last_sample=row['at_unix'],cell=row['cell'],group=row['group'])))
                if row['status'] in ['COMPLETE32_RUNS_AND_EVALUATIONS_LOCAL_VERIFIED','STOPPED_WITH_ORIGINALS_PRESERVED']:return
            except Exception as error:
                (directory/'monitor_progress.json').write_text(json.dumps(dict(status='RETRYING_READ_ONLY_OBSERVER',error=repr(error),at_unix=time.time())))
            time.sleep(15)


def snapshot(remote,root,group,out):
    remote=Path(remote);root=Path(root);out=Path(out)
    path=remote/'resources/GPU_context_samples.jsonl';rows=[]
    if path.exists():
        data=path.read_text();lines=data.splitlines()
        if data and not data.endswith('\n'):lines=lines[:-1]
        for line in lines:
            row=json.loads(line)
            if row['cell']==root.name and row['group']==group and row['status']=='RUNNING_NATIVE_GROUP':rows.append(row)
    start=json.loads((root/'runs'/group/'shared_load_start.json').read_text())
    started=datetime.datetime.fromisoformat(start['at']).timestamp()
    overlap=[r for r in rows if any(not p['task_owned'] for p in r['GPU_contexts']) or len(r['GPU_contexts'])>1]
    complete=bool(rows) and rows[0]['at_unix']-started<=30
    pause_file=root/'runs'/group/'demo_extension_pause_resolved.json'
    paused_seconds=json.loads(pause_file.read_text())['user_priority_pause_seconds'] if pause_file.exists() else 0.
    result=dict(status='OVERLAP_OBSERVED' if overlap else 'USER_PRIORITY_PAUSE' if paused_seconds>0 else 'NO_OVERLAP_OBSERVED' if rows else 'NO_RESOURCE_SAMPLES',
        samples=len(rows),overlap_samples=len(overlap),sampling_interval_seconds=15,
        sampling_covers_native_start=complete,runtime_comparison_eligible=complete and not overlap and paused_seconds==0,
        user_priority_pause_seconds=paused_seconds,
        maximum_other_GPU_MiB=max((sum(p['memory_MiB'] for p in r['GPU_contexts'] if not p['task_owned']) for r in rows),default=0),
        note='Sampling does not prove absence of short unsampled overlap; no timing subtraction or quality-causality claim',rows=rows)
    (out/'shared_GPU_resource_audit.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':
    from experiment import REMOTE
    parser=argparse.ArgumentParser();parser.add_argument('--monitor',action='store_true');args=parser.parse_args()
    if args.monitor:monitor(REMOTE)
    else:print(json.dumps(sample(REMOTE)))
