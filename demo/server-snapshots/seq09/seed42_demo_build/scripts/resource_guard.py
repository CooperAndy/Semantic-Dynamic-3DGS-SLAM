"""Shared-GPU admission and per-process PyTorch limits; never stop other work."""
import json,os,subprocess,time

def snapshot():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    total,used,free=map(int,row.split(','))
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    own=sum(int(r.split(',')[1]) for r in rows if r.strip() and int(r.split(',')[0])==os.getpid())
    return dict(total_mib=total,used_mib=used,free_mib=free,own_mib=own,other_mib=used-own,pid=os.getpid())

def wait_for_headroom(stage,frame=-1):
    limit=float(os.environ.get('TASK_GPU_GIB','6'))
    while True:
        s=snapshot();other=2560 if limit>6 else 3584
        # Heavy fixed-shape inference is capped at7.5GiB (observed peak6.965).
        # Keep >=1.5GiB beyond any remaining allocation budget. Frame-end cache
        # release gives other jobs their workspace back before the next check.
        needed=max(0,int(limit*1024)-s['own_mib'])+1536 if limit>6 else (int(limit*1024+700) if s['own_mib']==0 else 1100)
        if s['other_mib']<=other and s['free_mib']>=needed:return
        print(json.dumps(dict(waiting=stage,frame=frame,**s)),flush=True);time.sleep(2 if limit>6 else 15)

if __name__=='__main__':
    import sys,runpy
    from pathlib import Path
    wait_for_headroom('before_model_load')
    import torch
    limit=float(os.environ.get('TASK_GPU_GIB','6'))
    torch.cuda.set_per_process_memory_fraction(limit*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    here=Path(__file__).resolve().parent
    while not (here/'input_spec.json').exists() and here!=here.parent:here=here.parent
    sys.path.insert(0,str(here/'vendor'))
    print(json.dumps(dict(gpu_limit_gib=limit,**snapshot())),flush=True)
    script=sys.argv[1];sys.argv=sys.argv[1:]
    try:runpy.run_path(script,run_name='__main__')
    finally:print(json.dumps(dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,**snapshot())),flush=True)
