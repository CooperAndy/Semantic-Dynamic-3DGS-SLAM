"""Server coordinator: smoke gate, then independent 200-frame Stage B run."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,shutil

R=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
W=R/'stage_a/cuda_sm120';O=R/'stage_b'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
assert shutil.disk_usage(R).free>4*1024**3
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
preparation=json.loads((O/'preparation.json').read_text())
assert sha(R/'scripts/depth_consistent_selector.py')==preparation['selector_sha256']
assert sha(R/'scripts/run_stage_b.py')==preparation['worker_sha256']
assert sha(R/'source_v4_depthselection/dg_slam/gs_tracking_mapping.py')==preparation['mapper_sha256']
native=json.loads((W/'receipt.json').read_text())
for name,digest in native['extensions'].items():assert sha(W/'lib'/name)==digest
env=os.environ.copy();env.update(LD_PRELOAD=str(W/'runtime_libs/libstdc++.so.6'),
    LD_LIBRARY_PATH=str(W/'runtime_libs')+':'+env.get('LD_LIBRARY_PATH',''),
    PYTHONPATH=str(W/'lib')+':'+str(R/'scripts'),CUDA_HOME='/usr/local/cuda',
    OMP_NUM_THREADS='4',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
receipt={'status':'PREPARED','steps':[],'preparation_sha256':sha(O/'preparation.json'),
    'scope':'CHY KITTI00 Stage A+B; no BA or distance-limit change; native components identical to Stage A'}
def save(): (O/'pipeline_receipt.json').write_text(json.dumps(receipt,indent=2))
save()
try:
    for phase,frames,name in [('smoke40',40,'KITTI00_200_B_smoke40_20261012_01'),('full200',200,'KITTI00_200_B_on_20261012_01')]:
        if phase=='full200':
            d=R/'04_reproduction/KITTI00_200_B_smoke40_20261012_01'
            m=json.loads((d/'run_manifest.json').read_text());assert m['status']=='COMPLETED' and m['counters']['input_frames']==40
            h=[json.loads(x) for x in (d/'selection_history.jsonl').read_text().splitlines()]
            active=[x for x in h if x['keyframe']>1]
            success=[x for x in active if not x['fallback'] and x['iterations_completed']==50]
            assert active and len(success)/len(active)>=0.5, 'More than half refinements skipped; review observed-depth support before full run'
            g=[json.loads(x) for x in (d/'map_range_history.jsonl').read_text().splitlines()]
            assert g and all(x['xyz_finite'] and x['scale_finite'] and x['max_distance_current_camera_m']<1000 and x['max_activated_scale_m']<100 for x in g)
            receipt['smoke_gate']={'eligible':len(active),'full_50_iterations':len(success),'fallback':sum(x['fallback'] for x in active)}
        receipt.update(status=phase.upper()+'_RUNNING',run_name=name);save()
        with (O/(phase+'.log')).open('w') as log:
            completed=subprocess.run([sys.executable,str(R/'scripts/run_stage_b.py'),'--run-name',name,'--frames',str(frames),'--gs-refine','on'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        (O/(phase+'.exit')).write_text(str(completed.returncode)+'\n')
        assert completed.returncode==0,(phase,completed.returncode)
        receipt['steps'].append(phase+'_completed');save()
    receipt['status']='COMPLETED';save()
except Exception as error:
    receipt.update(status='FAILED',error=repr(error));save();raise
