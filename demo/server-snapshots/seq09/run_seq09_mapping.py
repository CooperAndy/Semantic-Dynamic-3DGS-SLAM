"""Prepare isolated S23 configurations and run B then TMD on seq09."""
from pathlib import Path
import json, os, subprocess, shutil, hashlib, sys
ROOT=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114'); S=Path('/root/autodl-tmp/s23_effects_20261105_p16247'); B=ROOT/'bundle'; R=ROOT/'S0'
sys.path.insert(0,str(S))
code=(S/'run_registered_matrix.py').read_text()
code=code.replace("m['frame_count']==1000", "m['frame_count']==200").replace('range(1000)', 'range(200)').replace("cell['source_start']+1000", "cell['source_start']+200").replace('range(19,1000,20)', 'range(19,200,20)').replace('num_frames=1000', 'num_frames=200')
ns={'__name__':'seq09_adapter'}; exec(compile(code,str(ROOT/'matrix_adapter.py'),'exec'),ns)
cell=dict(stage='E3', sequence='09', source_start=300, frames=200, seed=0)
ns['prepare_cell'](cell,B,R,candidate=S/'candidate_v4_fp64accum')
env=dict(os.environ,S23_RASTER_ROOT=str(S/'raster_fp64_accum_v1'),PYTHONPATH=str(S/'raster_fp64_accum_v1'),PYTORCH_CUDA_ALLOC_CONF='backend:native',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled')
py='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'
for group in ['B','TMD']:
    print('START',group,flush=True)
    with (ROOT/f'{group}.log').open('x') as log:
        subprocess.run([py,str(R/'scripts/worker.py'),group],cwd=R/'scripts',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((R/'runs'/group/'receipt.json').read_text())
    assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and receipt['frames']==200
    if group=='B':
        files=[R/'runs/B/target_schedule.npz']+sorted((R/'runs/B/addition_masks').glob('*.npz'))
        assert len(files)==190, len(files)
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        (R/'receipts/donor.json').write_text(json.dumps(dict(status='PASS_SEALED_DONOR_INPUTS',files=[dict(path=p.relative_to(R).as_posix(),sha256=digest(p)) for p in files]),indent=2))
    print('COMPLETE',group,flush=True)
