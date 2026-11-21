"""Generate TSDD observations for the independent seq09 experiment."""
from pathlib import Path
import os, subprocess, sys, numpy as np
R=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114')
B=R/'bundle'; S=Path('/root/autodl-tmp/s23_effects_20261105_p16247'); D=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499'); P=Path('/root/autodl-tmp/slam_project/stage2_tsdd')
py='/root/venvs/semantic_seg/bin/python'; image=D/'sequences/09/image_2'
(B/'motion_depth').mkdir(exist_ok=False)
for i in range(200):
    a=np.load(B/'raw_left'/f'{i:06}_depth.npy',allow_pickle=False)
    np.save(B/'motion_depth'/f'{i:06}_depth.npy',np.where(np.isfinite(a)&(a>=.5)&(a<=80),a,0).astype('float32'),allow_pickle=False)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
commands=[
 [py,str(P/'scripts/compute_motion_evidence.py'),'--image-dir',str(image),'--depth-dir',str(B/'motion_depth'),'--pose-file',str(D/'poses_estimated_camera2.txt'),'--pose-coordinate','p2','--calib-file',str(B/'calib.txt'),'--instance-dir',str(B/'instances'),'--output-dir',str(B/'motion'),'--start','0','--end','199','--device','cuda:0'],
 [py,str(P/'scripts/run_tsdd.py'),'--image-dir',str(image),'--instance-dir',str(B/'instances'),'--evidence-dir',str(B/'motion'),'--config',str(P/'configs/tsdd.yaml'),'--output-dir',str(B/'states'),'--start','0','--end','198','--run-status','ENGINEERING_SMOKE'],
 [py,str(S/'export_observations.py'),str(B),'09']]
for i,cmd in enumerate(commands):
    print('START',i,flush=True)
    with (R/f'input_step_{i}.log').open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    print('COMPLETE',i,flush=True)
