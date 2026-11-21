from pathlib import Path
import os,subprocess,time,sys
R=Path('/root/autodl-tmp/A2_union_20261029'); py='/root/miniconda3/bin/python';env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled')
for _ in range(720):
 log=(R/'run_00.log').read_text(errors='replace') if (R/'run_00.log').exists() else ''
 if 'ALL_COMPLETE' in log:break
 if 'Traceback (most recent call last)' in log:raise RuntimeError('00 mapping pipeline failed; inspect run_00.log')
 time.sleep(30)
else:raise TimeoutError('00 did not complete within 6h')
for seed in (0,1,2):
 with (R/f'export00_s{seed}.log').open('xb') as f:subprocess.run([py,str(R/'export_00.py'),'200',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'lidar00.log').open('xb') as f:subprocess.run([py,str(R/'score_lidar_00.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('COMPLETE_00_WITH_LIDAR',flush=True)
