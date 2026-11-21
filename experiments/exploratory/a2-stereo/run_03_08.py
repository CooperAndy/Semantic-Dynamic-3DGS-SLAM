from pathlib import Path
import os,subprocess,sys,time,json
P=Path('/root/autodl-tmp/A2_union_20261029');py='/root/miniconda3/bin/python';env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled')
arms={'B':('C0','08',60,3000),'A':('C2','08',60,3000),'F':('C0','06',240,6000),'AF':('C2','06',240,6000)}
for seq in ('03','08'):
 r=P/f'seq{seq}';r.mkdir(exist_ok=False)
 print('START',seq,flush=True)
 for name,(arm,gate,steps,cap) in arms.items():
  with (r/f'smoke_{name}.log').open('xb') as log:
   subprocess.run([py,str(P/'union_worker_seq.py'),str(steps),str(cap),'25','0',gate,arm,seq],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('SMOKE PASS',seq,name,flush=True)
 with (r/'smoke_score.log').open('xb') as log:subprocess.run([py,str(P/'union_score_seq.py'),'25','0',seq],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (r/'run_formal.log').open('xb') as log:subprocess.run([py,str(P/'run_seq.py'),seq],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (r/'lidar.log').open('xb') as log:subprocess.run([py,str(P/'score_lidar_seq.py'),seq],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('COMPLETE',seq,flush=True)
