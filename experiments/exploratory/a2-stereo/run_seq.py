import json, hashlib, os, subprocess, sys, time
from pathlib import Path
import numpy as np
P=Path('/root/autodl-tmp/A2_union_20261029')
seq=sys.argv[1];assert seq in ('03','08')
R=P/f'seq{seq}'
py='/root/miniconda3/bin/python'
env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled')
arms={'B':('C0','08',60,3000),'A':('C2','08',60,3000),'F':('C0','06',240,6000),'AF':('C2','06',240,6000)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def path(name,n,seed):
 a,g,s,c=arms[name];return R/'runs'/f'{a}_g{g}_{s}_{c}_{n}_s{seed}'
def check(name,n,seed):
 p=path(name,n,seed); rec=json.loads((p/'combo_receipt.json').read_text());assert rec['status']=='PASS_ENGINEERING' and rec['seed']==seed and sha(p/'params.npz')==rec['map_sha256'];return rec
assert json.loads((P/f'seq{seq}_gate06'/'audit.json').read_text())['status']=='PASS'
sm={a:check(a,25,0) for a in arms}
assert sm['A']['initial_hash']==sm['B']['initial_hash'] and sm['F']['initial_hash']==sm['AF']['initial_hash']
locks=[P/'union_worker_seq.py',P/'union_score_seq.py',P/'prepare_seq_gates.py',P/f'seq{seq}_gate06'/'audit.json',P/'source/scripts/splatam_external_tm.py',P/'source/a2_factorial_losses.py',P/'source/c04n_addition.py']
(R/'pre_run_locks.json').write_text(json.dumps({'items':[{'path':str(p),'sha256':sha(p)} for p in locks]},indent=2))
for name,(arm,gate,steps,cap) in arms.items():
 procs=[]
 for seed in (0,1,2):
  log=(R/f'run_{name}_s{seed}.log').open('xb');cmd=[py,str(P/'union_worker_seq.py'),str(steps),str(cap),'200',str(seed),gate,arm,seq]
  procs.append((seed,log,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)))
 print('START',name,flush=True)
 for seed,log,p in procs:
  code=p.wait();log.close()
  if code:raise RuntimeError(f'{name} seed {seed} exit {code}; run halted')
  check(name,200,seed)
 print('PASS',name,flush=True)
for seed in (0,1,2):
 rec={a:check(a,200,seed) for a in arms};assert rec['B']['initial_hash']==rec['A']['initial_hash'] and rec['F']['initial_hash']==rec['AF']['initial_hash']
 with np.load(path('B',200,seed)/'params.npz') as base:
  camera={k:base[k].copy() for k in ('cam_unnorm_rots','cam_trans','intrinsics','w2c')}
 for name in arms:
  with np.load(path(name,200,seed)/'params.npz') as z:
   assert all(np.array_equal(z[k],v) for k,v in camera.items()),(seed,name,'camera')
for p in locks: assert sha(p)==next(i['sha256'] for i in json.loads((R/'pre_run_locks.json').read_text())['items'] if i['path']==str(p))
seal={'status':'MAPS_SEALED_BEFORE_SCORE','items':[{'arm':a,'seed':seed,'path':str(path(a,200,seed)/'params.npz'),'sha256':sha(path(a,200,seed)/'params.npz')} for seed in (0,1,2) for a in arms]}
(R/'pre_score_seal.json').write_text(json.dumps(seal,indent=2))
for seed in (0,1,2):
 with (R/f'score_s{seed}.log').open('xb') as log:
  subprocess.run([py,str(P/'union_score_seq.py'),'200',str(seed),seq],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
print('ALL_COMPLETE',flush=True)
