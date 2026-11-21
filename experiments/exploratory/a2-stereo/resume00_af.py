import json,hashlib,os,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/A2_union_20261029');py='/root/miniconda3/bin/python'
env=os.environ.copy();env.update(PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled')
arms={'B':('C0','08',60,3000),'A':('C2','08',60,3000),'F':('C0','06',240,6000),'AF':('C2','06',240,6000)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def path(a,s):
 arm,g,steps,cap=arms[a];return R/'runs'/f'{arm}_g{g}_{steps}_{cap}_200_s{s}'
def check(a,s):
 p=path(a,s); rec=json.loads((p/'combo_receipt.json').read_text());assert rec['status']=='PASS_ENGINEERING' and rec['seed']==s and sha(p/'params.npz')==rec['map_sha256'],(a,s);return rec
locks=json.loads((R/'pre_run_locks.json').read_text())['items']
for x in locks:assert sha(x['path'])==x['sha256'],x['path']
assert json.loads((R/'input_audit.json').read_text())['status']=='PASS'
for s in (0,1,2):
 b=check('B',s);a=check('A',s);f=check('F',s);assert a['initial_hash']==b['initial_hash'];assert not path('AF',s).exists()
print('PREFLIGHT PASS; 00 AF ONLY',flush=True)
procs=[]
for s in (0,1,2):
 log=(R/f'resume_AF_s{s}.log').open('xb');p=subprocess.Popen([py,str(R/'union_worker.py'),'240','6000','200',str(s),'06','C2'],stdout=log,stderr=subprocess.STDOUT,env=env);procs.append((s,log,p))
for s,log,p in procs:
 code=p.wait();log.close()
 if code:raise RuntimeError(f'AF seed {s} failed {code}, preserved')
 check('AF',s)
 print('PASS AF',s,flush=True)
items=[]
for s in (0,1,2):
 rec={a:check(a,s) for a in arms};assert rec['B']['initial_hash']==rec['A']['initial_hash'] and rec['F']['initial_hash']==rec['AF']['initial_hash']
 with np.load(path('B',s)/'params.npz') as z:camera={k:z[k].copy() for k in ('cam_unnorm_rots','cam_trans','intrinsics','w2c')}
 for a in arms:
  with np.load(path(a,s)/'params.npz') as z:assert all(np.array_equal(z[k],v) for k,v in camera.items()),(a,s,'camera mismatch')
  mp=path(a,s)/'params.npz';items.append({'arm':a,'seed':s,'path':str(mp),'sha256':sha(mp)})
for x in locks:assert sha(x['path'])==x['sha256'],x['path']
seal=R/'pre_score_seal.json';assert not seal.exists();seal.write_text(json.dumps({'status':'MAPS_SEALED_BEFORE_SCORE','items':items},indent=2))
print('SEALED 12 MAPS',flush=True)
for s in (0,1,2):
 with (R/f'score_s{s}.log').open('xb') as log:subprocess.run([py,str(R/'union_score.py'),'200',str(s)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
 with (R/f'export00_s{s}.log').open('xb') as log:subprocess.run([py,str(R/'export_00.py'),'200',str(s)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
 print('SCORED',s,flush=True)
with (R/'lidar00.log').open('xb') as log:subprocess.run([py,str(R/'score_lidar_00.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
with (R/'independent_lidar_audit_00.log').open('xb') as log:subprocess.run([py,str(R/'audit_lidar_scores.py'),'00'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
for x in items:assert sha(x['path'])==x['sha256']
(R/'resume00_complete.json').write_text(json.dumps({'status':'COMPLETE_00_SCORED_INDEPENDENT_LIDAR_ARITHMETIC_AUDIT','quality_adoption':False,'seeds':[0,1,2]},indent=2))
print('RESUME00_COMPLETE',flush=True)
