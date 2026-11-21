from pathlib import Path
import sys,os,subprocess,json,hashlib,time
R=Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1');OLD=Path('/root/autodl-tmp/A2_stereo_robust_geometry/recheck08_v1');SEEDS=(0,42,2026)
phase=sys.argv[1];assert phase in ('smoke','formal','score')
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTHONPATH='/root/autodl-tmp/stage3_g15_20261028_p47149/native')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run(label,cmd):
 print('START',label,flush=True);start=time.monotonic()
 with (R/f'{label}.log').open('xb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=2200)
 if r.returncode:raise RuntimeError(f'{label} returned {r.returncode}; inspect log')
 print('PASS',label,'seconds',round(time.monotonic()-start,1),flush=True)
def audit_pass(seed,n):
 s=R/f'seed_{seed}';p=s/f'independent_audit_v2_{n}.json'
 assert p.exists() and json.loads(p.read_text())['status']=='PASS'
 for arm in ('C0','C2'):
  d=s/'runs'/f'{arm}_{n}';rec=json.loads((d/'a2_receipt.json').read_text());assert rec['status']=='PASS_ENGINEERING' and rec['seed']==seed and sha(d/'params.npz')==rec['map_sha256']
def gpu_idle():
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if p:raise RuntimeError('Other GPU processes active; wait for idle')
for seed in SEEDS:
 reg=json.loads((R/f'seed_{seed}'/'run_registration.json').read_text())
 assert reg['seed']==seed
 for lock in reg['file_locks']:
  if sha(Path(lock['path']))!=lock['sha256']:raise RuntimeError('Registered file changed: '+lock['path'])
if phase=='smoke':
 for seed in SEEDS:
  gpu_idle();s=R/f'seed_{seed}';run(f'smoke_s{seed}',[sys.executable,str(s/'recheck08_run25.py')]);run(f'audit25_s{seed}',[sys.executable,str(s/'recheck08_audit.py'),'25']);audit_pass(seed,25)
 (R/'smoke_ready.json').write_text(json.dumps({'status':'PASS_ALL_THREE_SEED_PAIRS_ENGINEERING_ONLY','seeds':list(SEEDS),'no_quality_score':True},indent=2))
elif phase=='formal':
 assert json.loads((R/'smoke_ready.json').read_text())['status']=='PASS_ALL_THREE_SEED_PAIRS_ENGINEERING_ONLY'
 for seed in SEEDS:audit_pass(seed,25)
 for seed in SEEDS:
  gpu_idle();s=R/f'seed_{seed}';batch=s/'batch200.json'
  if batch.exists():
   assert json.loads(batch.read_text())['state']=='RUN200_TWO_RUNS_COMPLETE_AUDIT_PENDING', 'Partial batch requires investigation'
  else:run(f'formal_s{seed}',[sys.executable,str(s/'recheck08_run200.py')])
  if not (s/'independent_audit_v2_200.json').exists():run(f'audit200_s{seed}_adapter_v2',[sys.executable,str(s/'recheck08_audit.py'),'200'])
  audit_pass(seed,200)
 (R/'all_maps_ready.json').write_text(json.dumps({'status':'SIX_MAPS_AUDITED_SCORE_PENDING','seeds':list(SEEDS)},indent=2))
else:
 assert json.loads((R/'all_maps_ready.json').read_text())['status']=='SIX_MAPS_AUDITED_SCORE_PENDING'
 for seed in SEEDS:audit_pass(seed,200)
 seal=[]
 for seed in SEEDS:
  for arm in ('C0','C2'):
   p=R/f'seed_{seed}'/'runs'/f'{arm}_200'/'params.npz';seal.append({'path':str(p),'sha256':sha(p)})
 for p in [R/'protocol.md',R/'input_manifest.json',R/'lidar_projected/manifest.json']:
  seal.append({'path':str(p),'sha256':sha(p)})
 for p in sorted((R/'lidar_projected').glob('*.npz')):seal.append({'path':str(p),'sha256':sha(p)})
 (R/'pre_score_seal.json').write_text(json.dumps({'status':'SEALED_BEFORE_SCORE','items':seal},indent=2))
 for seed in SEEDS:
  gpu_idle();s=R/f'seed_{seed}'
  run(f'image_score_s{seed}',[sys.executable,str(s/'recheck08_score.py'),'200'])
  run(f'lidar_export_s{seed}',[sys.executable,str(s/'recheck08_export.py')])
  run(f'lidar_config_s{seed}',[sys.executable,str(R/'build_lidar03_config.py'),str(seed)])
  run(f'lidar_score_s{seed}',[sys.executable,str(OLD/'source/score_lidar_renders.py'),'--config',str(s/'lidar_score_config_200.json'),'--out',str(s/'lidar_score_200.json')])
  run(f'lidar_summary_s{seed}',[sys.executable,str(OLD/'aggregate_lidar_score_v1.py'),'--infile',str(s/'lidar_score_200.json'),'--out',str(s/'lidar_summary_200.json')])
  run(f'independent_lidar_audit_s{seed}',[sys.executable,str(R/'independent_score03.py'),'--window','03','--config',str(s/'lidar_score_config_200.json'),'--score',str(s/'lidar_score_200.json'),'--out',str(s/'independent_lidar_audit.json')])
 for x in seal:assert sha(Path(x['path']))==x['sha256'],x['path']
 (R/'scoring_complete.json').write_text(json.dumps({'status':'SCORED_PER_SEED_NO_ADOPTION','seeds':list(SEEDS),'quality_adoption':False},indent=2))
