"""Isolated ORB/H3 cap3000 run: 20-frame smoke, then 260-frame O3 and F1."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
P=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115');S23=Path('/root/autodl-tmp/s23_effects_20261105_p16247');CAND=S23/'candidate_v4_fp64accum';RASTER=S23/'raster_fp64_accum_v1';PY='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def state(s,**kw):
 p=P/'o3f_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=s,time=time.strftime('%FT%T%z'),**kw),indent=2));t.replace(p);print(s,kw,flush=True)
def run(name,cmd,env):
 if shutil.disk_usage(P).free<10*2**30:raise RuntimeError('Free disk below 10 GiB; stop and retain outputs')
 state(name)
 with (P/f'{name}.log').open('x') as f:subprocess.run(list(map(str,cmd)),cwd=S23,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def case(n,name):
 source=json.loads((P/'bundle/runtime.json').read_text());bundle=P/f'bundle_{name}';bundle.mkdir();rows=source['rows'][:n];assert len(rows)==n
 # ManifestDataset checks that pose file length exactly equals frame_count.
 pose=np.loadtxt(source['pose_file']);assert pose.shape==(260,12);small=bundle/'estimated_p2.txt';np.savetxt(small,pose[:n],fmt='%.12e');source.update(frame_count=n,rows=rows,pose_file=str(small),pose_sha256=sha(small));(bundle/'runtime.json').write_text(json.dumps(source,indent=2));(bundle/'projection_beta.txt').write_text((P/'bundle/projection_beta.txt').read_text())
 (bundle/'observations').mkdir();(bundle/'observations/manifest.json').write_text(json.dumps(dict(sequence='13',camera='P2',GT_runtime=False,note='Required preparation identity only; semantic bridge is off, no sidecars used.')))
 sys.path.insert(0,str(S23));code=(S23/'run_registered_matrix.py').read_text().replace('1000',str(n));ns={'__name__':'o3f_cell_adapter'};exec(compile(code,str(P/'prepare_cell_adapter.py'),'exec'),ns)
 cell=P/f'cell_{name}';ns['prepare_cell'](dict(stage='E3',sequence='13',source_start=780,frames=n,seed=0),bundle,cell,candidate=CAND)
 p=cell/'source/configs/kitti/B.py';local={};exec(p.read_text(),local);cfg=local['config'];cfg['feedback_policy']='ORB';cfg['factorial']['history_budget']=3;cfg['factorial']['seed']=0;cfg['eval_only_frames']=[];cfg['semantic'].update(T=False,M=False,D=False,labels=False,shuffle=False);cfg['save_checkpoints']=False;cfg['mapping']['use_gaussian_splatting_densification']=False
 assert cfg['mapping']['num_iters']==60; p.write_text('config = '+repr(cfg)+'\n')
 worker=cell/'scripts/worker.py';txt=worker.read_text();old="'dynamic_filter' if sf['D'] else 'semantic_only'";assert old in txt;worker.write_text(txt.replace(old,"'off'"))
 hook=cell/'scripts/factor_hooks.py';txt=hook.read_text();old="assert idx<=t and idx%20!=19";assert old in txt;hook.write_text(txt.replace(old,"assert idx<=t and idx not in cfg['eval_only_frames']"))
 reg=json.loads((cell/'registration.json').read_text());reg['config_sha256']['B']=sha(p);reg['scope']='B donor schedules itself, ORB/H3, cap3000, all frames used; no T/M/D or semantic labels';(cell/'registration.json').write_text(json.dumps(reg,indent=2))
 locks=[dict(path=x.relative_to(cell).as_posix(),sha256=sha(x)) for x in cell.rglob('*.py')];(cell/'source_locks.json').write_text(json.dumps(locks,indent=2))
 return cell
def check_mapping(cell,n,require_history):
 out=cell/'runs/B';receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and receipt['frames']==n and receipt['mapping_backward']==n*60
 rows=[json.loads(x) for x in (out/'native_addition.jsonl').read_text().splitlines()];assert len(rows)==n-1 and all(x['selected_count']<=3000 for x in rows)
 updates=[json.loads(x) for x in (out/'mapping_updates.jsonl').read_text().splitlines()];history=[x for x in updates if x['history_slot']];changed=[x for x in history if x['history_changed']];assert len(updates)==n*60 and all(x['target']<=x['frame']-200 and x['iteration'] in [15,35,55] for x in history)
 if require_history and not history:raise RuntimeError('H3 never triggered; cannot call the run O3F')
 report=dict(status='PASS',frames=n,addition_events=len(rows),max_selected=max((x['selected_count'] for x in rows),default=0),history_slots=len(history),history_changed=len(changed),history_first=min((x['frame'] for x in history),default=None),mapping_updates=len(updates),initial_points=receipt['initial_points'],final_gaussians=receipt['final_gaussians'],source_map_sha256=sha(out/'params.npz'))
 (cell/'engineering_check.json').write_text(json.dumps(report,indent=2));return report
def main():
 state('waiting_for_inputs')
 for _ in range(720):
  x=json.loads((P/'status.json').read_text())
  if x['stage']=='failed':raise RuntimeError('Input preparation failed; see status.json')
  if x['stage']=='inputs_complete':break
  time.sleep(30)
 else:raise TimeoutError('Input preparation did not finish within six hours')
 env=dict(os.environ,S23_RASTER_ROOT=str(RASTER),PYTHONPATH=str(RASTER),PYTORONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='backend:native')
 state('prepare_smoke');smoke=case(20,'smoke20');run('smoke_mapping',[PY,smoke/'scripts/worker.py','B'],env);smoke_check=check_mapping(smoke,20,False)
 run('smoke_f1',[PY,P/'refine.py','--cell',smoke,'--steps','20','--out',smoke/'f1_smoke'],env)
 assert json.loads((smoke/'f1_smoke/summary.json').read_text())['status']=='PASS'
 state('prepare_o3');cell=case(260,'O3');run('O3_mapping',[PY,cell/'scripts/worker.py','B'],env);check=check_mapping(cell,260,True)
 run('O3F_refinement',[PY,P/'refine.py','--cell',cell,'--steps','1040','--out',cell/'runs/O3F'],env);f=json.loads((cell/'runs/O3F/summary.json').read_text());assert f['status']=='PASS'
 state('complete',scope='260-frame O3F engineering/development smoke, seen views only',engineering_check=check,smoke_check=smoke_check,f1=f,selected_for_demo=False)
try:main()
except BaseException as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
