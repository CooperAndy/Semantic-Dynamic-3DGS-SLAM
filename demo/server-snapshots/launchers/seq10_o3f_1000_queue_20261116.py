"""Reproducible seq10 ORB/H3 baseline paired with the completed M3F run."""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
import numpy as np
P=Path('/root/autodl-tmp/seq10_m3f_20261116');R=Path('/root/autodl-tmp/seq10_o3f_1000_20261116');S23=Path('/root/autodl-tmp/s23_effects_20261105_p16247');P03=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116');RASTER=S23/'raster_fp64_accum_v1';PY='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python';M=P/'cell_full'
ENV=dict(os.environ,S23_RASTER_ROOT=str(RASTER),PYTHONPATH=f'{RASTER}:{M}/source',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='backend:native')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def state(stage,**kw):
 a=dict(stage=stage,time=time.strftime('%FT%T%z'),pid=os.getpid(),**kw);p=R/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2));t.replace(p);print(a,flush=True)
def gpu():
 x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines();return [int(a.strip()) for a in x if a.strip().isdigit()]
def ready(stage):
 for _ in range(720):
  free=shutil.disk_usage(R).free
  if free<12*2**30:raise RuntimeError(f'Free disk below 12 GiB: {free/2**30:.2f}')
  pids=gpu()
  if not pids:return
  state('waiting_gpu',next_stage=stage,foreign_pids=pids);time.sleep(30)
 raise TimeoutError('GPU remained occupied for six hours')
def run(name,cmd):
 ready(name);state(name)
 with (R/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(list(map(str,cmd)),cwd=S23,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   try:
    free=shutil.disk_usage(R).free
    if free<10*2**30:raise RuntimeError(f'Free disk below 10 GiB: {free/2**30:.2f}')
   except BaseException:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
   time.sleep(10)
  if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see log')
def verify():
 status=json.loads((P/'status.json').read_text());assert status['stage']=='complete'
 assert sha(M/'runs/B/params.npz')==status['mapping']['map_sha256'] and sha(M/'runs/M3F/params.npz')==status['f1_sha256']
 source=json.loads((P/'bundle/runtime.json').read_text());current=json.loads((M/'inputs/runtime/input_manifest.json').read_text());assert source['frame_count']==current['frame_count']==1000 and source['rows']==current['rows'] and source['GT_runtime'] is False
 assert sha(source['pose_file'])==source['pose_sha256']
 for row in source['rows']:
  for k in ('rgb','right','depth','confidence'):assert sha(row[k])==row['hashes'][k],(row['frame_id'],k)
 (R/'input_receipt.json').write_text(json.dumps(dict(status='PASS',frames=1000,matched_M3_rows_and_hashes=True,pose_sha256=source['pose_sha256'],scope='same offline Demo confidence .60, seed 0, no GT'),indent=2))
def case(n,tag):
 source=json.loads((P/'bundle/runtime.json').read_text());bundle=R/f'bundle_{tag}';bundle.mkdir();pose=np.loadtxt(source['pose_file']);assert pose.shape==(1000,12);small=bundle/'pose.txt';np.savetxt(small,pose[:n],fmt='%.12e');source.update(frame_count=n,rows=source['rows'][:n],pose_file=str(small),pose_sha256=sha(small));(bundle/'runtime.json').write_text(json.dumps(source,indent=2));shutil.copy2(P/'bundle/projection_beta.txt',bundle/'projection_beta.txt');shutil.copytree(P/'bundle/observations',bundle/'observations')
 sys.path.insert(0,str(S23));code=(S23/'run_registered_matrix.py').read_text().replace('1000',str(n));ns={'__name__':'seq10_o3_adapter'};exec(compile(code,'seq10_o3_adapter.py','exec'),ns);cell=R/f'cell_{tag}';ns['prepare_cell'](dict(stage='E3',sequence='10',source_start=0,frames=n,seed=0),bundle,cell,candidate=P03/'candidate')
 p=cell/'source/configs/kitti/B.py';ns={};exec(p.read_text(),ns);c=ns['config'];c['feedback_policy']='ORB';c['tracking']['camera_policy']='external_fixed';c['factorial'].update(history_budget=3,seed=0);c['eval_only_frames']=[];c['semantic'].update(T=False,M=False,D=False,labels=False,shuffle=False);c['save_checkpoints']=False;c['mapping']['use_gaussian_splatting_densification']=False;c['mapping']['sampling_seed']=0;c['seed']=0
 assert c['mapping']['num_iters']==60 and not c['mapping'].get('depth_confidence_enabled',False);p.write_text('config = '+repr(c)+'\n')
 for fn,old,new in [('worker.py',"'dynamic_filter' if sf['D'] else 'semantic_only'","'off'"),('factor_hooks.py',"assert idx<=t and idx%20!=19","assert idx<=t and idx not in cfg['eval_only_frames']"),('stereo_proposal.py','   if anchor%20==19:anchor-=1','   # all frames are mapper frames')]:
  q=cell/'scripts'/fn;t=q.read_text();assert old in t;q.write_text(t.replace(old,new))
 reg=json.loads((cell/'registration.json').read_text());reg['config_sha256']['B']=sha(p);(cell/'registration.json').write_text(json.dumps(reg,indent=2));(cell/'source_locks.json').write_text(json.dumps([dict(path=x.relative_to(cell).as_posix(),sha256=sha(x)) for x in cell.rglob('*.py')],indent=2));return cell
def check(c,n):
 out=c/'runs/B';r=json.loads((out/'receipt.json').read_text());assert r['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and r['frames']==n and r['mapping_backward']==n*60
 e=[json.loads(x) for x in (out/'backend_events.jsonl').read_text().splitlines()];a=[json.loads(x) for x in (out/'native_addition.jsonl').read_text().splitlines()];u=[json.loads(x) for x in (out/'mapping_updates.jsonl').read_text().splitlines()];h=[x for x in u if x['history_slot']];assert len(e)==n and all(x['policy']=='ORB' for x in e) and len(a)==n-1 and all(x['selected_count']<=3000 for x in a)
 if n==1000:assert h
 result=dict(status='PASS',frames=n,history_slots=len(h),history_changed=sum(x['history_changed'] for x in h),max_added=max(x['selected_count'] for x in a),gaussians=r['final_gaussians'],map_sha256=sha(out/'params.npz'));(c/'check.json').write_text(json.dumps(result,indent=2));return result
def prepare_compare(c):
 src=(P/'compare.py').read_text();src=src.replace("O=R/'comparison20';O.mkdir(exist_ok=False)","O=Path('/root/autodl-tmp/seq10_o3f_1000_20261116/comparison20');O.mkdir(exist_ok=False)")
 a=src.index('cases=[');b=src.index(';results={}',a)
 cases="cases=[('o3_common',Path('/root/autodl-tmp/seq10_o3f_1000_20261116/cell_full/runs/B/params.npz'),orb),('o3f_common',Path('/root/autodl-tmp/seq10_o3f_1000_20261116/cell_full/runs/O3F/params.npz'),orb)]"
 src=src[:a]+cases+src[b:];(R/'compare_o3.py').write_text(src)
def main():
 R.mkdir(exist_ok=False);lock=(R/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state('preflight');verify()
 sm=case(20,'smoke');run('smoke_mapping',[PY,sm/'scripts/worker.py','B']);check(sm,20);run('smoke_f1',[PY,P03/'refine.py','--cell',sm,'--steps',20,'--out',sm/'f1']);assert json.loads((sm/'f1/summary.json').read_text())['status']=='PASS'
 c=case(1000,'full');run('mapping',[PY,c/'scripts/worker.py','B']);m=check(c,1000);run('f1',[PY,P03/'refine.py','--cell',c,'--steps',4000,'--out',c/'runs/O3F']);f=json.loads((c/'runs/O3F/summary.json').read_text());assert f['status']=='PASS' and f['frames']==1000 and f['steps']==4000
 prepare_compare(c);run('comparison',[PY,R/'compare_o3.py']);a=json.loads((R/'comparison20/results.json').read_text());b=json.loads((P/'comparison20/results.json').read_text());assert a['frames']==b['frames'] and a['status']==b['status']=='COMPLETE';combined=dict(status='COMPLETE',scope='same 20 seen views, common ORB camera; selected camera rows separate; old ORB candidate differs in point budget and resume history',frames=a['frames'],results={**a['results'],**b['results']},o3=m,o3f=f);(R/'combined_results.json').write_text(json.dumps(combined,indent=2));state('complete',mapping=m,f1_sha256=f['output_map_sha256'],comparison=str(R/'combined_results.json'))
try:main()
except BaseException as e:
 if R.exists():state('failed',error=repr(e),traceback=traceback.format_exc())
 raise
