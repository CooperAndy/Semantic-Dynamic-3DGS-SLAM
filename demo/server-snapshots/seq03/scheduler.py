"""Isolated ORB/H3 cap3000 run: 20-frame smoke, then 260-frame O3 and F1."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
P=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116');S23=Path('/root/autodl-tmp/s23_effects_20261105_p16247');CAND=P/'candidate';RASTER=S23/'raster_fp64_accum_v1';PY='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def state(s,**kw):
 p=P/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stage=s,time=time.strftime('%FT%T%z'),**kw),indent=2));t.replace(p);print(s,kw,flush=True)
def run(name,cmd,env):
 if shutil.disk_usage(P).free<10*2**30:raise RuntimeError('Free disk below 10 GiB; stop and retain outputs')
 state(name)
 with (P/f'{name}.log').open('x') as f:subprocess.run(list(map(str,cmd)),cwd=S23,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def case(n,name,policy):
 source=json.loads((P/'bundle/runtime.json').read_text());bundle=P/f'bundle_{name}';bundle.mkdir();rows=source['rows'][:n];assert len(rows)==n
 # ManifestDataset checks that pose file length exactly equals frame_count.
 pose=np.loadtxt(source['pose_file']);assert pose.shape==(801,12);small=bundle/'estimated_p2.txt';np.savetxt(small,pose[:n],fmt='%.12e');source.update(frame_count=n,rows=rows,pose_file=str(small),pose_sha256=sha(small));(bundle/'runtime.json').write_text(json.dumps(source,indent=2));(bundle/'projection_beta.txt').write_text((P/'bundle/projection_beta.txt').read_text())
 (bundle/'observations').mkdir();(bundle/'observations/manifest.json').write_text(json.dumps(dict(sequence='13',camera='P2',GT_runtime=False,note='Required preparation identity only; semantic bridge is off, no sidecars used.')))
 sys.path.insert(0,str(S23));code=(S23/'run_registered_matrix.py').read_text().replace('1000',str(n));ns={'__name__':'o3f_cell_adapter'};exec(compile(code,str(P/'prepare_cell_adapter.py'),'exec'),ns)
 cell=P/f'cell_{name}';ns['prepare_cell'](dict(stage='E3',sequence='03',source_start=0,frames=n,seed=42),bundle,cell,candidate=CAND)
 p=cell/'source/configs/kitti/B.py';local={};exec(p.read_text(),local);cfg=local['config'];cfg['feedback_policy']=policy;cfg['tracking']['camera_policy']='external_fixed' if policy=='ORB' else 'external_init_refine';cfg['factorial']['history_budget']=3;cfg['factorial']['seed']=42;cfg['eval_only_frames']=[];cfg['semantic'].update(T=False,M=False,D=False,labels=False,shuffle=False);cfg['save_checkpoints']=False;cfg['mapping']['use_gaussian_splatting_densification']=False;cfg['mapping']['sampling_seed']=42;cfg['seed']=42
 assert not cfg['mapping'].get('depth_confidence_enabled',False), 'Demo .60 is already applied to mapper hard depth' 
 assert cfg['mapping']['num_iters']==60; p.write_text('config = '+repr(cfg)+'\n')
 worker=cell/'scripts/worker.py';txt=worker.read_text();old="'dynamic_filter' if sf['D'] else 'semantic_only'";assert old in txt;worker.write_text(txt.replace(old,"'off'"))
 hook=cell/'scripts/factor_hooks.py';txt=hook.read_text();old="assert idx<=t and idx%20!=19";assert old in txt;hook.write_text(txt.replace(old,"assert idx<=t and idx not in cfg['eval_only_frames']"))
 stereo=cell/'scripts/stereo_proposal.py';txt=stereo.read_text();old='   if anchor%20==19:anchor-=1';assert old in txt;stereo.write_text(txt.replace(old,'   # All frames are mapping frames: use the backend-selected preceding frame.'))
 reg=json.loads((cell/'registration.json').read_text());reg['config_sha256']['B']=sha(p);reg['scope']='B donor schedules itself, fixed ORB or MASS/H3, cap3000, all frames used; no T/M/D or semantic labels; causal stereo anchor follows actual holdout policy';(cell/'registration.json').write_text(json.dumps(reg,indent=2))
 locks=[dict(path=x.relative_to(cell).as_posix(),sha256=sha(x)) for x in cell.rglob('*.py')];(cell/'source_locks.json').write_text(json.dumps(locks,indent=2))
 return cell
def check_mapping(cell,n,require_history):
 out=cell/'runs/B';receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and receipt['frames']==n and receipt['mapping_backward']==n*60
 rows=[json.loads(x) for x in (out/'native_addition.jsonl').read_text().splitlines()];assert len(rows)==n-1 and all(x['selected_count']<=3000 for x in rows)
 updates=[json.loads(x) for x in (out/'mapping_updates.jsonl').read_text().splitlines()];history=[x for x in updates if x['history_slot']];changed=[x for x in history if x['history_changed']];assert len(updates)==n*60 and all(x['target']<=x['frame']-200 and x['iteration'] in [15,35,55] for x in history)
 if require_history and not history:raise RuntimeError('H3 never triggered; cannot call the run O3F')
 report=dict(status='PASS',frames=n,addition_events=len(rows),max_selected=max((x['selected_count'] for x in rows),default=0),history_slots=len(history),history_changed=len(changed),history_first=min((x['frame'] for x in history),default=None),mapping_updates=len(updates),initial_points=receipt['initial_points'],final_gaussians=receipt['final_gaussians'],source_map_sha256=sha(out/'params.npz'))
 events=[json.loads(x) for x in (out/'backend_events.jsonl').read_text().splitlines()];assert len(events)==n and all(x['policy'] in ('ORB','MASS') for x in events);report['MASS_accepted']=sum(x['accepted'] is True for x in events);report['MASS_attempts']=sum(x['frame']>0 for x in events);report['MASS_accepted_frames']=[x['frame'] for x in events if x['accepted'] is True];(cell/'engineering_check.json').write_text(json.dumps(report,indent=2));return report


# Independent server-side scheduler. It never changes or signals unrelated jobs.
import fcntl,signal
PREV=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115')
QUEUE_PID=os.getpid(); CURRENT=None
MIN_START_DISK=22*2**30; MIN_RUNNING_DISK=12*2**30; MIN_START_GPU_FREE=18*1024; MIN_RUNNING_GPU_FREE=2048

def gpu_info():
 cmd=['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=True);lines=r.stdout.strip().splitlines();assert len(lines)==1;free,util=map(int,lines[0].split(','))
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True);pids=[int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()];return free,util,pids

def same_previous():
 p=Path('/proc/100701/stat')
 if not p.exists():return False
 try:return int(p.read_text().split()[21])==1419255638
 except (OSError,IndexError,ValueError):raise RuntimeError('Cannot establish prior process identity')

def predecessor():
 state('waiting_previous',reason='Waiting for seq13 M3F controller PID 100701 to exit and write complete receipts',queue_pid=QUEUE_PID,previous_pid=100701)
 while same_previous():time.sleep(15)
 status=json.loads((PREV/'m3f_status.json').read_text());
 if status.get('stage')!='complete':raise RuntimeError('Previous seq13 M3F failed or final status cannot be established: '+str(status.get('stage')))
 mapcheck=json.loads((PREV/'cell_M3/engineering_check.json').read_text());f1=json.loads((PREV/'cell_M3/runs/M3F/summary.json').read_text());comparison=json.loads((PREV/'compare_m3f_o3f/results.json').read_text())
 if not(mapcheck.get('status')==f1.get('status')=='PASS' and comparison.get('status')=='COMPLETE'):raise RuntimeError('Previous completion receipts are missing or incomplete')
 (P/'previous_task_receipt.json').write_text(json.dumps(dict(result='complete',observed_pid=100701,observed_startticks=1419255638,controller='/root/miniconda3/bin/python -u m3f_controller.py',cwd=str(PREV),status_path=str(PREV/'m3f_status.json'),status_sha256=sha(PREV/'m3f_status.json'),mapping_receipt_sha256=sha(PREV/'cell_M3/engineering_check.json'),f1_receipt_sha256=sha(PREV/'cell_M3/runs/M3F/summary.json'),comparison_sha256=sha(PREV/'compare_m3f_o3f/results.json')),indent=2))

def ready(stage):
 while True:
  free,util,others=gpu_info();disk=shutil.disk_usage(P).free
  if disk<MIN_START_DISK:raise RuntimeError(f'Disk {disk/2**30:.2f} GiB < start requirement 22 GiB; stop')
  if not others and free>=MIN_START_GPU_FREE and util<=5:return
  state('waiting_resources',reason=f'{stage}: GPU free={free} MiB, utilization={util}%, other compute pids={others}; require no other compute jobs, >=18432 MiB free and <=5% utilization',queue_pid=QUEUE_PID)
  time.sleep(30)

def run(name,cmd,env):
 ready(name);state(name,queue_pid=QUEUE_PID,reason='launching')
 with (P/f'{name}.log').open('x') as log:
  child=subprocess.Popen(list(map(str,cmd)),cwd=S23,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   try:
    disk=shutil.disk_usage(P).free;free,util,pids=gpu_info();foreign=[]
    for pid in pids:
     try:
      if os.getpgid(pid)!=child.pid:foreign.append(pid)
     except ProcessLookupError:pass
    if disk<MIN_RUNNING_DISK or free<MIN_RUNNING_GPU_FREE or foreign:
     reason=f'Insufficient resources or competing GPU job: disk_GiB={disk/2**30:.2f}, gpu_free_MiB={free}, foreign_gpu_pids={foreign}; terminated only this queue stage'
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise RuntimeError(reason)
    state(name,queue_pid=QUEUE_PID,child_pid=child.pid,reason='running',disk_free_gib=round(disk/2**30,2),gpu_free_mib=free,gpu_utilization=util)
   except BaseException:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
   time.sleep(10)
  if child.returncode!=0:raise RuntimeError(f'{name} exited {child.returncode}; see {P/(name+".log")}')

def verify():
 stage=json.loads((P/'staging_receipt.json').read_text());assert stage['status']=='PASS' and stage['frames']==801 and stage['stereo_images']==1602
 locks=json.loads((P/'dependency_sha256.json').read_text());assert sha(P/'dependency_sha256.json')==stage['dependency_manifest_sha256']
 for p,h in locks.items():
  if sha(p)!=h:raise RuntimeError('Changed dependency: '+p)
 for name,h in json.loads((P/'launch_sha256.json').read_text()).items():
  if sha(P/name)!=h:raise RuntimeError('Changed queued script: '+name)
 # Exact Demo producer, not the G19C past-only formula. The latter is not used.
 demo=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499/scripts/build_depth_confidence_kitti.py');local=P/'scripts/build_depth_confidence_kitti.py'
 if sha(demo)!=sha(local):raise RuntimeError('Confidence producer differs from the Demo source; formal runs blocked')
 (P/'confidence_preflight.json').write_text(json.dumps(dict(status='PASS_SOURCE_IDENTITY',source_sha256=sha(demo),seed_threshold=.60,application='hard depth before mapper initialization and birth; depth confidence weighting disabled',MASS_internal_proposal_gate=.8,temporal_scope='offline previous and next; not strictly causal',final_gate='all 801 hard-depth masks validated before any formal mapping'),indent=2))

def validate_confidence():
 s=json.loads((P/'input_status.json').read_text());audit=json.loads((P/'bundle/confidence_audit.json').read_text());m=json.loads((P/'bundle/runtime.json').read_text())
 assert s['stage']=='inputs_complete' and audit['status']=='PASS' and audit['frames']==801 and audit['seed_threshold']==.6 and m['frame_count']==801 and m['GT_runtime'] is False
 assert audit['confidence_script_sha256']==sha(P/'scripts/build_depth_confidence_kitti.py') and all(x['frame_id']==i and x['source_frame_id']==i for i,x in enumerate(m['rows']))

# Validate policies and logging before any of the long runs are adopted.
def validate_mass(cell,n):
 out=cell/'runs/B';rows=[json.loads(x) for x in (out/'backend_events.jsonl').read_text().splitlines()];assert len(rows)==n and all(x['policy']=='MASS' for x in rows)
 for row in rows[1:]:
  assert 'accepted' in row and 'reason' in row and 'geometry_ok' in row and 'candidate_w2c' in row and 'proposal_path' in row
  assert (out/row['proposal_path']).is_file()

def write_report():
 comp=json.loads((P/'compare_m3f_o3f/results.json').read_text());assert comp['status']=='COMPLETE';outputs={}
 for name in ['O3','O3F','M3','M3F']:
  cell=P/('cell_O3' if name.startswith('O') else 'cell_M3');folder=cell/'runs'/('B' if len(name)==2 else name);f=folder/'params.npz';assert f.is_file();outputs[name]=dict(params=str(f),sha256=sha(f),bytes=f.stat().st_size)
  report=json.loads((folder/('receipt.json' if len(name)==2 else 'summary.json')).read_text());outputs[name]['run']=report
 result=dict(status='COMPLETE',frames=[0,800],seed=42,scope='full-sequence display/development; F1 uses all frames, no held-out test; confidence includes adjacent future frame',outputs=outputs,metrics=comp,previous_task=str(P/'previous_task_receipt.json'),confidence_audit=str(P/'bundle/confidence_audit.json'))
 (P/'final_report.json').write_text(json.dumps(result,indent=2));return result

def main():
 lock=(P/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 predecessor();state('preflight',queue_pid=QUEUE_PID);verify()
 env=dict(os.environ,S23_RASTER_ROOT=str(RASTER),PYTHONPATH=str(RASTER),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='backend:native')
 run('prepare_inputs',[PY,P/'prepare.py'],env);validate_confidence()
 # 20-frame MASS smoke checks proposal production, pose selection, F1 gradients and invariants.
 s20=case(20,'smoke20','MASS');run('smoke20_mapping',[PY,s20/'scripts/worker.py','B'],env);r20=check_mapping(s20,20,False);validate_mass(s20,20);run('smoke20_f1',[PY,P/'refine.py','--cell',s20,'--steps',20,'--out',s20/'f1_smoke'],env)
 # 260-frame MASS smoke verifies legal H3 targets and real replay, before formal runs.
 s260=case(260,'smoke260','MASS');run('smoke260_mapping',[PY,s260/'scripts/worker.py','B'],env);r260=check_mapping(s260,260,True);validate_mass(s260,260);run('smoke260_f1',[PY,P/'refine.py','--cell',s260,'--steps',20,'--out',s260/'f1_smoke'],env)
 assert json.loads((s20/'f1_smoke/summary.json').read_text())['status']==json.loads((s260/'f1_smoke/summary.json').read_text())['status']=='PASS';(P/'smoke_receipt.json').write_text(json.dumps(dict(status='PASS',smoke20=r20,smoke260=r260),indent=2))
 o=case(801,'O3','ORB');run('O3_mapping',[PY,o/'scripts/worker.py','B'],env);oc=check_mapping(o,801,True)
 run('O3F_refinement',[PY,P/'refine.py','--cell',o,'--steps',3204,'--out',o/'runs/O3F'],env);assert json.loads((o/'runs/O3F/summary.json').read_text())['status']=='PASS'
 m=case(801,'M3','MASS');run('M3_mapping',[PY,m/'scripts/worker.py','B'],env);mc=check_mapping(m,801,True);validate_mass(m,801)
 orc=json.loads((o/'runs/B/receipt.json').read_text());mrc=json.loads((m/'runs/B/receipt.json').read_text());assert orc['initial_gaussian_sha256']==mrc['initial_gaussian_sha256'],'Initial Gaussian sets differ'
 run('M3F_refinement',[PY,P/'refine.py','--cell',m,'--steps',3204,'--out',m/'runs/M3F'],env);assert json.loads((m/'runs/M3F/summary.json').read_text())['status']=='PASS'
 run('evaluate_and_preview',[PY,P/'compare.py'],env);report=write_report();state('complete',queue_pid=QUEUE_PID,report=str(P/'final_report.json'),MASS_accepted=mc['MASS_accepted'],O3_H3_slots=oc['history_slots'],M3_H3_slots=mc['history_slots'])
try:main()
except BaseException as e:
 state('failed',queue_pid=QUEUE_PID,error=repr(e),traceback=traceback.format_exc(),reason='Stopped without deleting other files');raise
