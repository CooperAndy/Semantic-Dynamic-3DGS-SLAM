"""Isolated MASS mapping and F1 for the selected seq07 and seq09 demo windows."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, traceback
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/seq07_seq09_mass_f1_20261116')
S23=Path('/root/autodl-tmp/s23_effects_20261105_p16247')
P03=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116')
PY='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'
RASTER=S23/'raster_fp64_accum_v1'
OLD09=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
R09=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114')
R07=Path('/root/autodl-tmp/seq07_000850_001049_tuning_20261115')
ENV=dict(os.environ,S23_RASTER_ROOT=str(RASTER),PYTHONPATH=f'{RASTER}:{R09}/S0/source',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='backend:native')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw):
 x=dict(stage=stage,time=time.strftime('%FT%T%z'),**kw); (ROOT/'status.tmp').write_text(json.dumps(x,indent=2)); (ROOT/'status.tmp').replace(ROOT/'status.json'); print(x,flush=True)
def run(name,cmd):
 if shutil.disk_usage(ROOT).free < 12*2**30: raise RuntimeError('Disk below 12 GiB; stopped with existing results preserved')
 status(name)
 with (ROOT/(name+'.log')).open('x') as f:subprocess.run(list(map(str,cmd)),cwd=S23,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def prepare_bundle(seq,n):
 start=850 if seq=='07' else 300; src=R07 if seq=='07' else OLD09
 depth=R07/'confidence/depth_confhard' if seq=='07' else R09/'f1_confidence_tuning/depth_50'
 conf=src/'confidence/confidence'; folder=src/'sequences'/seq; bundle=ROOT/f'bundle_{seq}_{n}';bundle.mkdir()
 mats={}
 for line in (folder/'calib.txt').read_text().splitlines():
  k,v=line.split(':',1);mats[k]=np.fromstring(v,sep=' ').reshape(3,4)
 p2,p3=mats['P2'],mats['P3']; (bundle/'projection_beta.txt').write_text(str(p2[0,3]-p3[0,3])); rows=[]
 W,H=Image.open(folder/'image_2/000000.png').size
 for i in range(n):
  paths=dict(rgb=folder/f'image_2/{i:06}.png',right=folder/f'image_3/{i:06}.png',depth=depth/f'{i:06}_depth.npy',confidence=conf/f'{i:06}_confidence.npy')
  for p in paths.values():assert p.is_file(),p
  d=np.load(paths['depth'],allow_pickle=False); c=np.load(paths['confidence'],allow_pickle=False); assert d.shape==c.shape==(H,W) and np.isfinite(d).all() and np.isfinite(c).all()
  rows.append(dict(frame_id=i,source_frame_id=start+i,**{k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()}))
 poses=np.loadtxt(src/'poses_estimated_camera2.txt'); assert poses.shape==(200,12); pose=bundle/'pose.txt';np.savetxt(pose,poses[:n],fmt='%.12e')
 runtime=dict(task_id=f'DEMO_{seq}_MASS',pose_source='ORB_estimate',pose_file=str(pose),pose_sha256=sha(pose),frame_convention='P2',frame_count=n,K=p2[:,:3].tolist(),H=H,W=W,depth_units='meters',valid_depth_range_m=[.5,40],confidence_threshold=.6 if seq=='07' else .5,GT_runtime=False,temporal_scope='existing Demo confidence; offline adjacent frames',rows=rows)
 (bundle/'runtime.json').write_text(json.dumps(runtime,indent=2));(bundle/'observations').mkdir();(bundle/'observations/manifest.json').write_text(json.dumps(dict(sequence=seq,camera='P2',GT_runtime=False,note='semantic bridge off')))
 return bundle,start
def prepare_cell(seq,n,tag):
 bundle,start=prepare_bundle(seq,n);sys.path.insert(0,str(S23));code=(S23/'run_registered_matrix.py').read_text().replace('1000',str(n));ns={'__name__':'demo_adapter'};exec(compile(code,'demo_adapter.py','exec'),ns)
 cell=ROOT/f'cell_{seq}_{tag}';ns['prepare_cell'](dict(stage='E3',sequence=seq,source_start=start,frames=n,seed=42),bundle,cell,candidate=P03/'candidate')
 p=cell/'source/configs/kitti/B.py';local={};exec(p.read_text(),local);cfg=local['config'];cfg['feedback_policy']='MASS';cfg['tracking']['camera_policy']='external_init_refine';cfg['factorial'].update(history_budget=3,seed=42);cfg['eval_only_frames']=[];cfg['semantic'].update(T=False,M=False,D=False,labels=False,shuffle=False);cfg['save_checkpoints']=False;cfg['mapping']['use_gaussian_splatting_densification']=False;cfg['mapping']['sampling_seed']=42;cfg['seed']=42
 assert cfg['mapping']['num_iters']==60 and not cfg['mapping'].get('depth_confidence_enabled',False);p.write_text('config = '+repr(cfg)+'\n')
 for filename,old,new in [('worker.py',"'dynamic_filter' if sf['D'] else 'semantic_only'","'off'"),('factor_hooks.py',"assert idx<=t and idx%20!=19","assert idx<=t and idx not in cfg['eval_only_frames']"),('stereo_proposal.py','   if anchor%20==19:anchor-=1','   # all frames contribute to the display map')]:
  q=cell/'scripts'/filename;t=q.read_text();assert old in t;q.write_text(t.replace(old,new))
 reg=json.loads((cell/'registration.json').read_text());reg['config_sha256']['B']=sha(p);(cell/'registration.json').write_text(json.dumps(reg,indent=2));(cell/'source_locks.json').write_text(json.dumps([dict(path=x.relative_to(cell).as_posix(),sha256=sha(x)) for x in cell.rglob('*.py')],indent=2))
 return cell
def check(cell,n):
 out=cell/'runs/B'; r=json.loads((out/'receipt.json').read_text());assert r['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and r['frames']==n and r['mapping_backward']==n*60
 events=[json.loads(x) for x in (out/'backend_events.jsonl').read_text().splitlines()];assert len(events)==n and all(x['policy']=='MASS' for x in events)
 adds=[json.loads(x) for x in (out/'native_addition.jsonl').read_text().splitlines()];assert len(adds)==n-1 and all(x['selected_count']<=3000 for x in adds)
 updates=[json.loads(x) for x in (out/'mapping_updates.jsonl').read_text().splitlines()];hist=[x for x in updates if x['history_slot']]
 report=dict(frames=n,accepted=sum(x['accepted'] is True for x in events),attempts=n-1,history_slots=len(hist),max_added=max((x['selected_count'] for x in adds),default=0),map_sha256=sha(out/'params.npz'));(cell/'check.json').write_text(json.dumps(report,indent=2));return report
def main():
 ROOT.mkdir(exist_ok=False);status('preflight');assert shutil.disk_usage(ROOT).free>20*2**30
 for seq in ['07','09']:
  smoke=prepare_cell(seq,4,'smoke');run(f'{seq}_smoke_map',[PY,smoke/'scripts/worker.py','B']);check(smoke,4);run(f'{seq}_smoke_f1',[PY,P03/'refine.py','--cell',smoke,'--steps','4','--out',smoke/'f1']);assert json.loads((smoke/'f1/summary.json').read_text())['status']=='PASS'
  cell=prepare_cell(seq,200,'full');run(f'{seq}_mapping',[PY,cell/'scripts/worker.py','B']); result=check(cell,200)
  run(f'{seq}_f1',[PY,P03/'refine.py','--cell',cell,'--steps','800','--out',cell/'runs/F1']);f=json.loads((cell/'runs/F1/summary.json').read_text());assert f['status']=='PASS' and f['frames']==200
  # Original evaluation script uses original ORB camera poses for common-view comparisons.
  evaluator=R07/'scripts/evaluate.py' if seq=='07' else R09/'f1_confidence_tuning/evaluate_f1_candidates.py'; dest=ROOT/f'eval_{seq}'
  for name,mapfile in [('mass_base',cell/'runs/B/params.npz'),('mass_f1',cell/'runs/F1/params.npz')]:
   run(f'{seq}_eval_{name}',[PY,evaluator,'--name',name,'--map',mapfile,'--out',dest])
  (ROOT/f'{seq}_result.json').write_text(json.dumps(dict(mapping=result,f1=f,common_camera='original ORB; same camera as existing display-candidate scores',scope='all frames seen; comparison across complete configurations; no semantic or dynamic filtering'),indent=2))
 status('complete')
try:main()
except BaseException as e:
 if ROOT.exists():status('failed',error=repr(e),traceback=traceback.format_exc())
 raise
