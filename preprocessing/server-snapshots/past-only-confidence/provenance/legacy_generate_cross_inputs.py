"""Generate fresh08 ORB/RAFT/E6a inputs after current GPU batch finishes."""
from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,csv,shutil
import numpy as np
R=Path('/root/autodl-tmp/stage3_g14_20261027_p30915/data_pipeline')
PY='/root/autodl-tmp/stage3_g05_runtime_20261025/venv/bin/python'
reg=json.loads((R/'input_registration.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'input_result.json').exists()
for row in reg['rows']:
 for cam in ['image_2','image_3']:assert sha(Path(row[cam]['source']))==row[cam]['sha256']
for row in reg['pipeline_sources']:assert sha(Path(row['copied_path']))==row['copied_sha256']
for row in reg.get('raft_core_file_locks',[]):assert sha(Path(row['path']))==row['sha256']
assert sha(Path(reg['checkpoint']))==reg['checkpoint_sha256']
assert sha(R/'pipeline/stereo_kitti_headless')==reg['binary_sha256']
assert shutil.disk_usage(R).free>8*2**30
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled')
orb_env=env.copy();orb_env['LD_LIBRARY_PATH']=':'.join(reg['library_paths'])
commands=[];started=time.monotonic();deadline=started+reg['input_preparation_budget_seconds']
def run(label,cmd,limit,cwd=None):
 result=dict(label=label,command=cmd,started_at=datetime.datetime.now().astimezone().isoformat())
 with (R/'operator_logs'/f'{label}.log').open('xb') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=orb_env if label=='ORB' else env,timeout=min(limit,max(1,deadline-time.monotonic())))
 result.update(exit_code=p.returncode,elapsed_total_seconds=time.monotonic()-started);commands.append(result)
 (R/'input_commands.json').write_text(json.dumps(commands,indent=2));assert p.returncode==0,(label,p.returncode)
 print(json.dumps(result),flush=True)
try:
 assert shutil.which('strace')
 cmd=['strace','-f','-e','trace=openat','-o',str(R/'orb/file_open_audit.log'),str(R/'pipeline/stereo_kitti_headless'),reg['vocabulary'],str(R/'orb/settings.yaml'),str(R/'input_view'),'O0','-','-','08',str(R/'orb/frame_features.csv')]
 run('ORB',cmd,900,cwd=R/'orb')
 opens=(R/'orb/file_open_audit.log').read_text(errors='replace')
 assert not any(s in opens for s in ['/poses/','groundtruth.txt','/evaluation_only/'])
 raw=np.loadtxt(R/'orb/CameraTrajectory.txt',dtype=np.float64);assert raw.shape==(1000,12) and np.isfinite(raw).all()
 T=np.tile(np.eye(4),(1000,1,1));T[:,:3]=raw.reshape(1000,3,4)
 assert np.max(np.abs(T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]-np.eye(3)))<2e-5
 normalized=np.linalg.inv(T[0])@T;np.savetxt(R/'orb/external_P2_c2w.txt',normalized[:,:3].reshape(1000,12),fmt='%.9f')
 common=['--raft-root',reg['raft_root'],'--checkpoint',reg['checkpoint'],'--sequence-dir',str(R/'input_view'),'--num-frames','1000','--valid-iters','32','--min-depth','0.5','--max-depth','80']
 run('RAFT_LEFT',[PY,str(R/'pipeline/left.py'),*common,'--output-dir',str(R/'raw_left'),'--preview-indices','0','99','199'],600)
 run('RAFT_RIGHT',[PY,str(R/'pipeline/right.py'),*common,'--output-dir',str(R/'raw_right')],600)
 run('CONFIDENCE',[PY,str(R/'pipeline/confidence.py'),'--left-image-dir',str(R/'input_view/image_2'),'--right-image-dir',str(R/'input_view/image_3'),
  '--depth-left-dir',str(R/'raw_left'),'--depth-right-dir',str(R/'raw_right'),'--pose-path',str(R/'orb/external_P2_c2w.txt'),'--calib',str(R/'input_view/calib.txt'),
  '--output-root',str(R/'cache'),'--num-frames','1000','--min-depth','.5','--max-depth','40','--seed-threshold','.80','--loss-threshold','.20'],300)
 rows=[]
 for i,source in enumerate(reg['rows']):
  paths=dict(rgb=R/f'input_view/image_2/{i:06d}.png',right=R/f'input_view/image_3/{i:06d}.png',depth=R/f'cache/depth_confhard/{i:06d}_depth.npy',confidence=R/f'cache/confidence/{i:06d}_confidence.npy')
  d=np.load(paths['depth']);c=np.load(paths['confidence'])
  assert d.dtype==c.dtype==np.float32 and d.shape==c.shape==tuple(reg['shape']) and np.isfinite(d).all() and np.isfinite(c).all()
  assert np.all((d==0)|((d>=.5)&(d<=40))) and np.all(c[d>0]>=.8)
  rows.append(dict(frame_id=i,source_frame_id=source['source_frame_id'],**{k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()}))
 m=dict(task_id='T3-G14-INPUT',pose_source='ORB_estimate',pose_file=str(R/'orb/external_P2_c2w.txt'),pose_sha256=sha(R/'orb/external_P2_c2w.txt'),
  frame_convention='P2 c2w, relative to window firstP2, metres',frame_count=1000,K=reg['K'],H=reg['shape'][0],W=reg['shape'][1],
  depth_units='metres',valid_depth_range_m=[.5,80],confidence_threshold=.8,GT_runtime=False,
  temporal_scope='Cached replay; ORB export after window optimization and confidence uses t-1/t+1; not strictly online',rows=rows)
 for row in reg.get('raft_core_file_locks',[]):assert sha(Path(row['path']))==row['sha256']
 (R/'input_manifest.json').write_text(json.dumps(m,indent=2))
 result=dict(status='PASS_FRESH_GT_FREE_CROSS_WINDOW_INPUTS',frames=1000,hashes=4*len(rows),seconds=time.monotonic()-started,GT_file_opens=0,
  pose_sha256=m['pose_sha256'],input_manifest_sha256=sha(R/'input_manifest.json'),window=reg['window'])
except BaseException as e:
 result=dict(status='FAILED_INPUT_PREPARATION_PRESERVED',error=repr(e),commands=commands,seconds=time.monotonic()-started)
 (R/'input_result.json').write_text(json.dumps(result,indent=2));raise
else:
 (R/'input_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
