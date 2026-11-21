"""Rebuild seq09 Demo assets for the selected confidence .50 seed 42 map."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,traceback
import numpy as np
OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
ROOT=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114')
B=ROOT/'seed42_demo_build'; A=B/'annotations_20261107'; PY='/root/miniconda3/bin/python'; SPLAT=Path('/root/autodl-tmp/SplaTAM')
M=ROOT/'f1_confidence_tuning/c50_s42/map/params.npz'; F=ROOT/'f1_confidence_tuning/c50_s42/F1/params.npz'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw): (B/'build_status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
def run(name,cmd,gpu=False):
 status(name)
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TASK_GPU_GIB='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=f'{B}/scripts:{A}/scripts:{SPLAT}')
 if gpu:cmd=[cmd[0],str(B/'scripts/resource_guard.py'),*cmd[1:]]
 with (B/f'{name}.log').open('x') as f:subprocess.run(list(map(str,cmd)),cwd=str(SPLAT),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 B.mkdir(exist_ok=False); (B/'scripts').mkdir(); (A/'scripts').mkdir(parents=True); (B/'runs/F1_appearance800').mkdir(parents=True); (B/'runs/F200_orb_raft_conf060').mkdir(parents=True)
 for v,p in [('F1_appearance800',F),('F200_orb_raft_conf060',M)]: (B/'runs'/v/'params.npz').symlink_to(p)
 for name in ['sequences','raft_left','poses_estimated_camera2.txt','input_manifest.csv']:(B/name).symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())
 for name in ['semantic','instances','motion']:(A/name).symlink_to(OLD/'annotations_20261107'/name,target_is_directory=True)
 for p in (OLD/'scripts').glob('*.py'):
  text=p.read_text(); text=text.replace(str(OLD),str(B)).replace('confidence>=0.60','confidence>=0.50'); (B/'scripts'/p.name).write_text(text)
 for p in (OLD/'annotations_20261107/scripts').glob('*.py'):(A/'scripts'/p.name).write_text(p.read_text().replace(str(OLD),str(B)))
 spec=json.loads((OLD/'input_spec.json').read_text());spec.update(job='seq09_300_499_c50_s42',remote_root=str(B),seed=42,confidence_seed=.5);(B/'input_spec.json').write_text(json.dumps(spec,indent=2))
 with np.load(F,allow_pickle=False) as z:count=len(z['means3D']);assert z['cam_trans'].shape[-1]==200
 (B/'field_verification.json').write_text(json.dumps(dict(F1_sha256=sha(F),F200_sha256=sha(M),gaussian_count=count,scope='new seed42 map'),indent=2))
 run('map_export',[PY,B/'scripts/export_assets.py'])
 run('semantic_fusion',[PY,A/'scripts/fuse_semantic.py','--params',F,'--semantic-dir',A/'semantic','--rgb-dir',B/'sequences/09/image_2','--output-dir',A/'fusion','--splatam-root',SPLAT,'--expected-gaussians',str(count),'--expected-params-sha256',sha(F),'--expected-width','1226','--expected-height','370','--deep-check'],True)
 run('annotation_export',[PY,A/'scripts/export_annotations.py'])
 run('cleanup_legacy',[PY,B/'scripts/cleanup_legacy.py'])
 run('cleanup_v2',[PY,B/'scripts/cleanup_v2.py'])
 run('publish',[PY,OLD.parent/'server_publish.py',B])
 status('complete',gaussians=count,map_sha256=sha(F),receipt=json.loads((B/'delivery_receipt.json').read_text()))
except BaseException as e:
 if B.exists():status('failed',error=repr(e),traceback=traceback.format_exc())
 raise
