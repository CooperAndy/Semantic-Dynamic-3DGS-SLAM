from pathlib import Path
import os,subprocess,json,traceback,hashlib
ROOT=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114'); B=ROOT/'seed42_demo_build'; A=B/'annotations_20261107'; OLD=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499'); F=ROOT/'f1_confidence_tuning/c50_s42/F1/params.npz'; SPLAT=Path('/root/autodl-tmp/SplaTAM'); PY='/root/miniconda3/bin/python'; GPU='/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python'; raster='/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1'
def status(stage,**kw):(B/'build_status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
def run(name,cmd,env):
 status(name)
 with (B/f'{name}.log').open('x') as f:subprocess.run(list(map(str,cmd)),cwd=SPLAT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 (A/'fusion').rename(A/'fusion_failed_initial')
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TASK_GPU_GIB='6',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=f'{B}/scripts:{A}/scripts:{SPLAT}')
 envgpu=dict(env,PYTHONPATH=f'{raster}:{B}/scripts:{A}/scripts:{SPLAT}')
 h=hashlib.sha256(F.read_bytes()).hexdigest(); count=3166109
 run('semantic_fusion_retry',[GPU,A/'scripts/fuse_semantic.py','--params',F,'--semantic-dir',A/'semantic','--rgb-dir',B/'sequences/09/image_2','--output-dir',A/'fusion','--splatam-root',SPLAT,'--expected-gaussians',str(count),'--expected-params-sha256',h,'--expected-width','1226','--expected-height','370','--deep-check'],envgpu)
 run('annotation_export',[PY,A/'scripts/export_annotations.py'],env)
 run('cleanup_legacy',[PY,B/'scripts/cleanup_legacy.py'],env)
 run('cleanup_v2',[PY,B/'scripts/cleanup_v2.py'],env)
 run('publish',[PY,OLD.parent/'server_publish.py',B],env)
 status('complete',gaussians=count,map_sha256=h,receipt=json.loads((B/'delivery_receipt.json').read_text()))
except BaseException as e:status('failed',error=repr(e),traceback=traceback.format_exc()); raise
