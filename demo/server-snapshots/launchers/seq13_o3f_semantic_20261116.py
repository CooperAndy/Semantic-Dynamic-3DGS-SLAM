from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
R=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115');D=R/'semantic_demo_20261116';S=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114/seed42_demo_build/annotations_20261107/scripts');MAP=R/'cell_O3_retry1/runs/O3F/params.npz';RGB=R/'sequences/13/image_2';EXPECTED='6cebfe6efc70888706c17c443e5774e3894bd74e0d0baa6f495b51a33bb97c20'
def status(s,**kw):
 p=D/'status.json';p.write_text(json.dumps(dict(stage=s,time=time.strftime('%FT%T%z'),**kw),indent=2));print(s,kw,flush=True)
def run(stage,cmd,env):
 if shutil.disk_usage(D).free<8*2**30:raise RuntimeError('Less than 8 GiB available')
 status(stage)
 with (D/(stage+'.log')).open('x') as f:subprocess.run(list(map(str,cmd)),cwd='/root/autodl-tmp/SplaTAM',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 D.mkdir(exist_ok=False);status('preflight');h=hashlib.sha256()
 with MAP.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==EXPECTED
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 run('predict',['/root/venvs/semantic_seg/bin/python','/root/autodl-tmp/slam_project/stage2_semantic/scripts/generate_mask2former_semantics_kitti.py','--image-dir',RGB,'--output-dir',D/'predictions','--model-id','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-semantic-91281d04','--local-files-only','--start','0','--num-frames','260','--device','cuda','--preview-every','100'],env)
 assert len(list((D/'predictions/labels').glob('*.npy')))==260
 src=(S/'fuse_semantic.py').read_text();old='if num_frames != 200:';assert src.count(old)==1;src=src.replace(old,'if num_frames != 260:').replace('frame count mismatch: expected 200, got','frame count mismatch: expected 260, got');old='                (timestep <= frame)\n                & (z > 0.05)';assert src.count(old)==1;src=src.replace(old,'                (z > 0.05)  # fixed final map offline fusion');script=D/'fuse.py';script.write_text(src)
 env.update(PYTHONPATH=f'/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1:{S}:/root/autodl-tmp/SplaTAM',TASK_GPU_GIB='18')
 run('fusion',['/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python',script,'--params',MAP,'--semantic-dir',D/'predictions','--rgb-dir',RGB,'--output-dir',D/'fusion','--splatam-root','/root/autodl-tmp/SplaTAM','--expected-gaussians','834436','--expected-params-sha256',EXPECTED,'--expected-width','1241','--expected-height','376','--deep-check','--representative-frames','0,49,129,199,259'],env)
 status('complete')
try:main()
except BaseException as e:
 if D.exists():status('failed',error=repr(e),traceback=traceback.format_exc())
 raise
