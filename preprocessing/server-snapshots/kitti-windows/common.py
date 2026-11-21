from pathlib import Path
import json,hashlib,datetime,os
R=Path('/root/autodl-tmp/stage3_g21_inputs_20261029_p47149');E=R.parent/'stage3_g21_inputs_evidence_20261029_p47149';F=R.parent/'stage3_g19_causal_20261029_p47149';DATA=R.parent/'slam_project/data/kitti/dataset/sequences';PY=R.parent/'stage3_g15_20261028_p47149/runtime/venv/bin/python';EVAL=R.parent/'conda_envs/slam_eval/bin/python'
WINDOWS=[dict(id='K05_F000000_000999',sequence='05',source_start=0,frame_count=1000),dict(id='K10_F000000_000999',sequence='10',source_start=0,frame_count=1000)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(j,indent=2,allow_nan=False));q.replace(p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def env():return dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(F/'runtime/lib'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
