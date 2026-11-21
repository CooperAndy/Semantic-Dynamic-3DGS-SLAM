from pathlib import Path
import json,hashlib,datetime,os
R=Path('/root/autodl-tmp/s23_effects_20261105_p16247/E3_K05_S0_R3_serial');E=R.parent/'stage3_g30_recovery_v2_20261101_p44181';D=R.parent/'stage3_g18d_20261028_p47149';DE=D.parent/'stage3_g18d_evidence_20261028_p47149';C=R.parent/'stage3_g19_past_conf_20261029_p47149';F=R.parent/'stage3_g19_causal_20261029_p47149';P=R.parent/'stage3_g15_20261028_p47149';PY=str(P/'runtime/venv/bin/python');EVAL=str(R.parent/'conda_envs/slam_eval/bin/python')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
def env():return dict({k:v for k,v in os.environ.items() if k!='CUDA_VISIBLE_DEVICES' or v!=''},PYTHONPATH=str(R/'native'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',TZ='Asia/Shanghai')
