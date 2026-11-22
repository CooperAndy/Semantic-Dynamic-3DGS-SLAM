from pathlib import Path
import json,hashlib,datetime,os
R=Path(os.environ.get('S23_CELL_ROOT', Path(__file__).resolve().parents[1]))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
def env():return dict({k:v for k,v in os.environ.items() if k!='CUDA_VISIBLE_DEVICES' or v!=''},PYTHONPATH=str(R/'native'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',WANDB_MODE='disabled',TZ='Asia/Shanghai')
