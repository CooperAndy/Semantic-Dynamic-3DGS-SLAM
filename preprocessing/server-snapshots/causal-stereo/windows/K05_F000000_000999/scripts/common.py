from pathlib import Path
import json,hashlib,datetime,os
R=Path('/root/autodl-tmp/stage3_g21_stereo_20261029_p47149');E=R.parent/'stage3_g21_stereo_evidence_20261029_p47149';I=R.parent/'stage3_g21_inputs_20261029_p47149';C=R.parent/'stage3_g19_past_conf_20261029_p47149';P14=R.parent/'stage3_g14_20261027_p30915/data_pipeline';CPU=R.parent/'stage3_g15_20261028_p47149/runtime/venv/bin/python';RAFTPY=R.parent/'stage3_g05_runtime_20261025/venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(j,indent=2,allow_nan=False));q.replace(p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

P=P14
F=I/"windows"/'K05_F000000_000999'
R=Path('/root/autodl-tmp/stage3_g21_stereo_20261029_p47149/windows/K05_F000000_000999')
