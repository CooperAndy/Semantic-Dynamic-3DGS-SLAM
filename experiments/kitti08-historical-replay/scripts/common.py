from pathlib import Path
import json, hashlib, datetime
R=Path('/root/autodl-tmp/stage3_g14_recovery_20261027_p10781')
E=R.parent/'stage3_g14_recovery_evidence_20261027_p10781'
PY=str(R.parent/'stage3_g10_runtime_20261025_p46225/venv/bin/python')
N=1000
HOLD=list(range(19,N,20))
VARIANTS=[f'B{b:02d}C0_S{s}' for s in range(3) for b in [0,3]]
NATIVE='/root/miniconda3/lib/python3.12/site-packages/diff_gaussian_rasterization/_C.cpython-312-x86_64-linux-gnu.so'
CORE_SHA='6d65efd83b61c414c82231e67928c0c8f4dd564c9e7cb56440241b8343f544b7'
NATIVE_SHA='4afa94b9a287c553e0f893c3a9e18b30f6599df02ce6c1f805765c2149b19d70'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def locks(reg):
 for x in reg['file_locks']:
  p=R/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
def definition(v):return dict(history_budget=int(v[1:3]),history=int(v[1:3])>0,color_box=False,seed=int(v.rsplit('S',1)[1]))
def donor_manifest(seed):return R/'receipts'/f'donor_S{seed}.json'
def check_donor(seed):
 p=donor_manifest(seed);j=json.loads(p.read_text());assert j['status']=='SEALED_H0_PAIR_INPUTS'
 for x in j['files']:assert sha(R/x['path'])==x['sha256'],x['path']
 return j
