"""Bind new runtime to unchanged old native/source inputs before GPU work."""
import datetime
import hashlib
import json
from pathlib import Path
F=Path(__file__).resolve().parents[1]
OLD=F.parent/'10_T17_NO_GS_vs_ID_20261012'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (F/'execution_lock.json').exists()
old=json.loads((OLD/'execution_lock.json').read_text())
files=dict(old['runtime_immutable_files'])
for path,h in files.items():assert sha(path)==h,path
for part in ['source','scripts','inputs']:
    for p in (F/part).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files[str(p)]=sha(p)
files[str(F/'T19_diagnostic_registration.json')]=sha(F/'T19_diagnostic_registration.json')
reg=json.loads((F/'T19_diagnostic_registration.json').read_text())
for item in reg['candidate_source']:
    assert sha(F/'source'/item['path'])==item['sha256']
lock={'status':'FROZEN_BEFORE_NATIVE_SMOKE_AND_TWO_REGISTERED_STATE_BUILDS',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'protocol_sha256':sha(F/'protocol.json'),
      'runtime_immutable_files':files,
      'T19_registration_sha256':sha(F/'T19_diagnostic_registration.json'),
      'formal':False,'GT_execution':False}
(F/'execution_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps({'status':lock['status'],'runtime_files':len(files),'lock_sha256':sha(F/'execution_lock.json')}))
