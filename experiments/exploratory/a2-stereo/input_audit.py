import json,hashlib
from pathlib import Path
import numpy as np
base=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/A200/inputs/runtime/input_manifest.json')
gate=Path('/root/autodl-tmp/E2_depth/gate06/manifest.json')
a=json.loads(base.read_text());b=json.loads(gate.read_text()); assert len(a['rows'])==len(b['rows'])==200
assert a['pose_sha256']==b['pose_sha256'] and a['K']==b['K'] and (a['H'],a['W'])==(b['H'],b['W'])
report=[]
for x,y in zip(a['rows'],b['rows']):
 assert x['frame_id']==y['frame_id']
 for k in ('rgb','right','confidence'):
  assert x[k]==y[k] and x['hashes'][k]==y['hashes'][k]
 d0=np.load(x['depth']);d6=np.load(y['depth']);cf=np.load(x['confidence']); assert d0.shape==d6.shape==cf.shape
 m=np.isfinite(d0)&(d0>0); bad=int(np.count_nonzero(d0[m]!=d6[m])); missing=int(np.count_nonzero(m & ~(d6>0)))
 if bad or missing: raise RuntimeError(f"frame {x['frame_id']}: mismatch {bad} missing {missing}")
 report.append({'frame':x['frame_id'],'valid_08':int(m.sum()),'valid_06':int((d6>0).sum()),'intersection_exact':True})
out={'status':'PASS','n_frames':len(report),'pose_sha256':a['pose_sha256'],'base_manifest_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'gate_manifest_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),'per_frame':report}
Path('/root/autodl-tmp/A2_union_20261029/input_audit.json').write_text(json.dumps(out,indent=2))
print('PASS 200 frames',sum(x['valid_08'] for x in report),sum(x['valid_06'] for x in report))
