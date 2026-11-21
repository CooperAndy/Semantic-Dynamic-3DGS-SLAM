import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/A2_union_20261029')
sets={
 '03':('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1/input_manifest.json','/root/autodl-tmp/A2_stereo_robust_geometry/seq03_input_20261029_retry1/raw_left'),
 '08':('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200/inputs/runtime/input_manifest.json','/root/autodl-tmp/stage3_g05B_20261025/raw_left')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for seq,(m,raw) in sets.items():
 src=json.loads(Path(m).read_text()); out=ROOT/f'seq{seq}_gate06';out.mkdir(exist_ok=False);rows=[];checks=[]
 assert len(src['rows'])==200
 for x in src['rows']:
  f=x['frame_id']; d=np.load(Path(raw)/f'{f:06d}_depth.npy').astype(np.float32); cf=np.load(x['confidence']); old=np.load(x['depth']); d=d[...,0] if d.ndim==3 else d
  assert d.shape==cf.shape==old.shape
  mask=np.isfinite(old)&(old>0); mismatch=int(np.count_nonzero(old[mask]!=d[mask]));
  if mismatch: raise RuntimeError(f'{seq} frame {f} mismatch {mismatch}')
  new=np.where(cf>=.6,d,0).astype(np.float32);assert np.array_equal(old[mask],new[mask])
  p=out/f'{f:06d}.npy';np.save(p,new);nr=dict(x);nr['depth']=str(p);nr['hashes']=dict(x['hashes']);nr['hashes']['depth']=sha(p);rows.append(nr)
  checks.append({'frame':f,'raw_sha256':sha(Path(raw)/f'{f:06d}_depth.npy'),'n08':int(mask.sum()),'n06':int((new>0).sum())})
 dst=dict(src);dst['rows']=rows;dst['confidence_threshold']=.6;dst['source_manifest_sha256']=sha(m)
 (out/'manifest.json').write_text(json.dumps(dst,indent=2));(out/'audit.json').write_text(json.dumps({'status':'PASS','sequence':seq,'checks':checks},indent=2))
 print(seq,'PASS',len(rows),flush=True)
