"""Compare sealed historical G07 and A2 camera intermediates on CPU."""
import hashlib,json
from pathlib import Path
import numpy as np
B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200');O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');A=O/'mapper_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def comp(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));i=np.unravel_index(d.argmax(),d.shape)
 return {'exact':bool(np.array_equal(a,b)),'diff_count':int(np.count_nonzero(a!=b)),'max_abs':float(d[i]),'index':list(map(int,i)),'a':float(a[i]),'b':float(b[i])}
def main():
 out=O/'diagnostic_20261028'/'pose_historical_audit.json';assert not out.exists()
 old=B/'runs/G00_S0/params.npz';oldt=B/'runs/G00_S0/estimated_trajectory.npz';ref=B/'inputs/runtime/orb_adapter_reference.npz';new=A/'runs/B0_200/params.npz'
 oldseal=json.loads((B/'receipts/raw_seal.json').read_text());oldlocks={str(B/x['path']):x['sha256'] for x in oldseal['files']}
 newseal=json.loads((A/'A2_DEV_evidence_seal_20261028.json').read_text());newlocks={x['path']:x['sha256'] for x in newseal['files']}
 assert sha(old)==oldlocks[str(old)] and sha(oldt)==oldlocks[str(oldt)] and sha(new)==newlocks[str(new)]
 assert sha(ref)=='9b1bc315b6b2bd7a957f818a770ef9323046d0e2eaf96eff093ca49516adb46e'
 with np.load(old) as x,np.load(new) as y:
  a=x['external_w2c_all_frames'].copy();b=y['external_w2c_all_frames'].copy();at=x['cam_trans'].copy();bt=y['cam_trans'].copy();aq=x['cam_unnorm_rots'].copy();bq=y['cam_unnorm_rots'].copy()
 with np.load(oldt) as x,np.load(ref) as y:reference_same=comp(x['w2c'],y['w2c']);refc=comp(x['c2w'],y['c2w'])
 result={'status':'COMPLETE','old_map_sha256':sha(old),'new_map_sha256':sha(new),'old_trajectory_sha256':sha(oldt),'reference_sha256':sha(ref),
         'old_trajectory_vs_reference_w2c':reference_same,'old_trajectory_vs_reference_c2w':refc,
         'old_vs_new_gpu_input_inverse_w2c':comp(a,b),'old_vs_new_inverse_rotation':comp(a[:,:3,:3],b[:,:3,:3]),'old_vs_new_inverse_translation':comp(a[:,:3,3],b[:,:3,3]),
         'old_vs_new_saved_camera_translation':comp(at,bt),'old_vs_new_saved_quaternion':comp(aq,bq),
         'limits':'Historical GPU/runtime differ from current; re-running on identical historical hardware is needed to separate hardware, library, and floating-point causes. No GT accuracy claim.'}
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
