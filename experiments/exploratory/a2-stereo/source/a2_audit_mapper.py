"""Independent audit of sealed A2 loss experiments; CPU only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200');N=int(sys.argv[1]);assert N in (25,200)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 out=R/f'independent_audit_{N}.json';assert not out.exists();hold=set(range(19,N,20));expected=[i for i in range(1,N) if i not in hold]
 with np.load(B/'inputs/runtime/orb_adapter_reference.npz') as z:reference=z['c2w'][:N].copy()
 result={};initial=[]
 for arm in ('B0','B1','B2','B3','B4'):
  d=R/'runs'/f'{arm}_{N}';rec=json.loads((d/'a2_receipt.json').read_text());assert rec['status']=='PASS_ENGINEERING';initial.append(rec['initial_hash'])
  assert sha(d/'params.npz')==rec['map_sha256'] and sha(O/'gradient_calibration_20261028.json')==rec['gradient_calibration_sha256']
  hist=[json.loads(x) for x in (d/'addition_history.jsonl').read_text().splitlines()]
  assert [x['frame'] for x in hist]==expected and all(x['selected_count']==3000 for x in hist)
  for x in hist:
   p=d/x['mask_file'];assert sha(p)==x['mask_sha256']
   with np.load(p) as z:
    ids=z['selected_ids'];assert len(ids)==3000 and np.all(ids[1:]>ids[:-1])
  assert len(rec['mapping_loss_frames'])==N-len(hold) and all(v==60 for v in rec['mapping_loss_frames'].values())
  assert not (set(map(int,rec['mapping_loss_frames']))|set(rec['mapping_targets'])|set(rec['confidence_frames']))&hold
  with np.load(d/'estimated_trajectory.npz') as z:
   assert np.array_equal(z['frame_ids'],np.arange(N)) and np.array_equal(z['c2w'],reference)
  with np.load(d/'params.npz') as z:
   assert len(z['means3D'])==rec['survivors'] and not np.isin(z['timestep'],list(hold)).any() and not np.isin(z['keyframe_time_indices'],list(hold)).any()
   assert all(np.isfinite(z[k]).all() for k in ('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales'))
  result[arm]={'status':'PASS','map_sha256':rec['map_sha256'],'actual_addition_events':len(hist),'actual_new_points':sum(x['selected_count'] for x in hist),'survivors':rec['survivors'],'holdout_isolated':True,'orb_adapter_exact':True,'seconds':rec['seconds'],'peak_cuda_bytes':rec['peak_cuda_bytes']}
 assert len(set(initial))==1
 report={'status':'PASS','count':N,'initial_hash':initial[0],'arms':result,'quality_adoption':False,'scope':'engineering contract; not validation of noise or map quality'}
 out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
