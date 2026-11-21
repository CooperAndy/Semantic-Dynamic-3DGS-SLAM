"""Audit collected raw trajectories with SciPy alignment and batched RPE."""
from pathlib import Path
import hashlib,json,warnings,numpy as np
from scipy.spatial.transform import Rotation
F=Path(__file__).resolve().parents[1];P=json.loads((F/'protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((F/'receipts/raw_seal.json').read_text(encoding='utf-8'));assert seal['status']=='SEALED_BEFORE_CPU_GT'
for rel,h in seal['files'].items():assert sha(F/rel)==h,rel
scores=json.loads((F/'receipts/trial_scores.json').read_text(encoding='utf-8'));a=json.loads((F/'receipts/acceptance.json').read_text(encoding='utf-8'))
def mats(p):
 T=np.tile(np.eye(4),(len(p),1,1));T[:,:3,:3]=Rotation.from_quat(p[:,3:]).as_matrix();T[:,:3,3]=p[:,:3];return T
with np.load(F/'receipts/evaluation_reference.npz') as z:G=mats(z['reference'])
max_difference=0.;checked=0;counts={};trajectory_hashes={};poses_by_arm={arm:[] for arm in P['arms']}
for row in scores:
 if row['status']!='COMPLETED':continue
 folder=F/f"runs/DEV200_s{row['seed']}_{row['arm']}"
 with np.load(folder/'trajectories.npz') as z:
  E=mats(z['estimated']);assert np.array_equal(z['selected_frames'],np.arange(200))
  trajectory_hashes[f"s{row['seed']}_{row['arm']}"]=hashlib.sha256(z['estimated'].tobytes()).hexdigest();poses_by_arm[row['arm']].append(E)
 X=E[:,:3,3];Y=G[:,:3,3]
 _,rssd=Rotation.align_vectors(Y-Y.mean(0),X-X.mean(0));values={'ATE':float(rssd/np.sqrt(200))}
 for delta in [1,10]:
  A=np.linalg.inv(G[:-delta])@G[delta:];B=np.linalg.inv(E[:-delta])@E[delta:];error=np.linalg.inv(A)@B
  values[f'RPE_t_{delta}']=float(np.sqrt(np.mean(np.sum(error[:,:3,3]**2,axis=1))))
  angles=Rotation.from_matrix(error[:,:3,:3]).magnitude();values[f'RPE_r_{delta}']=float(np.sqrt(np.mean(angles**2)))
 for key,value in values.items():
  difference=abs(value-row['metrics'][key]);max_difference=max(max_difference,difference);assert difference<1e-8,(row['seed'],row['arm'],key,difference)
 m=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'));assert not m['formal'] and m['registered_DEV']
 assert m['cfg']['gs_pose_refine']==(row['arm']=='LOCAL_ID') and m['GT_guard']['blocked_GT_opens']==0
 if row['arm']=='NO_GS_REFINE':
  assert row['counts']['camera_backward']==0
  with np.load(folder/'keyframe_poses.npz') as z:assert np.array_equal(z['coarse'],z['tracked'])
 else:
  assert row['counts']['camera_backward']>0
  for p in folder.glob('camera_kf*.npz'):
   with np.load(p) as z:assert np.array_equal(z['C'],np.eye(6))
 counts[f"s{row['seed']}_{row['arm']}"]=row['counts'];checked+=1
assert checked==sum(row['status']=='COMPLETED' for row in scores)
seed_variation={}
for arm,values in poses_by_arm.items():
 if values:
  seed_variation[arm]={'unique_trajectory_array_hashes':len({h for key,h in trajectory_hashes.items() if key.endswith('_'+arm)}),'max_same_frame_translation_difference_vs_first_seed_m':float(max(np.linalg.norm(v[:,:3,3]-values[0][:,:3,3],axis=1).max() for v in values))}
receipt={'status':'PASS','raw_files_SHA256_verified':len(seal['files']),'trajectories_independently_recomputed':checked,'metric_max_abs_difference':max_difference,'independent_method':'SciPy align_vectors centered RSSD for ATE; batched relative SE3 for both RPE deltas','NO_GS_coarse_tracked_exact':True,'ID_C_identity':True,'runtime_GT_opens':0,'counts':counts,'trajectory_array_hashes':trajectory_hashes,'seed_variation':seed_variation,'formal_acceptance':False}
(F/'receipts/independent_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k!='counts'}))
