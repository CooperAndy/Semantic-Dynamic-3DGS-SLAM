"""CPU-only GT scoring after all eight raw DEV replays have been sealed."""
from pathlib import Path
import csv,hashlib,json,numpy as np
from scipy.spatial.transform import Rotation
F=Path(__file__).resolve().parents[1];B=F.parent;P=json.loads((F/'repair_protocol.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=json.loads((F/'receipts/replay_raw_seal.json').read_text());assert seal['status']=='SEALED_BEFORE_CPU_GT'
for path,h in seal['files'].items():assert sha(F/path)==h,path
assert json.loads((F/'receipts/repair_launch.json').read_text())['status']=='EIGHT_REPLAYS_COMPLETED'
gate=json.loads((F/'receipts/repaired_numerical_gate.json').read_text());assert gate['status']=='PASS'
PKG=B/'DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012/阶段3/DGSLAM_方案核查_20261011'
original=PKG/'t02_receipts/04_reproduction/T02_200_DIAG_20261011_195357'
audit=[json.loads(x) for x in (original/'keyframe_pose_audit.jsonl').read_text(encoding='utf-8').splitlines()]
with np.load(PKG/'t02_receipts/evaluation_reference.npz') as z:ref=z['reference'].astype(float)
ids=np.array([r['input_frame_id'] for r in audit],int);X=np.array([r['coarse_c2w'] for r in audit],float)[:,:3,3];Y=ref[ids,:3]
U,_,V=np.linalg.svd((Y-Y.mean(0)).T@(X-X.mean(0)));D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);Q=U@D@V;t=Y.mean(0)-Q@X.mean(0)
rows=[];saved={}
for frame in P['frames']:
 for arm in P['optimizer_recipe']['arms']:
  folder=F/f'replays/f{frame}_{arm}';r=json.loads((folder/'run_manifest.json').read_text());assert r['status']=='COMPLETED'
  for name,h in r['artifacts'].items():assert sha(folder/name)==h
  assert r['GT_audit']['blocked_GT_opens']==0 and r['rng_unchanged'] and r['map_unchanged']
  with np.load(folder/'trace.npz') as z:a={k:v.copy() for k,v in z.items()}
  saved[(frame,arm)]=a
  if arm!='LOCAL_ID':
   other=saved[(frame,'LOCAL_ID')]
   for key in ['initial_c2w','H','P_FULL']:assert np.array_equal(a[key],other[key])
   assert r['input_hash']==json.loads((F/f'replays/f{frame}_LOCAL_ID/run_manifest.json').read_text())['input_hash']
  truth=ref[frame];R=Rotation.from_quat(truth[3:]).as_matrix()
  def error(T):return float(np.linalg.norm(Q@T[:3,3]+t-truth[:3])*100),float(np.degrees(Rotation.from_matrix(R.T@Q@T[:3,:3]).magnitude()))
  before,brot=error(a['initial_c2w'].astype(float));after,arot=error(a['candidate_c2w'].astype(float))
  step=np.linalg.inv(a['initial_c2w'].astype(float))@a['candidate_c2w'].astype(float)
  row={k:r[k] for k in ['frame','arm','initial_loss','final_loss','rgb_l1','depth_l1','coverage','iterations','accepted','reason']}
  row.update(initial_position_error_cm=before,position_error_cm=after,initial_rotation_error_deg=brot,rotation_error_deg=arot,
    net_translation_cm=float(np.linalg.norm(step[:3,3])*100),net_rotation_deg=float(np.degrees(Rotation.from_matrix(step[:3,:3]).magnitude())),L1_drop_percent=100*(1-r['final_loss']/r['initial_loss']))
  rows.append(row)
  proposals=json.loads((folder/'proposals.json').read_text())
  for proposal in proposals:
   if arm=='LOCAL_FULL_CAP' and proposal['loss'] is not None:assert proposal['anchor_distance']<=P['optimizer_recipe']['total_anchor_radius']+1e-12
  assert a['loss'].shape[0]==a['post_c2w'].shape[0]==len(a['accepted'])
decisions=[]
for arm in [a for a in P['optimizer_recipe']['arms'] if a!='LOCAL_ID']:
 checks=[]
 for frame in P['frames']:
  r=next(r for r in rows if r['frame']==frame and r['arm']==arm);base=next(r for r in rows if r['frame']==frame and r['arm']=='LOCAL_ID')
  tests={'position':r['position_error_cm']<=.95*base['position_error_cm'],'rotation':np.deg2rad(r['rotation_error_deg'])<=1.05*np.deg2rad(base['rotation_error_deg'])+1e-6,
    'RGB':r['rgb_l1']<=1.05*base['rgb_l1'],'depth':r['depth_l1']<=1.05*base['depth_l1'],'coverage':r['coverage']>=base['coverage']-.02}
  tests={name:bool(value) for name,value in tests.items()}
  checks.append({'frame':frame,'tests':tests,'pass':all(tests.values())})
 decisions.append({'arm':arm,'pass':all(r['pass'] for r in checks),'mean_position_error_cm':float(np.mean([r['position_error_cm'] for r in rows if r['arm']==arm])),'checks':checks})
eligible=[r for r in decisions if r['pass']];tie=['LOCAL_DIAG_P','LOCAL_FULL_CAP','LOCAL_FULL']
candidate=min(eligible,key=lambda r:(r['mean_position_error_cm'],tie.index(r['arm'])))['arm'] if eligible else None
with (F/'T16_八次同图回放结果.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out={'status':'CPU_EVALUATED_AFTER_RAW_SEAL','scope':'Two exposed DEV snapshots only, fixed original coarse-keyframe SE3 alignment, no candidate realignment; not formal ATE or adoption.',
 'rows':rows,'candidate_screen':decisions,'eligible_full_window_candidate':candidate,'full_window_decision':'REGISTER_ONE_OLD_DEV_WINDOW' if candidate else 'DO_NOT_RUN_FULL_WINDOW',
 'raw_seal_sha256':sha(F/'receipts/replay_raw_seal.json'),'reference_sha256':sha(PKG/'t02_receipts/evaluation_reference.npz'),'coarse_audit_sha256':sha(original/'keyframe_pose_audit.jsonl'),
 'alignment':{'Q':Q.tolist(),'t':t.tolist(),'fit_keyframes':len(ids)},'original_T14':'NOT_ADOPTED_UNCHANGED','original_T15':'FAIL_UNCHANGED'}
(F/'receipts/replay_evaluation.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
