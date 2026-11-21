from pathlib import Path
import csv,datetime,hashlib,json
from audit_camera import audit_trace
F=Path(__file__).resolve().parents[1];R=F.parent;P=json.loads((F/'protocol.json').read_text());BASE=Path(P['T14_root'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
assert not (F/'execution_lock.json').exists() and not (F/'runs').exists()
launch=json.loads((F/'receipts/smoke_launch.json').read_text());assert launch['status']=='ALL_SMOKES_COMPLETE' and len(launch['runs'])==2
audits=[];camera=[]
for entry in launch['runs']:
 folder=Path(entry['folder']);m=json.loads((folder/'run_manifest.json').read_text());assert entry['exit_code']==0 and m['status']=='COMPLETED'
 assert m['counts']['input_frames']==200 and m['GT_guard']['blocked_GT_opens']==0 and m['mapping_identity_audit']=='PASS'
 assert m['counts']['gs_mapping_calls']>2
 assert (m['counts']['camera_backward']==0 if entry['arm']=='NO_GS_REFINE' else m['counts']['camera_backward']>0)
 for name,h in m['artifacts'].items():assert sha(folder/name)==h
 for p in folder.glob('camera_kf*.npz'):camera.append(audit_trace(p,[.02]*3+[.017453292519943295]*3))
 audits.append({'arm':entry['arm'],'counts':m['counts'],'seconds':m['seconds'],'keyframes':m['droid_keyframes']})
assert camera
dump(F/'receipts/smoke_audit.json',{'status':'PASS','runs':audits,'GT_read':False,'quality_evaluated':False,'camera_traces':len(camera),'independent_update_max_abs':max(x['pose_max_abs'] for x in camera)})
old=json.loads((BASE/'execution_lock.json').read_text());files=dict(old['runtime_immutable_files'])
for path,h in files.items():assert sha(Path(path))==h,path
for p in (F/'scripts').glob('*.py'):files[str(p)]=sha(p)
for p in (F/'source').rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
  rel=p.relative_to(F/'source')
  assert sha(p)==sha(BASE/'source'/rel),str(rel)
  files[str(p)]=sha(p)
assert sha(F/'source/dg_slam/dg_model.py')==sha(BASE/'source/dg_slam/dg_model.py')
data=R/'02_data_masks/TUM/rgbd_dataset_freiburg3_walking_xyz'
for p in (F/'inputs').glob('*_runtime.csv'):
 files[str(p)]=sha(p)
 rows=list(csv.DictReader(p.open()));assert all('gt_index' not in row for row in rows)
 for row in rows:
  for path in [data/row['rgb_path'],data/row['depth_path'],data/'seg_mask'/Path(row['rgb_path']).name]:files[str(path)]=sha(path)
design=json.loads((F/'receipts/design_registration.json').read_text());assert sha(F/'protocol.json')==design['protocol_sha256']
assert sha(F/'scripts/live_refinement.py')==sha(BASE/'scripts/live_refinement.py')
lock={'status':'FROZEN_BEFORE_SIX_RUNS','registered_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(F/'protocol.json'),'runtime_immutable_files':files,'evaluation_indices':{str(p):sha(p) for p in (F/'inputs').glob('*_evaluation.csv')},'GT_sha256':old['GT_sha256'],'evaluator_sha256':sha(F/'scripts/evaluate.py'),'smoke_audit_sha256':sha(F/'receipts/smoke_audit.json'),'all_registered_runs_not_started':True,'ID_code_identical_to_T14':True,'all_GT_metrics_unread':True,'T14_execution_lock_sha256':sha(BASE/'execution_lock.json')}
dump(F/'execution_lock.json',lock);print(json.dumps({'status':lock['status'],'runtime_files':len(files),'lock_sha256':sha(F/'execution_lock.json')}))
