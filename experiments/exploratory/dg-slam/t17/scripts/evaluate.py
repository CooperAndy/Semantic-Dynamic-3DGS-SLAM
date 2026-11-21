"""Separate CPU-only evaluator; read GT only after all runs are terminal/sealed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from audit_camera import audit_trace
F=Path(__file__).resolve().parents[1];R=F.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def mats(poses):
 assert poses.ndim==2 and poses.shape[1]==7 and np.isfinite(poses).all()
 assert np.allclose(np.linalg.norm(poses[:,3:],axis=1),1,atol=1e-4)
 T=np.tile(np.eye(4),(len(poses),1,1));T[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix();T[:,:3,3]=poses[:,:3];return T
def proper(M):
 U,_,V=np.linalg.svd(M);D=np.eye(3);D[2,2]=np.linalg.det(U@V);return U@D@V
def metrics(gt,est):
 assert gt.shape==est.shape
 X=est[:,:3,3];Y=gt[:,:3,3];x=X-X.mean(0);y=Y-Y.mean(0)
 Q=proper(y.T@x);t=Y.mean(0)-Q@X.mean(0);errors=np.linalg.norm((Q@X.T).T+t-Y,axis=1)
 # Independent Horn quaternion eigensystem for rigid translation RMSE.
 H=x.T@y;tr=np.trace(H);d=np.array([H[1,2]-H[2,1],H[2,0]-H[0,2],H[0,1]-H[1,0]])
 K=np.zeros((4,4));K[0,0]=tr;K[0,1:]=d;K[1:,0]=d;K[1:,1:]=H+H.T-tr*np.eye(3)
 _,V=np.linalg.eigh(K);q=V[:,-1];rot=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
 second=np.linalg.norm((rot@x.T).T-y,axis=1)
 assert abs(np.mean(errors**2)-np.mean(second**2))<=1e-12
 out={'ATE':float(np.sqrt(np.mean(errors**2)))}
 for delta in [1,10]:
  if len(gt)<=delta:continue
  trans=[];angles=[]
  for i in range(len(gt)-delta):
   a=np.linalg.inv(gt[i])@gt[i+delta];b=np.linalg.inv(est[i])@est[i+delta];err=np.linalg.inv(a)@b
   trans.append(np.linalg.norm(err[:3,3]));angles.append(Rotation.from_matrix(proper(err[:3,:3])).magnitude())
  out[f'RPE_t_{delta}']=float(np.sqrt(np.mean(np.square(trans))));out[f'RPE_r_{delta}']=float(np.sqrt(np.mean(np.square(angles))));out[f'pairs_{delta}']=len(trans)
 return out
# This body is appended to the unchanged, previously Horn-crosschecked metric functions.
def main():
 P=json.loads((F/'protocol.json').read_text());lock=json.loads((F/'execution_lock.json').read_text());launch=json.loads((F/'receipts/registered_launch.json').read_text())
 assert launch['status']=='ALL_SIX_TERMINAL' and len(launch['runs'])==6
 assert sha(F/'protocol.json')==lock['protocol_sha256'] and sha(Path(__file__))==lock['evaluator_sha256']
 sealed=json.loads((F/'receipts/raw_seal.json').read_text());assert sealed['status']=='SEALED_BEFORE_CPU_GT'
 for rel,h in sealed['files'].items():assert sha(F/rel)==h,rel
 for path,h in lock['runtime_immutable_files'].items():assert sha(path)==h,path
 for path,h in lock['evaluation_indices'].items():assert sha(path)==h,path
 gtpath=R/'02_data_masks/TUM/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt';assert sha(gtpath)==lock['GT_sha256']
 gtdata=np.loadtxt(gtpath);rows=list(csv.DictReader((F/'inputs/DEV200_evaluation.csv').open()));assert len(rows)==200
 reference=gtdata[[int(row['gt_index']) for row in rows],1:];truth=mats(reference)
 np.savez(F/'receipts/evaluation_reference.npz',reference=reference)
 trials=[];audits=[];dense_KF_changes=[]
 keys=['ATE','RPE_t_1','RPE_t_10','RPE_r_1','RPE_r_10']
 for entry in launch['runs']:
  folder=Path(entry['folder']);m=json.loads((folder/'run_manifest.json').read_text()) if (folder/'run_manifest.json').exists() else {}
  row={k:entry[k] for k in ['window','seed','arm','exit_code']};row.update(status='FAILED',error=m.get('error'))
  if entry['exit_code']==0 and m.get('status')=='COMPLETED':
   assert m['protocol_sha256']==lock['protocol_sha256'] and m['execution_lock_sha256']==sha(F/'execution_lock.json')
   assert m['GT_guard']['blocked_GT_opens']==0 and m['counts']['input_frames']==200 and m['mapping_identity_audit']=='PASS'
   assert m['mapped_keyframes']==m['droid_keyframes'] and m['droid_keyframes']>2 and m['counts']['gs_mapping_calls']>2
   assert (m['counts']['camera_backward']==0 if entry['arm']=='NO_GS_REFINE' else m['counts']['camera_backward']>0)
   for name,h in m['artifacts'].items():assert sha(folder/name)==h
   traces=list(folder.glob('camera_kf*.npz'))
   assert not traces if entry['arm']=='NO_GS_REFINE' else bool(traces)
   for path in traces:
    with np.load(path) as z:assert np.array_equal(z['C'],np.eye(6)),'ID must use I, not FULL or diagonal P'
    audits.append(audit_trace(path,[.02]*3+[.017453292519943295]*3))
   with np.load(folder/'trajectories.npz') as z:
    est=z['estimated'];ids=z['selected_frames'];ts=z['rgb_timestamps']
   assert np.array_equal(ids,np.arange(200)) and np.array_equal(ts,[float(r['rgb_timestamp']) for r in rows])
   T=mats(est);values=metrics(truth,T);assert values['pairs_1']==199 and values['pairs_10']==190
   with np.load(folder/'keyframe_poses.npz') as z:
    k=z['indices'].astype(int);coarse=z['coarse'];tracked=z['tracked'];mapped=z['mapped']
    assert np.allclose(mapped,tracked,atol=1e-5),'Mapping BA disabled; mapping should retain tracked pose'
    if entry['arm']=='NO_GS_REFINE':assert np.allclose(coarse,tracked,atol=1e-6)
    sparse={name:metrics(truth[k],array) for name,array in [('coarse',coarse),('tracked',tracked),('mapped',mapped)]}
    change=np.linalg.norm(T[k,:3,3]-mapped[:,:3,3],axis=1)
   dense_KF_changes.append({'seed':entry['seed'],'arm':entry['arm'],'keyframes':len(k),'post_filler_vs_mapping_keyframe_translation_max_m':float(change.max()),'mean_m':float(change.mean())})
   row.update(status='COMPLETED',metrics=values,sparse_metrics=sparse,seconds=m['seconds'],peak_memory_bytes=m['peak_torch_allocated_bytes'],gaussian_count=m['gaussian_count'],counts=m['counts'])
  trials.append(row)
 allok=all(r['status']=='COMPLETED' for r in trials);means={};ratios={};paired=[];idg={};nog={};resource={}
 if allok:
  for arm in P['arms']:
   rr=[r for r in trials if r['arm']==arm];assert len(rr)==3 and {r['seed'] for r in rr}==set(P['seeds'])
   means[arm]={key:float(np.mean([r['metrics'][key] for r in rr])) for key in keys}
   means[arm].update({key:float(np.mean([r[key] for r in rr])) for key in ['seconds','peak_memory_bytes','gaussian_count']})
  ratios={key:means['LOCAL_ID'][key]/max(means['NO_GS_REFINE'][key],1e-30) for key in keys}
  idg['mean_ATE']=ratios['ATE']<=.90
  for d in [1,10]:idg[f'mean_RPE_t_{d}']=ratios[f'RPE_t_{d}']<=.95;idg[f'mean_RPE_r_{d}']=ratios[f'RPE_r_{d}']<=1.05
  for key in keys:nog['mean_'+key]=1/max(ratios[key],1e-30)<=1.05
  for seed in P['seeds']:
   a=next(r for r in trials if r['seed']==seed and r['arm']=='LOCAL_ID');b=next(r for r in trials if r['seed']==seed and r['arm']=='NO_GS_REFINE')
   pair={'seed':seed,'ID_over_NO':{key:a['metrics'][key]/max(b['metrics'][key],1e-30) for key in keys},'ID_minus_NO':{key:a['metrics'][key]-b['metrics'][key] for key in keys}};paired.append(pair)
   for key in keys:idg[f's{seed}_{key}']=pair['ID_over_NO'][key]<=1.05;nog[f's{seed}_{key}']=1/max(pair['ID_over_NO'][key],1e-30)<=1.05
  resource={key:means['LOCAL_ID'][key]/max(means['NO_GS_REFINE'][key],1e-30) for key in ['seconds','peak_memory_bytes','gaussian_count']}
  nog['mean_seconds']=1/max(resource['seconds'],1e-30)<=1.05;nog['mean_peak_memory']=1/max(resource['peak_memory_bytes'],1e-30)<=1.20
 decision='ENGINEERING_INCOMPLETE' if not allok else ('ID_BENEFIT_SUPPORTED' if all(idg.values()) else ('PREFER_NO_GS_WITHIN_DEV' if all(nog.values()) else 'INCONCLUSIVE_NEITHER_QUALIFIED'))
 result={'task':'T17','integrity':'PASS','all_six_complete':allok,'decision':decision,'scope':P['scope'],'mean_metrics':means,'ID_over_NO_ratios':ratios,'paired':paired,'ID_benefit_gates':idg,'NO_GS_retention_gates':nog,'failed_ID_gates':[k for k,v in idg.items() if not v],'failed_NO_gates':[k for k,v in nog.items() if not v],'resource_ID_over_NO':resource,'equal_resource_ID_claim_allowed':bool(resource) and all(x<=1.2 for x in resource.values()),'failures':{arm:sum(r['status']!='COMPLETED' for r in trials if r['arm']==arm) for arm in P['arms']},'camera_audit':{'status':'PASS','traces':len(audits),'pose_max_abs':max([a['pose_max_abs'] for a in audits],default=0.)},'filler_keyframe_changes':dense_KF_changes,'protocol_sha256':sha(F/'protocol.json'),'execution_lock_sha256':sha(F/'execution_lock.json'),'raw_seal_sha256':sha(F/'receipts/raw_seal.json'),'no_failed_or_unfavorable_run_dropped':True,'independent_Horn_crosscheck':'PASS','formal_acceptance':False}
 dump(F/'receipts/trial_scores.json',trials);dump(F/'receipts/acceptance.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
