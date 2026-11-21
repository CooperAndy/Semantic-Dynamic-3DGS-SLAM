import sys
sys.path.insert(0,'/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/scripts')
import protect_parent;protect_parent.install()
from common import *
assert str(R)=='/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181' and str(E)=='/root/autodl-tmp/stage3_g30_recovery_v2_20261101_p44181'
import numpy as np,time
from scipy.spatial.transform import Rotation
from PIL import Image
start=time.monotonic();plan=json.loads((R/'plan.json').read_text());N=plan['frames'];reg=json.loads((R/'registration.json').read_text());assert json.loads((E/'controller.json').read_text())['status'].startswith('COMPLETE')
for path,digest in reg['source_files'].items():assert sha(R/path)==digest,path
for item in reg['parent_inputs']:assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
manifest=json.loads((R/'inputs/runtime/input_manifest.json').read_text());inputs=manifest['rows'];donor=R/'runs'/plan['donor'];held=set(plan['holdout']);native=np.load(R/'inputs/runtime/nominal_native.npz');schedule=np.load(donor/'target_schedule.npz');expected_pairs=[(t,it) for t in range(N) if t not in held for it in range(60)];initials=set();birth_files=0;birth_points=0;float_files=0;propagated=0;pose_rows=0;mapping_steps=0;max_birth=0.;max_prop=0.;max_rgb=0.;image_cache={};depth_cache={};conf_cache={};summaries=[]
def load_frame(i):
 if i not in depth_cache:
  depth_cache[i]=np.load(inputs[i]['depth']);conf_cache[i]=np.load(inputs[i]['confidence'])
  with Image.open(inputs[i]['rgb']) as im:image_cache[i]=np.asarray(im.convert('RGB'),np.float32)/255
 return depth_cache[i],conf_cache[i],image_cache[i]
def difference(A,B):
 D=np.asarray(A,np.float64)@np.linalg.inv(np.asarray(B,np.float64));return float(np.linalg.norm(D[:3,3])),float(np.degrees(Rotation.from_matrix(D[:3,:3]).magnitude()))
for v in plan['variants']:
 out=R/'runs'/v;receipt=json.loads((out/'receipt.json').read_text());assert receipt['stable_pose_sha256']==plan['stable_pose_sha256'];assert receipt['stable_birth_sha256']==plan['stable_birth_sha256'];assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION' and receipt['frames']==N and receipt['GT_blocked_opens']==0;initials.add(receipt['initial_gaussian_sha256']);assert receipt['dataset_access']==[0]+list(range(N));assert receipt['mapping_observation']['mapping_steps']==len(expected_pairs)==receipt['mapping_backward'];assert receipt['camera_backward']==0
 expected_history=3*sum(t>=200 for t in range(N) if t not in held) if int(v[1:3]) else 0;assert receipt['mapping_observation']['history_slots']==expected_history
 updates=list(map(json.loads,(out/'mapping_updates.jsonl').read_text().splitlines()));assert [(x['frame'],x['iteration']) for x in updates]==expected_pairs;mapping_steps+=len(updates)
 with np.load(out/'target_schedule.npz') as z:
  assert all(np.array_equal(z[k],schedule[k]) for k in schedule.files)
  for x in updates:
   t,it=x['frame'],x['iteration'];expected=int(z['history_03' if int(v[1:3]) else 'base'][t,it]);assert x['target']==expected and expected<=t and expected not in held
   if x['history_slot']:assert int(v[1:3])==3 and expected<=t-200 and it in [15,35,55]
  assert sum(x['history_slot'] for x in updates)==receipt['mapping_observation']['history_slots']
 events=list(map(json.loads,(out/'backend_events.jsonl').read_text().splitlines()));assert [x['frame'] for x in events]==list(range(N));core_events=list(map(json.loads,(out/'external_tracking.jsonl').read_text().splitlines()));assert len(core_events)==N and all(x['mapping_heldout']==(x['frame'] in held) for x in core_events)
 with np.load(out/'estimated_trajectory.npz') as z:trajectory=z['w2c'].copy();assert np.array_equal(z['frame_ids'],np.arange(N))
 with np.load(out/'params.npz') as z:
  assert not np.isin(z['timestep'],list(held)).any() and not np.isin(z['keyframe_time_indices'],list(held)).any();assert max(z['timestep'])<N
  for key in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales']:assert np.isfinite(z[key]).all()
  qsave=z['cam_unnorm_rots'].copy();tsave=z['cam_trans'].copy();gaussians=len(z['means3D'])
 checkpoints=[]
 for snap in range(0,N,199):
  f=out/f'params{snap}.npz';assert f.exists()
  with np.load(f) as z:
   for key in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales']:assert np.isfinite(z[key]).all()
   assert np.array_equal(z['cam_unnorm_rots'][...,:snap+1],qsave[...,:snap+1]) and np.array_equal(z['cam_trans'][...,:snap+1],tsave[...,:snap+1]);checkpoints.append(dict(frame=snap,points=len(z['means3D']),sha256=sha(f)))
  kf=np.load(out/f'keyframe_time_indices{snap}.npy');assert (kf<=snap).all() and not np.isin(kf,list(held)).any()
 dump(out/'checkpoint_audit.json',dict(status='PASS_ALL_SAME_RUN_CHECKPOINTS_AND_SELECTED_CAMERAS',checkpoints=checkpoints,history_slots=expected_history))
 feedback=False
 for i,row in enumerate(events):
  pose_rows+=1;nominal=np.array(row['nominal_w2c']);selected=np.array(row['selected_w2c']);prediction=np.array(row['prediction_w2c']);assert np.array_equal(nominal,native['w2c'][i]);assert np.array_equal(qsave[...,i],np.array(row['selected_q'],np.float32)) and np.array_equal(tsave[...,i],np.array(row['selected_t'],np.float32));assert np.array_equal(selected,np.asarray(trajectory[i],np.float64));assert row['had_feedback']==feedback
  if not feedback:assert np.array_equal(prediction,nominal)
  else:
   expected=nominal@np.linalg.inv(np.array(events[i-1]['nominal_w2c']))@np.array(events[i-1]['selected_w2c']);t,r=difference(prediction,expected);assert t<.0003 and r<.002;propagated+=1;max_prop=max(max_prop,t)
  if i and row['policy']!='ORB':
   assert row['anchor']<i and row['anchor'] not in held;assert row['map_before_sha256']==row['map_after_sha256'];assert sha(out/row['proposal_path'])==row['proposal_sha256'];proposal=json.loads((out/row['proposal_path']).read_text());assert proposal['ok']==row['geometry_ok'];candidate=np.array(row['candidate_w2c'])
   if row['metrics'] is not None:
    a,b=row['metrics']['initial'],row['metrics']['candidate'];base=bool(row['geometry_ok'] and row['support']['sufficient'] and a['finite'] and b['finite'] and b['total']<=.99*a['total'] and b['rgb']<=1.05*a['rgb']+1e-12 and b['depth']<=1.05*a['depth']+1e-12 and row['translation_m']<=.5 and row['rotation_deg']<=3);assert row['hard']['accepted']==(base and b['coverage']>=.98);assert row['mass']['accepted']==(base and row['opacity_retention']>=.98)
   decision=row['mass'] if row['policy']=='MASS' else row['hard'];assert decision['accepted']==row['accepted'];assert np.array_equal(selected,candidate if row['accepted'] else prediction)
   if not row['accepted']:assert row['selected_q']==row['prediction_q'] and row['selected_t']==row['prediction_t']
   if proposal.get('inliers'):
    pts=np.array(proposal['object_points']);pixels=np.array(proposal['image_points']);idx=proposal['inliers'];K=np.array(manifest['K'],np.float32).astype(np.float64);A=np.array(events[row['anchor']]['selected_w2c']);initial=prediction@np.linalg.inv(A);T=np.array(proposal['relative_w2c'])
    def projection(T):
     X=pts@T[:3,:3].T+T[:3,3];uv=X@K.T;return uv[:,:2]/uv[:,2,None]
    med=float(np.median(np.linalg.norm(projection(T)[idx]-pixels[idx],axis=1)));base_med=float(np.median(np.linalg.norm(projection(initial)[idx]-pixels[idx],axis=1)));assert abs(med-proposal['median_reprojection_px'])<1e-7 and abs(base_med-proposal['initial_median_reprojection_px'])<1e-7
   if 'float_path' in row:
    f=out/row['float_path'];assert sha(f)==row['float_sha256'];float_files+=1
    with np.load(f) as z:
     mask=z['support'];assert hashlib.sha256(mask.tobytes()).hexdigest()==row['support_sha256'];expected=(z['target_depth']>0)&(z['confidence']>=.8)&(z['initial_ds'][1]>.99)&np.isfinite(z['initial_ds'][0])&np.isfinite(z['initial_ds'][2]-z['initial_ds'][0]**2);assert np.array_equal(mask,expected)
     if mask.any():
      mass=float(np.clip(z['candidate_ds'][1][mask].astype(np.float64),0,1).sum()/np.clip(z['initial_ds'][1][mask].astype(np.float64),0,1).sum());assert abs(mass-row['opacity_retention'])<1e-12
      for state in ['initial','candidate']:
       a=row['metrics'][state];rgb=float(np.abs(z[state+'_rgb'][:,mask].astype(np.float64)-z['target_rgb'][:,mask]).mean());depth=float(np.abs(z[state+'_depth'][mask].astype(np.float64)-z['target_depth'][mask]).mean());assert abs(rgb-a['rgb'])<2e-5 and abs(depth-a['depth'])<2e-5
  if row['accepted']:feedback=True
 births=list(map(json.loads,(out/'addition_history.jsonl').read_text().splitlines()));assert [x['frame'] for x in births]==[i for i in range(1,N) if i not in held];assert sum(x['added'] for x in births)==receipt['mapping_observation']['added']
 for birth in births:
  i=birth['frame'];maskfile=out/'addition_masks'/f'{i:06}.npz';assert sha(maskfile)==sha(donor/'addition_masks'/maskfile.name);dep,conf,rgb=load_frame(i)
  with np.load(maskfile) as z:ids=z['selected_ids'];assert len(np.unique(ids))==len(ids)==birth['added'] and len(ids)<=3000;assert (dep.reshape(-1)[ids]>0).all() and (conf.reshape(-1)[ids]>=.8).all()
  fp=out/'birth_geometry'/maskfile.name;assert fp.exists()
  with np.load(fp) as z:
   assert np.array_equal(ids,z['ids']);T=z['w2c'].astype(np.float64);K=z['intrinsics'].astype(np.float64);assert np.array_equal(T,np.array(events[i]['selected_w2c']));assert np.array_equal(z['sampled_depth'],dep.reshape(-1)[ids]);h,w=dep.shape;y,x=np.divmod(ids,w);depth=dep.reshape(-1)[ids].astype(np.float64);rays=np.column_stack(((x-K[0,2])/K[0,0]*depth,(y-K[1,2])/K[1,1]*depth,depth,np.ones(len(ids))));expected=(rays@np.linalg.inv(T).T)[:,:3]
   if len(ids):
    error=float(np.max(np.linalg.norm(expected-z['pointcloud'][:,:3],axis=1)));assert error<.0003;max_birth=max(max_birth,error);rgberr=float(np.max(np.abs(z['pointcloud'][:,3:]-rgb.reshape(-1,3)[ids])));assert rgberr<1e-7;max_rgb=max(max_rgb,rgberr)
   birth_files+=1;birth_points+=len(ids)
 summaries.append(dict(variant=v,frames=N,accepted=sum(x['accepted'] for x in events),mapping_steps=len(updates),added=sum(x['added'] for x in births),final_gaussians=gaussians))
assert len(initials)==1 and float_files==12 and pose_rows==6*N and mapping_steps==6*(N-len(held))*60
dump(R/'receipts/independent_audit.json',dict(status='PASS_SIX_NATIVE_MAPPING_POSE_BIRTH_TARGET_AND_FLOAT_AUDIT',at=now(),pose_rows=pose_rows,mapping_steps=mapping_steps,birth_files=birth_files,birth_points=birth_points,float_files=float_files,propagated_frames=propagated,max_birth_world_error_m=max_birth,max_birth_rgb_difference=max_rgb,max_propagation_quantization_m=max_prop,initial_gaussian_sha256=next(iter(initials)),variants=summaries,seconds=time.monotonic()-start,quality_adoption=False));print((R/'receipts/independent_audit.json').read_text())
