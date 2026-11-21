"""NumPy/SciPy CPU analysis only; never import DG-SLAM/Torch/native modules."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
from scipy.linalg import expm,logm
from scipy.spatial.transform import Rotation
F=Path(__file__).resolve().parents[1];B=Path(os.environ.get('T18_INPUT_ROOT',str(F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012')))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write_csv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mats(p):
 p=np.asarray(p,dtype=float);assert np.isfinite(p).all() and np.max(abs(np.linalg.norm(p[:,3:],axis=1)-1))<1e-4
 T=np.tile(np.eye(4),(len(p),1,1));T[:,:3,:3]=Rotation.from_quat(p[:,3:]).as_matrix();T[:,:3,3]=p[:,:3];return T
def proper(H):
 U,_,V=np.linalg.svd(H);D=np.eye(3);D[-1,-1]=np.linalg.det(U@V);return U@D@V
def align(E,G):
 X=E[:,:3,3];Y=G[:,:3,3];Q=proper((Y-Y.mean(0)).T@(X-X.mean(0)));t=Y.mean(0)-Q@X.mean(0);return Q,t
def errors(E,G,Q,t):return np.linalg.norm(E[:,:3,3]@Q.T+t-G[:,:3,3],axis=1)
def metrics(E,G):
 Q,t=align(E,G);e=errors(E,G,Q,t);r={'ATE_cm':float(np.sqrt(np.mean(e**2))*100)}
 for d in [1,10]:
  if len(E)<=d:continue
  A=np.linalg.inv(G[:-d])@G[d:];C=np.linalg.inv(E[:-d])@E[d:];err=np.linalg.inv(A)@C
  r[f'RPE_t_{d}_cm']=float(np.sqrt(np.mean(np.sum(err[:,:3,3]**2,axis=1)))*100)
  r[f'RPE_r_{d}_deg']=float(np.degrees(np.sqrt(np.mean(Rotation.from_matrix(err[:,:3,:3]).magnitude()**2))))
 return r
def changes(A,C):
 return np.linalg.norm(A[:,:3,3]-C[:,:3,3],axis=1)*100,np.degrees(Rotation.from_matrix(np.linalg.inv(A)[:,:3,:3]@C[:,:3,:3]).magnitude())
def interpolate(W,k,N,eps=.001,times=None):
 # Mirrors Exp(alpha*Log(W1 W0^-1)) W0; W are world-to-camera poses.
 result=[];logs={};base=np.arange(N,dtype=float) if times is None else times
 for frame in range(N):
  a=int(np.searchsorted(k,frame,side='right')-1);assert a>=0
  b=min(a+1,len(k)-1)
  if a==b:pose=W[a].copy()
  else:
   pair=(a,b)
   if pair not in logs:
    L=logm(W[b]@np.linalg.inv(W[a]));assert np.max(abs(np.imag(L)))<1e-10;logs[pair]=np.real(L)
   alpha=(base[frame]-base[k[a]])/(base[k[b]]-base[k[a]]+eps)
   pose=expm(alpha*logs[pair])@W[a]
  result.append(np.linalg.inv(pose))
 return np.asarray(result)
scope=load(F/'CPU_scope.json');assert sha(B/'delivery_manifest.json')==scope['T17_manifest_sha256']
manifest=load(B/'delivery_manifest.json') if not os.environ.get('T18_INPUT_ROOT') else load(F/'receipts/input_snapshot_manifest.json')
for item in manifest['files']:assert sha(B/item['path'])==item['sha256'],item['path']
with np.load(B/'receipts/evaluation_reference.npz') as z:G=mats(z['reference'])
rows=list(csv.DictReader((B/'inputs/DEV200_runtime.csv').open(encoding='utf-8')));rgbtime=np.array([float(r['rgb_timestamp']) for r in rows]);assert len(rows)==len(G)==200 and (np.diff(rgbtime)>0).all()
original_scores=load(B/'receipts/trial_scores.json');all_metrics=[];all_sparse=[];frames=[];kfrows=[];runs=[];intervals=[];batchrows=[];max_historical_diff=0.
for entry in load(B/'receipts/registered_launch.json')['runs']:
 seed=entry['seed'];arm=entry['arm'];folder=B/f'runs/DEV200_s{seed}_{arm}'
 with np.load(folder/'keyframe_poses.npz') as z:data={name:v.copy() for name,v in z.items()}
 with np.load(folder/'trajectories.npz') as z:raw=z['estimated'].copy();ids=z['selected_frames'].copy();ts=z['rgb_timestamps'].copy()
 audit=[json.loads(line) for line in (folder/'keyframe_audit.jsonl').read_text(encoding='utf-8').splitlines()]
 k=data['indices'].astype(int);N=len(raw);assert np.array_equal(ids,np.arange(200)) and np.array_equal(ts,rgbtime)
 assert np.array_equal(k,[r['local_input_frame'] for r in audit]) and np.array_equal(k,[r['selected_frame'] for r in audit]) and np.array_equal(np.arange(len(k)),[r['keyframe'] for r in audit])
 for row in audit:
  assert row['rgb_sha256_resized'] and np.allclose(row['coarse_c2w'],data['coarse'][row['keyframe']],atol=0,rtol=0)
 C=data['coarse'].astype(float);M=data['mapped'].astype(float);afterW=mats(data['droid_after_filler']);D=np.linalg.inv(mats(data['droid_before_injection']))
 seedraw=np.load(folder/'droid_pose_seeds.npy');S=mats(seedraw);O=mats(raw);stored=np.linalg.inv(afterW)
 assert np.array_equal(data['tracked'],data['mapped']) and np.max(abs(S-M))<1e-6
 # Inverse quaternion may be normalized again by native float32 group operations.
 conjugate=seedraw[:,3:]*np.array([-1,-1,-1,1],dtype=np.float32)
 conjugate_error=float(np.max(abs(conjugate-data['droid_after_filler'][:,3:])))
 assert conjugate_error<1e-6
 qround=np.max(abs(Rotation.from_matrix(M[:,:3,:3]).as_matrix()-S[:,:3,:3]));assert qround<1e-6
 variants={'interp_final_DROID':interpolate(np.linalg.inv(D),k,N),'interp_mapping_coarse':interpolate(np.linalg.inv(C),k,N),'interp_injected_mapped':interpolate(np.linalg.inv(S),k,N),'actual_post_filler':O}
 restored=O.copy();restored[k]=S;variants['actual_restore_KF_only']=restored
 exact_interp=interpolate(np.linalg.inv(S),k,N,eps=0);wall_interp=interpolate(np.linalg.inv(S),k,N,eps=0,times=rgbtime)
 Q,t=align(D,G[k]);e={name:errors(T,G,Q,t)*100 for name,T in variants.items()}
 for name,T in variants.items():
  metric=metrics(T,G);common=float(np.sqrt(np.mean(e[name]**2)))
  all_metrics.append({'seed':seed,'arm':arm,'variant':name,**metric,'common_alignment_RMSE_cm':common,'common_KF_RMSE_cm':float(np.sqrt(np.mean(e[name][k]**2))),'common_nonKF_RMSE_cm':float(np.sqrt(np.mean(np.delete(e[name],k)**2)))})
  for part,index in [('KF',k),('nonKF',np.setdiff1d(np.arange(N),k))]:
   all_sparse.append({'seed':seed,'arm':arm,'variant':name,'subset':part,'frames':len(index),'ATE_same_subset_cm':metrics(T[index],G[index])['ATE_cm'],'common_RMSE_cm':float(np.sqrt(np.mean(e[name][index]**2)))})
 original=next(r for r in original_scores if r['seed']==seed and r['arm']==arm)
 actual=next(r for r in all_metrics if r['seed']==seed and r['arm']==arm and r['variant']=='actual_post_filler')
 for key in ['ATE','RPE_t_1','RPE_t_10']:
  old=original['metrics'][key]*100;delta=abs(actual[key+'_cm']-old);max_historical_diff=max(max_historical_diff,delta);assert delta<1e-8
 du,rot=changes(S,O[k]);saved_delta,saved_rot=changes(S,stored);stale,stale_rot=changes(D,S);coarse_stale,_=changes(D,C)
 sparse_common={name:errors(T,G[k],Q,t)*100 for name,T in [('DROID_final',D),('coarse',C),('mapped',S),('returned_duplicate',O[k])]}
 for ordinal,frame in enumerate(k):
  right=min(ordinal+1,len(k)-1);A=np.linalg.inv(D[ordinal])@D[right];Z=np.linalg.inv(S[ordinal])@S[right];rel=np.linalg.inv(A)@Z
  kfrows.append({'seed':seed,'arm':arm,'KF_ordinal':ordinal,'frame':int(frame),'rgb_timestamp':rgbtime[frame],'batch_start':int(frame//16*16),'temporary_index':int(len(k)+frame%16),'right_KF_frame':int(k[right]),'duplicate_translation_cm':float(du[ordinal]),'duplicate_rotation_deg':float(rot[ordinal]),'stored_anchor_roundtrip_translation_cm':float(saved_delta[ordinal]),'mapping_coarse_vs_final_DROID_cm':float(coarse_stale[ordinal]),'injected_seed_vs_final_DROID_cm':float(stale[ordinal]),'relative_two_anchor_translation_change_cm':float(np.linalg.norm(rel[:3,3])*100),'relative_two_anchor_rotation_change_deg':float(np.degrees(Rotation.from_matrix(rel[:3,:3]).magnitude())),**{name+'_common_error_cm':float(value[ordinal]) for name,value in sparse_common.items()}})
 delta_sse=e['actual_post_filler']**2-e['interp_injected_mapped']**2
 KF=np.isin(np.arange(N),k);move,move_rot=changes(variants['interp_injected_mapped'],O)
 for frame in range(N):
  left=int(np.searchsorted(k,frame,side='right')-1);right=min(left+1,len(k)-1)
  alpha=0. if left==right else (frame-k[left])/(k[right]-k[left]+.001)
  alpha_time=0. if left==right else (rgbtime[frame]-rgbtime[k[left]])/(rgbtime[k[right]]-rgbtime[k[left]])
  frames.append({'seed':seed,'arm':arm,'frame':frame,'rgb_timestamp':rgbtime[frame],'is_keyframe':bool(KF[frame]),'left_KF':int(k[left]),'right_KF':int(k[right]),'batch_start':frame//16*16,'source_alpha':float(alpha),'walltime_alpha_noeps':float(alpha_time),'actual_vs_initializer_translation_cm':float(move[frame]),'actual_vs_initializer_rotation_deg':float(move_rot[frame]),'delta_SSE_vs_initializer_cm2':float(delta_sse[frame]),**{name+'_common_error_cm':float(value[frame]) for name,value in e.items()}})
 for left in range(len(k)):
  right=min(left+1,len(k)-1);ix=np.arange(k[left],k[right] if right>left else N,dtype=int)
  intervals.append({'seed':seed,'arm':arm,'left_KF':int(k[left]),'right_KF':int(k[right]),'frames':len(ix),'delta_SSE_cm2':float(delta_sse[ix].sum()),'max_actual_vs_init_translation_cm':float(move[ix].max())})
 for start in range(0,N,16):
  ix=np.arange(start,min(start+16,N));batchrows.append({'seed':seed,'arm':arm,'batch_start':start,'frames':len(ix),'delta_SSE_cm2':float(delta_sse[ix].sum()),'max_actual_vs_init_translation_cm':float(move[ix].max())})
 j=int(du.argmax());runs.append({'seed':seed,'arm':arm,'keyframes':k.tolist(),'identity_timestamp_audit':'PASS','inverse_quaternion_roundoff_max_abs':conjugate_error,'seed_matrix_roundtrip_max_abs':float(np.max(abs(S-M))),'stored_anchor_max_translation_cm':float(saved_delta.max()),'stored_anchor_max_rotation_deg':float(saved_rot.max()),'duplicate_max_translation_cm':float(du.max()),'duplicate_max_frame':int(k[j]),'DROID_final_vs_mapping_coarse_max_cm':float(coarse_stale.max()),'DROID_final_vs_injected_max_cm':float(stale.max()),'epsilon_interpolation_max_effect_cm':float(changes(variants['interp_injected_mapped'],exact_interp)[0].max()),'wall_time_vs_frame_time_max_effect_cm':float(changes(exact_interp,wall_interp)[0].max()),'delta_SSE_all_cm2':float(delta_sse.sum()),'delta_SSE_KF_cm2':float(delta_sse[KF].sum()),'delta_SSE_nonKF_cm2':float(delta_sse[~KF].sum()),'common_alignment_Q':Q.tolist(),'common_alignment_t':t.tolist()})
 (F/'arrays').mkdir(exist_ok=True);np.savez_compressed(F/f'arrays/s{seed}_{arm}.npz',**variants,groundtruth=G,indices=k,seed_c2w=S,final_DROID_c2w=D,coarse_c2w=C,stored_anchor_after_filler=stored,common_Q=Q,common_t=t)
for name,values in [('T18_阶段轨迹指标.csv',all_metrics),('T18_同一子集指标.csv',all_sparse),('T18_逐帧误差分解.csv',frames),('T18_关键帧身份与偏移.csv',kfrows),('T18_区间误差分解.csv',intervals),('T18_批次误差分解.csv',batchrows)]:write_csv(F/name,values)
assert 'torch' not in sys.modules and 'lietorch' not in sys.modules
summary={'task':'T18','status':'CPU_ANALYSIS_COMPLETED','scope':scope,'runs':runs,'stage_metrics':all_metrics,'subset_metrics':all_sparse,'original_dense_metrics_max_difference_cm':max_historical_diff,'input_file_manifest_verified':len(manifest['files']),'input_root_used':str(B),'input_manifest_mode':'snapshot' if os.environ.get('T18_INPUT_ROOT') else 'original_281_file_manifest','CPU_only':True,'remote_connections':0,'GPU_or_SLAM_runs':0,'no_torch_or_native_import':True,'interpolation_is_mathematical_counterfactual_not_saved_GPU_stage':True}
dump(F/'receipts/analysis.json',summary);print(json.dumps({'status':summary['status'],'runs':[{k:r[k] for k in ['seed','arm','duplicate_max_frame','duplicate_max_translation_cm','stored_anchor_max_translation_cm','delta_SSE_all_cm2','delta_SSE_KF_cm2','delta_SSE_nonKF_cm2']} for r in runs]},indent=2))
