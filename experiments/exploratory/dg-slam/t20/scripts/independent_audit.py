"""Local independent CPU verification of sources, paired state, metrics and ID steps."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parents[1];OLD=F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrices(x):
    T=np.repeat(np.eye(4)[None],len(x),axis=0);T[:,:3,:3]=Rotation.from_quat(x[:,3:]).as_matrix();T[:,:3,3]=x[:,:3];return T
server=load(F/'receipts/server_scoring.json');local=load(F/'independent_cpu/receipts/scoring.json')
maximum=0.
for A,B in zip(server['summaries'],local['summaries']):
    assert A.keys()==B.keys()
    for key in A:
        if isinstance(A[key],(int,float)):
            maximum=max(maximum,abs(A[key]-B[key]));assert abs(A[key]-B[key])<=1e-9,(A['case'],key)
        else:assert A[key]==B[key]
historical=0
for name in ['CHY_T17_NO_GS_vs_ID完整轨迹_20261012','CHY_T19_补全接口修复与CPU验证_20261012']:
    root=F.parent/name
    for item in load(root/'delivery_manifest.json')['files']:
        file=root/item['path'];assert sha(file)==item['sha256'];historical+=1
reg=load(F/'T19_diagnostic_registration.json')
for item in reg['candidate_source']:assert sha(F/'source'/item['path'])==item['sha256']
spec=importlib.util.spec_from_file_location('frozen_camera_audit',OLD/'scripts/audit_camera.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
paired=[];reproduction=[];camera=[]
for arm in ['NO_GS_REFINE','LOCAL_ID']:
    folder=F/f'runs/DEV200_s2718_{arm}'
    capture=load(folder/'state_capture.json');assert sha(folder/'native_state.pt')==capture['snapshot_sha256']
    names=['N_MAP_LEGACY','N_MAP_FIXED'] if arm=='NO_GS_REFINE' else ['I_MAP_LEGACY','I_MAP_FIXED']
    states=[]
    for name in [c['case'] for c in reg['future_filler_cases'] if c['arm']==arm]:
        state=load(folder/name/'same_state_before_policy.json')
        assert state['expected']==state['actual']==capture['active_hashes'];states.append(name)
    with np.load(folder/names[0]/'trajectory32.npz') as z:legacy=z['estimated_c2w'].copy();frames=z['frames'].copy();keys=z['keyframe_indices'].copy()
    with np.load(folder/names[1]/'trajectory32.npz') as z:fixed=z['estimated_c2w'].copy()
    mask=~np.isin(frames,keys)
    assert np.array_equal(legacy[mask],fixed[mask]),'Inspect unexpected non-keyframe change'
    paired.append({'arm':arm,'cases_with_identical_prepolicy_hashes':states,'nonKF_outputs_bitwise_equal_legacy_vs_fixed':True,'nonKF_frames':int(mask.sum())})
    with np.load(folder/'keyframe_poses.npz') as z:new={k:v.copy() for k,v in z.items()}
    with np.load(OLD/f'runs/DEV200_s2718_{arm}/keyframe_poses.npz') as z:previous={k:v.copy() for k,v in z.items()}
    with np.load(OLD/f'runs/DEV200_s2718_{arm}/trajectories.npz') as z:old_dense=z['estimated'][144:176].copy()
    assert np.array_equal(new['indices'],previous['indices'])
    reproduction.append({'arm':arm,'same_keyframe_ids_as_T17':True,
        'native_DROID_anchor_array_max_abs_difference_from_T17':float(np.max(np.abs(new['droid_before_injection']-previous['droid_before_injection']))),
        'mapped_anchor_translation_max_difference_from_T17_cm':float(np.max(np.linalg.norm(new['mapped'][:,:3,3]-previous['mapped'][:,:3,3],axis=1)))*100,
        'legacy32_position_max_difference_from_T17_full_filler_cm':float(np.max(np.linalg.norm(legacy[:,:3]-old_dense[:,:3],axis=1)))*100,
        'limit':'Fresh state rebuild and selected two-batch replay, not a promise of bitwise reproduction of the old full200 filler; do not transfer T18 attribution percentages'})
    for path in folder.glob('camera_kf*.npz'):
        with np.load(path) as z:assert np.array_equal(z['C'],np.eye(6))
        camera.append(module.audit_trace(path,[.02]*3+[np.pi/180]*3))
assert camera
result={'status':'PASS','server_vs_local_metric_max_abs_difference':maximum,'metric_tolerance':1e-9,
    'historical_T17_T19_files_unchanged':historical,'T19_candidate_files_match':6,
    'paired_state_and_outputs':paired,'reproduction_checks':reproduction,
    'ID_camera_trace_audits':len(camera),'camera_iterations':sum(c['iterations'] for c in camera),
    'camera_proposals':sum(c['proposals'] for c in camera),'camera_pose_max_abs_difference':max(c['pose_max_abs'] for c in camera),
    'formal_acceptance':False,'GPU_reruns':0}
(F/'receipts/independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
