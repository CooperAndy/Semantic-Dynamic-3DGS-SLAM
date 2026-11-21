"""Single-window exact G19C compute function with current/past inputs only."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import cv2,numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

R=Path(sys.argv[1]);context=json.loads(Path(sys.argv[2]).read_text())['window']
old=Path('/root/autodl-tmp/stage3_g19_past_conf_20261029_p47149');sys.path.insert(0,str(old))
spec=importlib.util.spec_from_file_location('frozen_G19C_compute',old/'producer.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raw=np.loadtxt(context['primary_pose_path'],dtype=np.float64);assert raw.shape==(260,12)
assert sha(context['primary_pose_path'])==context['primary_pose_sha256']
poses=np.tile(np.eye(4),(260,1,1));poses[:,:3]=raw.reshape(-1,3,4)
K=np.asarray(context['K'],np.float64);calib=dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),baseline_m=float(context['projection_beta']/K[0,0]))
for name in ['depth_confhard','confidence']:(R/'cache'/name).mkdir(parents=True,exist_ok=False)
rows=[];previous_depth=previous_pose=None;image=Path(context['input_view'])
for i in range(260):
    left=image/'image_2'/f'{i:06}.png';right=image/'image_3'/f'{i:06}.png'
    L=cv2.cvtColor(cv2.imread(str(left)),cv2.COLOR_BGR2RGB);RR=cv2.cvtColor(cv2.imread(str(right)),cv2.COLOR_BGR2RGB)
    dl=module.legacy.load_depth(R/'raw_left'/f'{i:06}_depth.npy',tuple(context['shape']))
    dr=module.legacy.load_depth(R/'raw_right'/f'{i:06}_depth.npy',tuple(context['shape']))
    result=module.compute(L,RR,dl,dr,poses[i],previous_depth,previous_pose,calib)
    d=R/'cache/depth_confhard'/f'{i:06}_depth.npy';c=R/'cache/confidence'/f'{i:06}_confidence.npy'
    np.save(d,result['depth_confhard'],allow_pickle=False);np.save(c,result['confidence'],allow_pickle=False)
    rows.append(dict(frame_id=i,source_frame_id=context['source_start']+i,rgb=str(left),right=str(right),depth=str(d),confidence=str(c),hashes=dict(rgb=sha(left),right=sha(right),depth=sha(d),confidence=sha(c))))
    previous_depth=dl;previous_pose=poses[i]
    if (i+1)%100==0:print('SEALED_PAST_ONLY',i+1,flush=True)
m=dict(task_id='S23_FORMAL_PAST_ONLY_STEREO',pose_source='ORB_estimate',pose_file=context['primary_pose_path'],pose_sha256=context['primary_pose_sha256'],frame_convention='P2',frame_count=260,K=K.tolist(),H=context['shape'][0],W=context['shape'][1],depth_units='meters',valid_depth_range_m=[.5,80],confidence_threshold=.8,GT_runtime=False,temporal_scope='current and previous only',rows=rows)
(R/'runtime.json').write_text(json.dumps(m,indent=2))
(R/'confidence_receipt.json').write_text(json.dumps(dict(status='COMPLETE_EXACT_G19C_FUNCTION_SINGLE_WINDOW',compute_source_sha256=sha(old/'producer.py'),legacy_formula_sha256=sha(old/'provenance/legacy_confidence.py'),GT_prediction=False),indent=2))
