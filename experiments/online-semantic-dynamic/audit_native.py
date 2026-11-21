"""Independent metric birth geometry / recorded pose consistency, without GT."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from online_state import OnlineState

def audit_run(directory):
    R=Path(directory);receipt=json.loads((R/'receipt.json').read_text())
    assert receipt['status']=='PASS_NATIVE_MAPPING_INTEGRATION'
    events=[json.loads(x) for x in (R/'backend_events.jsonl').read_text().splitlines()]
    maximum=0.;pose_m=0.;pose_deg=0.;points=0;files=0
    for file in sorted((R/'birth_geometry').glob('*.npz')):
        frame=int(file.stem)
        with np.load(file,allow_pickle=False) as z:
            ids=z['ids'];cloud=z['pointcloud'];depth=z['sampled_depth'];K=z['intrinsics'].astype(np.float64);W=z['w2c'].astype(np.float64);H,width=z['image_shape']
            assert ids.ndim==1 and len(ids)==len(cloud)==len(depth) and len(np.unique(ids))==len(ids)
            x=ids%width;y=ids//width
            camera=np.c_[(x-K[0,2])/K[0,0]*depth,(y-K[1,2])/K[1,1]*depth,depth]
            world=np.linalg.solve(W[:3,:3],(camera-W[:3,3]).T).T
            error=np.linalg.norm(world-cloud[:,:3],axis=1)
            if len(error):maximum=max(maximum,float(error.max()))
            selected=np.asarray(events[frame]['selected_w2c'],np.float64);delta=W@np.linalg.inv(selected)
            pose_m=max(pose_m,float(np.linalg.norm(delta[:3,3])));pose_deg=max(pose_deg,float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())))
            points+=len(cloud);files+=1
    assert maximum<=.0003 and pose_m<=.0003 and pose_deg<=.002
    with np.load(R/'estimated_trajectory.npz') as z:poses=z['w2c']
    assert len(events)==len(poses)==receipt['frames']
    assert all(np.array_equal(T,np.asarray(e['selected_w2c'],np.float32)) for T,e in zip(poses,events))
    ids=np.load(R/'source_gaussian_index.npy',allow_pickle=False);state=OnlineState.load(R/'semantic_state_final.npz',ids)
    with np.load(R/'params.npz',allow_pickle=False) as z:assert len(z['means3D'])==len(ids)
    assert len(set(state.ids.tolist()))==len(ids)
    assert not any(int(x)%20==19 for x in state.processed)
    result=dict(status='PASS_INDEPENDENT_BIRTH_POSE_AND_SEMANTIC_IDENTITIES',birth_files=files,birth_points=points,
                max_birth_error_m=maximum,max_birth_pose_error_m=pose_m,max_birth_pose_error_deg=pose_deg,
                frames=len(poses),GT_used=False,semantic_source_frames=len(state.processed))
    (R/'independent_numeric_audit.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import sys
    print(json.dumps(audit_run(sys.argv[1])))
