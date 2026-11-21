"""CPU inspection of future T19 native traces, with no GT or parameter tuning."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def positions_c2w(w2c):
    x=np.asarray(w2c,dtype=float)
    R=Rotation.from_quat(x[:,3:]).as_matrix()
    return -np.einsum('nji,nj->ni',R,x[:,:3])

def inspect(root):
    root=Path(root)
    events=[json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    groups={}
    for event in events:
        if event['iteration']<0:
            continue
        key=(tuple(event['frames']),event['iteration'])
        stages=groups.setdefault(key,{})
        if event['stage'] in stages:
            raise ValueError('Duplicate trace stage in one batch/iteration')
        with np.load(root/event['file']) as z:
            stages[event['stage']]={k:v.copy() for k,v in z.items()}
    rows=[]
    for (frames,iteration),stages in sorted(groups.items()):
        required={'graph_before_ba','native_before_ba','native_after_ba','graph_after_ba'}
        if not required <= stages.keys():
            raise ValueError('Incomplete real trace; mock_step fixtures are not native diagnostic data')
        g0=stages['graph_before_ba'];n0=stages['native_before_ba'];n1=stages['native_after_ba'];g1=stages['graph_after_ba']
        target=n0['target'].transpose(0,2,3,1);weight=n0['effective_weight'].transpose(0,2,3,1)
        before=g0['projected_before'][0];after=g1['projected_after'][0]
        if target.shape!=weight.shape or target.shape!=before.shape or after.shape!=before.shape:
            raise ValueError('Unexpected native trace shape')
        if not all(np.isfinite(x).all() for x in [target,weight,before,after,n0['poses_w2c'],n1['poses_w2c']]):
            raise ValueError('Nonfinite trace values require inspection; refusing finite-only silent filtering')
        if np.any(weight<0):raise ValueError('Negative native effective weights')
        delta=np.linalg.norm(positions_c2w(n1['poses_w2c'])-positions_c2w(n0['poses_w2c']),axis=1)*100
        R0=Rotation.from_quat(n0['poses_w2c'][:,3:]);R1=Rotation.from_quat(n1['poses_w2c'][:,3:])
        rot=np.degrees((R1*R0.inv()).magnitude())
        e0=float(np.sum(weight*(target-before)**2));e1=float(np.sum(weight*(target-after)**2))
        rows.append({'first_temporary_frame':frames[0],'last_temporary_frame':frames[-1],'iteration':iteration,
                     'max_c2w_step_cm':float(delta.max()),'max_rotation_step_deg':float(rot.max()),
                     'fixed_learned_target_weighted_residual_before':e0,'fixed_learned_target_weighted_residual_after':e1,
                     'effective_weight_nonzero_fraction':float(np.count_nonzero(weight)/weight.size),
                     'learned_delta_max_abs_pixels':float(np.max(abs(g0['learned_delta']))),
                     'projection_valid_fraction_before':float(np.mean(g0['projection_valid']))})
    if not rows:raise ValueError('No real optimization iterations found')
    return rows

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('trace_root',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    rows=inspect(args.trace_root)
    with args.output.open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'iterations':len(rows),'output':str(args.output),'GT_used':False,
                      'limit':'Residual is a fixed learned-target diagnostic surrogate; not the full native BA objective, and cannot establish GT pose accuracy.'}))
