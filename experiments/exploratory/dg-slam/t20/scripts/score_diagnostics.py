"""CPU scoring only after all five diagnostics have completed and raw outputs sealed."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.spatial.transform import Rotation

F=Path(__file__).resolve().parents[1]
OLD=Path(os.environ.get('T20_T17_ROOT',str(F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Import only the unchanged mathematical functions, not the old experiment driver.
source=(OLD/'scripts/evaluate.py').read_text(encoding='utf-8')
tree=ast.parse(source)
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('mats','proper','metrics')]
assert len(functions)==3
namespace={'np':np,'Rotation':Rotation}
exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen T17 math only>','exec'),namespace)
mats=namespace['mats'];proper=namespace['proper'];metrics=namespace['metrics']

def fixed_errors(c2w,truth,Q,t):
    return np.linalg.norm(c2w[:,:3,3]@Q.T+t-truth[:,:3,3],axis=1)*100

def w2c_to_c2w(raw):
    A=np.linalg.inv(mats(np.asarray(raw,dtype=np.float64)))
    # Independent closed-form quaternion pose inversion, no homogeneous inverse.
    raw=np.asarray(raw,dtype=float);R=Rotation.from_quat(raw[:,3:]).as_matrix()
    position=-np.einsum('nji,nj->ni',R,raw[:,:3])
    assert np.max(np.abs(position-A[:,:3,3]))<1e-12
    return A

def main():
    launch=load(F/'receipts/launch_v2.json')
    assert launch['status']=='ALL_FIVE_DIAGNOSTICS_COMPLETE'
    seal=load(F/'receipts/raw_seal.json');assert seal['status']=='SEALED_BEFORE_CPU_GT'
    assert seal['protocol_sha256']==sha(F/'protocol.json')
    for rel,h in seal['files'].items():assert sha(F/rel)==h,rel
    assert not (F/'receipts/scoring.json').exists()
    # First GT access in this program occurs below, after the raw integrity checks.
    with np.load(OLD/'receipts/evaluation_reference.npz') as z:truth=mats(z['reference'].astype(np.float64))
    assert len(truth)==200
    P=load(F/'protocol.json');summaries=[];frame_rows=[];iteration_rows=[];alignments={}
    for arm in P['arms']:
        folder=F/f'runs/DEV200_s2718_{arm}'
        manifest=load(folder/'run_manifest.json');assert manifest['status']=='COMPLETED'
        assert manifest['GT_guard']['blocked_GT_opens']==0 and manifest['counts']['input_frames']==200
        with np.load(folder/'native_anchors.npz') as z:
            k=z['indices'].astype(int);native=w2c_to_c2w(z['native_w2c'])
        assert len(k)==24
        X=native[:,:3,3];Y=truth[k,:3,3]
        Q=proper((Y-Y.mean(0)).T@(X-X.mean(0)));t=Y.mean(0)-Q@X.mean(0)
        # Frozen Horn/SVD crosscheck also checks the native anchor alignment.
        check=metrics(truth[k],native)
        assert abs(np.sqrt(np.mean(fixed_errors(native,truth[k],Q,t)**2))/100-check['ATE'])<1e-12
        alignments[arm]={'R':Q.tolist(),'t_m':t.tolist(),'native_anchor_ATE_cm':check['ATE']*100,'keyframe_indices':k.tolist()}
        for case in [c for c in P['future_filler_cases'] if c['arm']==arm]:
            name=case['case'];out=folder/name;gate=load(out/'engineering_gate.json');assert gate['status']=='PASS'
            with np.load(out/'trajectory32.npz') as z:
                frames=z['frames'].astype(int);final=mats(z['estimated_c2w'].astype(float));selected=w2c_to_c2w(z['selected_anchor_w2c'])
            assert np.array_equal(frames,np.arange(144,176))
            init=np.empty((32,4,4));filled=np.zeros(32,dtype=bool)
            events=[json.loads(line) for line in (out/'trace/events.jsonl').read_text(encoding='utf-8').splitlines()]
            groups={}
            for event in events:
                assert not any(event['nonfinite'].values())
                with np.load(out/'trace'/event['file']) as z:values={key:v.copy() for key,v in z.items()}
                if event['stage']=='initialization':
                    ids=values['temporary_frames'].astype(int);positions=ids-144
                    init[positions]=w2c_to_c2w(values['temporary_w2c']);filled[positions]=True
                else:
                    key=(tuple(event['frames']),event['iteration'])
                    groups.setdefault(key,{})[event['stage']]=values
            for i,frame in enumerate(frames):
                if not filled[i]:
                    assert frame in k
                    init[i]=selected[np.where(k==frame)[0][0]];filled[i]=True
            assert filled.all()
            error0=fixed_errors(init,truth[frames],Q,t);error1=fixed_errors(final,truth[frames],Q,t)
            keymask=np.isin(frames,k)
            oldmetrics=metrics(truth[frames],final)
            # Independently vectorized RPE must match the unchanged loop implementation.
            for delta in [1,10]:
                A=np.linalg.inv(truth[frames[:-delta]])@truth[frames[delta:]]
                C=np.linalg.inv(final[:-delta])@final[delta:]
                E=np.linalg.inv(A)@C
                assert abs(np.sqrt(np.mean(np.sum(E[:,:3,3]**2,axis=1)))-oldmetrics[f'RPE_t_{delta}'])<1e-12
            init_rot=Rotation.from_matrix(init[:,:3,:3]);final_rot=Rotation.from_matrix(final[:,:3,:3])
            movement=np.linalg.norm(final[:,:3,3]-init[:,:3,3],axis=1)*100
            for i,frame in enumerate(frames):
                frame_rows.append({'case':name,'arm':arm,'frame':int(frame),'keyframe':bool(keymask[i]),
                    'initial_error_cm':float(error0[i]),'final_error_cm':float(error1[i]),
                    'error_change_cm':float(error1[i]-error0[i]),'SSE_change_cm2':float(error1[i]**2-error0[i]**2),
                    'movement_cm':float(movement[i]),'rotation_movement_deg':float(np.degrees((final_rot[i]*init_rot[i].inv()).magnitude()))})
            divergent=0
            for (temporary,itr),stages in sorted(groups.items()):
                assert set(stages)=={'graph_before_ba','native_before_ba','native_after_ba','graph_after_ba'}
                g0=stages['graph_before_ba'];n0=stages['native_before_ba'];n1=stages['native_after_ba'];g1=stages['graph_after_ba']
                target=n0['target'].transpose(0,2,3,1);weight=n0['effective_weight'].transpose(0,2,3,1)
                before=g0['projected_before'][0];after=g1['projected_after'][0]
                assert target.shape==weight.shape==before.shape==after.shape and np.all(weight>=0)
                residual0=float(np.sum(weight*(target-before)**2));residual1=float(np.sum(weight*(target-after)**2))
                C0=w2c_to_c2w(n0['poses_w2c']);C1=w2c_to_c2w(n1['poses_w2c']);ids=np.array(temporary,dtype=int)
                e0=fixed_errors(C0,truth[ids],Q,t);e1=fixed_errors(C1,truth[ids],Q,t)
                step=np.linalg.norm(C1[:,:3,3]-C0[:,:3,3],axis=1)*100
                diverged=residual1<residual0 and np.sum(e1**2)>np.sum(e0**2)
                divergent+=int(diverged)
                iteration_rows.append({'case':name,'batch_first_frame':int(ids.min()),'batch_last_frame':int(ids.max()),
                    'iteration':int(itr),'temporary_nodes':len(ids),'fixed_target_residual_before':residual0,
                    'fixed_target_residual_after':residual1,'GT_SSE_before_cm2':float(np.sum(e0**2)),
                    'GT_SSE_after_cm2':float(np.sum(e1**2)),'GT_RMSE_before_cm':float(np.sqrt(np.mean(e0**2))),
                    'GT_RMSE_after_cm':float(np.sqrt(np.mean(e1**2))),'max_step_cm':float(step.max()),
                    'residual_down_GT_error_up':bool(diverged),'effective_weight_nonzero_fraction':float(np.count_nonzero(weight)/weight.size),
                    'learned_delta_abs_max_pixels':float(np.max(np.abs(g0['learned_delta'])))})
            summary={'case':name,'arm':arm,'frames':32,'KF_frames':int(keymask.sum()),
                'fixed_native_alignment_RMSE_initial_cm':float(np.sqrt(np.mean(error0**2))),
                'fixed_native_alignment_RMSE_final_cm':float(np.sqrt(np.mean(error1**2))),
                'KF_RMSE_final_cm':float(np.sqrt(np.mean(error1[keymask]**2))),
                'nonKF_RMSE_final_cm':float(np.sqrt(np.mean(error1[~keymask]**2))),
                'RPE1_translation_cm':oldmetrics['RPE_t_1']*100,'RPE10_translation_cm':oldmetrics['RPE_t_10']*100,
                'RPE1_rotation_deg':float(np.degrees(oldmetrics['RPE_r_1'])),
                'max_movement_cm':float(movement.max()),'SSE_change_KF_cm2':float(np.sum(error1[keymask]**2-error0[keymask]**2)),
                'SSE_change_nonKF_cm2':float(np.sum(error1[~keymask]**2-error0[~keymask]**2)),
                'within_iteration_residual_down_GT_up':divergent,'iterations':len(groups),
                'returned_KF_w2c_matrix_max_abs_difference':gate['returned_KF_w2c_matrix_max_abs_difference']}
            assert len(groups)==12
            summaries.append(summary)
    write_csv(F/'T20_五组诊断结果.csv',summaries)
    write_csv(F/'T20_逐帧误差与移动.csv',frame_rows)
    write_csv(F/'T20_逐轮目标与GT误差.csv',iteration_rows)
    dump(F/'receipts/scoring.json',{'status':'PASS','scope':'Exposed DEV seed2718 frames144:176 only; not formal adoption or full200-frame ATE',
        'raw_seal_verified_files':len(seal['files']),'GT_reference_sha256':sha(OLD/'receipts/evaluation_reference.npz'),
        'metric_source_sha256':sha(OLD/'scripts/evaluate.py'),'alignment':'Per-arm native DROID24 no-scale SE3 alignment, fixed across all cases',
        'alignments':alignments,'summaries':summaries,'frame_rows':len(frame_rows),'iteration_rows':len(iteration_rows),
        'independent_RPE_and_pose_inverse_check':'PASS','full_BA_objective_claim':False,'formal_acceptance':False})
    print(json.dumps({'status':'PASS','cases':len(summaries),'frame_rows':len(frame_rows),'iteration_rows':len(iteration_rows)},ensure_ascii=False))

if __name__=='__main__':main()
