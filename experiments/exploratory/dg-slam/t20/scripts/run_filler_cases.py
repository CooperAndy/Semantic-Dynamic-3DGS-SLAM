"""Execute the five T19-registered mechanisms after each full DEV state build."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from lietorch import SE3
from runtime_boot import F,P,dump,sha
from dg_slam.filler_adapter import configure_filler
from state_control import NativeState
from analyze_trace import inspect


def run_cases(model, mapped, data, output, arm, audit):
    n=int(model.video.counter.value)
    ids=model.video.tstamp[:n].detach().cpu().numpy().astype(int)
    assert n==24, 'T19 registered alignment expects native 24 anchors; inspect before changing registration'
    assert len(data)==200 and [row[0] for row in data[144:176]]==list(range(144,176))
    original=model.video.poses[:n].detach().clone()
    np.savez(output/'native_anchors.npz',indices=ids,native_w2c=original.cpu().numpy(),mapped_c2w=mapped)
    snap=NativeState(model,temporary_capacity=16)
    snap.save(output/'native_state.pt')
    dump(output/'state_capture.json',{'status':'CAPTURED_BEFORE_POLICY','keyframes':n,'tensor_slots':snap.capacity,
        'active_hashes':snap.expected,'snapshot_sha256':sha(output/'native_state.pt'),
        'original_anchor_policy':'native DROID, not overwritten by mapping snapshots',
        'unused_buffer_slots':'Only active plus 16 potentially reused temporary slots captured; remaining buffer slots not indexed by these cases'})
    results=[]
    cases=[case for case in P['future_filler_cases'] if case['arm']==arm]
    for case in cases:
        started=time.time()
        restored=snap.restore(model)
        name=case['case'];case_output=output/name
        print('CASE_START',name,flush=True)
        policy=configure_filler(model,mapped,arm,case['anchors'],case_output,
                                mixed_geometry_diagnostic=case['mixed_geometry_diagnostic'],
                                preserve_keyframes=case['preserve_KF'],trace_frames=range(144,176))
        selected=model.video.poses[:n].detach().clone()
        dump(case_output/'same_state_before_policy.json',{'status':'PASS','expected':snap.expected,'actual':restored})
        trajectory=model.terminate_woBA(data[144:176])
        assert trajectory.shape==(32,7) and np.isfinite(trajectory).all()
        assert np.allclose(np.linalg.norm(trajectory[:,3:],axis=1),1,atol=1e-4)
        assert int(model.video.counter.value)==n and torch.equal(model.video.poses[:n],selected)
        restored_w2c=SE3(torch.as_tensor(trajectory,device='cuda')).inv().matrix().cpu().numpy()
        selected_mats=SE3(selected).matrix().cpu().numpy()
        matches=[(i,int(np.where(ids==frame)[0][0])) for i,frame in enumerate(range(144,176)) if frame in ids]
        anchor_error=max(float(np.max(np.abs(restored_w2c[i]-selected_mats[k]))) for i,k in matches)
        if case['preserve_KF']: assert anchor_error<=1e-6,anchor_error
        np.savez(case_output/'trajectory32.npz',estimated_c2w=trajectory,frames=np.arange(144,176),
                 keyframe_indices=ids,selected_anchor_w2c=selected.cpu().numpy())
        trace=model.traj_filler.diagnostic_trace
        events=[json.loads(line) for line in (case_output/'trace/events.jsonl').read_text().splitlines()]
        assert len(events)==50, len(events)
        assert sum(event['stage']=='initialization' for event in events)==2
        assert all(not any(event['nonfinite'].values()) for event in events)
        steps=inspect(case_output/'trace');assert len(steps)==12
        assert {tuple(event['frames']) for event in events if event['stage']=='initialization'}=={
            tuple(frame for frame in range(a,a+16) if not case['preserve_KF'] or frame not in ids) for a in [144,160]}
        assert audit['blocked_GT_opens']==0
        receipt={'status':'PASS','case':name,'policy':policy,'frames':32,'keyframes_in_request':[int(ids[k]) for _,k in matches],
                 'same_pre_policy_state':True,'anchors_unchanged':True,'counter_restored':True,
                 'returned_KF_w2c_matrix_max_abs_difference':anchor_error,'iterations':len(steps),
                 'trace_records':len(events),'trace_reserved_bytes':trace.bytes,'GT_guard':audit,'seconds':time.time()-started,
                 'quality_gate':None,'formal':False,'native_BA_and_weights_logged':True}
        dump(case_output/'engineering_gate.json',receipt)
        results.append(receipt)
        print('CASE_PASS',name,'seconds',receipt['seconds'],flush=True)
    snap.restore(model)
    assert torch.equal(model.video.poses[:n],original)
    dump(output/'diagnostic_gate.json',{'status':'PASS','arm':arm,'cases':results,'native_state_restored_at_end':True,
         'formal':False,'GT_guard':audit,'new_adoption_claim':False})
    return results
