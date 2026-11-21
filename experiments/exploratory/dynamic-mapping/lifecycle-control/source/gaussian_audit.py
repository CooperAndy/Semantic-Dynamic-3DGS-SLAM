import json, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['P5_AUDIT_ROOT']); OUT=Path(os.environ['P5_AUDIT_OUT']); POLICY=os.environ['P5_POLICY']; rows={}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def capture(frame, candidate, depth, before):
    candidate=candidate.detach().reshape(-1).bool(); depth=depth.detach().reshape(-1).bool()
    state_path=ROOT/'p5_window/sidecars/D3'/f'{frame:06d}_state.npy'
    state=torch.as_tensor(np.load(state_path).reshape(-1),device=candidate.device)
    assert candidate.numel()==depth.numel()==state.numel()
    names={'static':0,'uncertain':1,'dynamic':2,'ignore':255};count=lambda x:int(x.sum().item())
    assert all(int(x) in names.values() for x in torch.unique(state).tolist())
    allowed=(state==0) if POLICY=='D3' else torch.ones_like(candidate)
    accepted=candidate & allowed; created=accepted & depth
    row={'frame':frame,'policy':POLICY,'kind':'initialization' if frame==0 else 'addition','pixels':len(state),'valid_depth_pixels':count(depth),'candidate_before_state':count(candidate),'candidate_after_state':count(accepted),'candidate_after_depth':count(created),'rejected_by_state':count(candidate & ~allowed),'rejected_invalid_depth_after_state':count(accepted & ~depth),'before_total':before,'state_sha256':digest(state_path),'track_id':'unknown','birth_frame':'window_frame_index'}
    for name,code in names.items():
        m=state==code;row['state_'+name]=count(m);row['candidate_'+name]=count(candidate&m);row['candidate_valid_'+name]=count(candidate&m&depth);row['rejected_'+name]=count(candidate&m&~allowed);row['rejected_valid_'+name]=count(candidate&m&~allowed&depth)
    assert row['candidate_before_state']==sum(row['candidate_'+x] for x in names)
    assert row['candidate_before_state']==row['candidate_after_state']+row['rejected_by_state']
    assert row['candidate_after_state']==row['candidate_after_depth']+row['rejected_invalid_depth_after_state']
    rows[frame]=row

def finish(frame,total,seconds,peak):
    row=rows[frame];row['actual_created']=total-row['before_total'];row['total_gaussians']=int(total);row['frame_elapsed_seconds']=seconds;row['torch_peak_allocated_bytes']=peak
    assert row['actual_created']==row['candidate_after_depth'],row
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'creation_audit.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
