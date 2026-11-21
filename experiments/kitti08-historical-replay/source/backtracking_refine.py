"""Original optimization and gates plus bounded, explicitly rescored step rollback."""
import hashlib,numpy as np,torch
import torch.nn.functional as F
import external_tracking as original
from tracking_policy import residuals,decide

FRACTIONS=(.5,.25,.125,.0625)

def refine(params,variables,data,index,config,core):
    q0=params['cam_unnorm_rots'][...,index].detach().clone()
    t0=params['cam_trans'][...,index].detach().clone()
    T0=original.camera(params,index,core).cpu().numpy()
    _,d0=original.render(params,data,index,core)
    target_rgb=data['im'].detach().cpu().numpy()
    target_depth=data['depth'].detach().cpu().numpy()[0]
    support=(target_depth>0)&np.isfinite(d0[0])&np.isfinite(d0[2]-d0[0]**2)&(d0[1]>.99)
    params,variables,event=original.refine(params,variables,data,index,config,core)
    event['validation_renders']+=2
    event['backtracking_trials']=[]
    event['selection_kind']='original_scored_pre_step'
    if not event['accepted'] and event['map_write_allowed'] and event.get('steps') and 'selected_pre_step' in event:
        endpoint=next(s for s in event['steps'] if s['iteration']==event['selected_pre_step'])
        q1=F.normalize(torch.tensor(endpoint['q_pre'],device=q0.device,dtype=q0.dtype))
        qstart=F.normalize(q0)
        if float((qstart*q1).sum())<0:q1=-q1
        t1=torch.tensor(endpoint['t_pre'],device=t0.device,dtype=t0.dtype)
        event['raw_min_candidate']=event['candidate']
        event['raw_min_decision']={k:event.get(k) for k in ['accepted','reason','gates','translation_m','rotation_deg']}
        for fraction in FRACTIONS:
            q=F.normalize((1-fraction)*qstart+fraction*q1)
            t=(1-fraction)*t0+fraction*t1
            with torch.no_grad():
                params['cam_unnorm_rots'][...,index]=q
                params['cam_trans'][...,index]=t
            rgb,d=original.render(params,data,index,core)
            event['validation_renders']+=2
            candidate=residuals(rgb,d[0],target_rgb,target_depth,support,d[1],config['tracking']['loss_weights'])
            T=original.camera(params,index,core).cpu().numpy()
            delta=T0@np.linalg.inv(T)
            translation=float(np.linalg.norm(delta[:3,3]))
            angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
            decision=decide(event['initial'],candidate,translation,angle)
            trial=dict(fraction=fraction,candidate=candidate,translation_m=translation,rotation_deg=angle,
                       q=q.cpu().numpy().tolist(),t=t.cpu().numpy().tolist(),**decision)
            event['backtracking_trials'].append(trial)
            if decision['accepted']:
                event.update(decision,candidate=candidate,translation_m=translation,rotation_deg=angle,
                             selection_kind='rescored_backtracked_pre_step',backtracking_fraction=fraction)
                event['reason']='accepted_backtracking'
                break
            if not decision['map_write_allowed']:
                event.update(decision)
                break
        if not event['accepted']:
            with torch.no_grad():
                params['cam_unnorm_rots'][...,index]=q0
                params['cam_trans'][...,index]=t0
    event['selected_w2c']=original.camera(params,index,core).cpu().numpy().tolist()
    event['map_after_sha256']=original.map_hash(params)
    assert event['map_before_sha256']==event['map_after_sha256']
    event['initial_support_sha256']=hashlib.sha256(support.tobytes()).hexdigest()
    return params,variables,event
