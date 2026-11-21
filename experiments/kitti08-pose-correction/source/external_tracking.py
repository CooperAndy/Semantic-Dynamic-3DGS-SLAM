"""C02/C03 candidate. GPU renderer still requires the separately recorded smoke gate."""
import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from tracking_policy import DEFAULTS,support_info,residuals,decide
MAP_KEYS=('means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales')

def set_external(params,w2c,index,core):
    with torch.no_grad():
        params['cam_unnorm_rots'][...,index]=core.matrix_to_quaternion(w2c[:3,:3].unsqueeze(0))
        params['cam_trans'][...,index]=w2c[:3,3]

def camera(params,index,core):
    T=torch.eye(4,device=params['cam_trans'].device)
    T[:3,:3]=core.build_rotation(F.normalize(params['cam_unnorm_rots'][...,index].detach()))[0]
    T[:3,3]=params['cam_trans'][...,index].detach()
    return T

def render(params,data,index,core):
    with torch.no_grad():
        g=core.transform_to_frame(params,index,gaussians_grad=False,camera_grad=False)
        rgb,_,_,=core.Renderer(raster_settings=data['cam'])(**core.transformed_params2rendervar(params,g))
        ds,_,_,=core.Renderer(raster_settings=data['cam'])(**core.transformed_params2depthplussilhouette(params,data['w2c'],g))
        return rgb.cpu().numpy(),ds.cpu().numpy()

def map_hash(params):
    h=hashlib.sha256()
    for k in MAP_KEYS:h.update(params[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

def refine(params,variables,data,index,config,core):
    before=map_hash(params);initial_q=params['cam_unnorm_rots'][...,index].detach().clone()
    initial_t=params['cam_trans'][...,index].detach().clone();T0=camera(params,index,core).cpu().numpy()
    rgb0,ds0=render(params,data,index,core);target_rgb=data['im'].detach().cpu().numpy()
    target_depth=data['depth'].detach().cpu().numpy()[0];valid=target_depth>0
    uncertainty=ds0[2]-ds0[0]**2
    support=valid&np.isfinite(ds0[0])&np.isfinite(uncertainty)&(ds0[1]>.99)
    if '_common_validation_support' in data:
        support=np.asarray(data['_common_validation_support'],dtype=bool).copy()
        assert support.shape==target_depth.shape and not (support & (target_depth<=0)).any()
    info=support_info(support,valid);event={'frame':index,'map_before_sha256':before,'support':info,'steps':[],'validation_renders':2}
    if not info['sufficient']:
        event.update(accepted=False,reason='insufficient_support',map_write_allowed=True)
    else:
        weights=config['tracking']['loss_weights'];ini=residuals(rgb0,ds0[0],target_rgb,target_depth,support,ds0[1],weights)
        optimizer=core.initialize_optimizer(params,config['tracking']['lrs'],tracking=True)
        best=float('inf');qbest=initial_q;tbest=initial_t;numerical=False
        for it in range(config['tracking']['num_iters']):
            qpre=params['cam_unnorm_rots'][...,index].detach().clone();tpre=params['cam_trans'][...,index].detach().clone()
            loss,variables,_=core.get_loss(params,data,variables,index,weights,config['tracking']['use_sil_for_loss'],
                config['tracking']['sil_thres'],config['tracking']['use_l1'],False,tracking=True)
            if not torch.isfinite(loss).item():numerical=True;break
            loss.backward()
            gradients=[params[k].grad[...,index] if params[k].grad is not None else None for k in ['cam_unnorm_rots','cam_trans']]
            if any(g is None or not torch.isfinite(g).all().item() for g in gradients):numerical=True;break
            norm=sum(float(g.square().sum().item()) for g in gradients)**.5
            if norm==0:event['zero_gradient']=True;break
            if loss.item()<best:best=float(loss.item());qbest=qpre;tbest=tpre;event['selected_pre_step']=it
            step={'iteration':it,'loss_pre':float(loss.item()),'gradient_norm':norm,
                'q_pre':qpre.cpu().numpy().tolist(),'t_pre':tpre.cpu().numpy().tolist(),
                'q_gradient':gradients[0].detach().cpu().numpy().tolist(),'t_gradient':gradients[1].detach().cpu().numpy().tolist()}
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            finite_post=all(torch.isfinite(params[k][...,index]).all().item() for k in ['cam_unnorm_rots','cam_trans'])
            step.update(q_post=params['cam_unnorm_rots'][...,index].detach().cpu().numpy().tolist() if finite_post else None,
                        t_post=params['cam_trans'][...,index].detach().cpu().numpy().tolist() if finite_post else None,
                        finite_post=finite_post)
            event['steps'].append(step)
            if not finite_post:numerical=True;break
        with torch.no_grad():params['cam_unnorm_rots'][...,index]=qbest;params['cam_trans'][...,index]=tbest
        rgb1,ds1=render(params,data,index,core);event['validation_renders']+=2
        cand=residuals(rgb1,ds1[0],target_rgb,target_depth,support,ds1[1],weights)
        T1=camera(params,index,core).cpu().numpy();delta=T0@np.linalg.inv(T1)
        translation=float(np.linalg.norm(delta[:3,3]));angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
        decision=decide(ini,cand,translation,angle)
        if numerical:decision=dict(accepted=False,reason='nonfinite_optimization',map_write_allowed=False)
        if event.get('zero_gradient'):decision=dict(accepted=False,reason='zero_gradient',map_write_allowed=True)
        event.update(decision,initial=ini,candidate=cand,translation_m=translation,rotation_deg=angle)
        if not event['accepted']:
            with torch.no_grad():params['cam_unnorm_rots'][...,index]=initial_q;params['cam_trans'][...,index]=initial_t
    event['map_after_sha256']=map_hash(params)
    assert event['map_after_sha256']==before,'Gaussian parameters changed during tracking'
    event['selected_w2c']=camera(params,index,core).cpu().numpy().tolist()
    for k in ['cam_unnorm_rots','cam_trans']:params[k].grad=None
    return params,variables,event
