"""Native semantic channels with geometry-only gradients and a fixed pixel set."""
import numpy as np
import torch
import torch.nn.functional as F

GEOMETRY=('means3D','log_scales','unnorm_rotations')


def render_channels(core,params,data,frame,features,geometry_grad):
    # Opacity, RGB, camera, and semantic evidence cannot absorb semantic loss.
    p={k:(v if geometry_grad and k in GEOMETRY else v.detach()) for k,v in params.items()}
    transformed=core.transform_to_frame(p,frame,gaussians_grad=geometry_grad,camera_grad=False)
    padded=F.pad(features,(0,(-features.shape[1])%3));chunks=[]
    for start in range(0,padded.shape[1],3):
        var=core.transformed_params2rendervar(p,transformed)
        var['colors_precomp']=padded[:,start:start+3].detach()
        rgb,_,_=core.Renderer(raster_settings=data['cam'])(**var);chunks.append(rgb)
    return torch.cat(chunks,0)[:features.shape[1]]


def class_loss(mass,labels,mask):
    if not bool(mask.any()):return mass.sum()*0.
    # Fixed mask: no dropping pixels because optimization made them disappear.
    denominator=mass.sum(0).clamp_min(1e-8)
    classes=labels.clamp(0,mass.shape[0]-1)
    selected=mass.gather(0,classes[None]).squeeze(0)
    probability=(selected/denominator).clamp(1e-6,1.)
    return -torch.log(probability[mask]).mean()


def initial_mask(mass,alpha,labels,confidence,depth,allow=None):
    mask=(labels>=0)&(labels<19)&torch.isfinite(confidence)&(confidence>=.8)&torch.isfinite(depth)&(depth>0)
    mask &= torch.isfinite(alpha)&(alpha>=.05)&(mass.sum(0)>=.5*alpha)
    if allow is not None:mask &= allow
    return mask.detach().clone()


def semantic_gradients(core,params,data,frame,features,labels,mask,weight=.1):
    if not bool(mask.any()):return None,0.
    mass=render_channels(core,params,data,frame,features,True)
    loss=class_loss(mass,labels,mask)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite semantic loss')
    # Explicit allowlist prevents color/opacity/camera or evidence gradients.
    grads=torch.autograd.grad(float(weight)*loss,[params[k] for k in GEOMETRY],allow_unused=True)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('Nonfinite semantic gradient')
    return dict(zip(GEOMETRY,grads)),float(loss.detach())


def fuse_native(core,params,data,frame,category,allow=None):
    """One native footprint distribution per source frame, independent of iterations."""
    n=len(params['means3D']);labels=torch.as_tensor(category,device=params['means3D'].device,dtype=torch.long)
    with torch.no_grad():
        ones=params['rgb_colors'].new_ones((n,3))
        alpha=render_channels(core,params,data,frame,ones,False)[0]
        dep=data['depth'].squeeze(0);valid=(dep>0)&torch.isfinite(dep)&(labels>=0)&(labels<19)&(alpha>=.05)
        if allow is not None:valid &= allow
    grads=[]
    # Only colors are differentiable for the visibility adjoint, no map update.
    detached={k:v.detach() for k,v in params.items()}
    transformed=core.transform_to_frame(detached,frame,gaussians_grad=False,camera_grad=False)
    for start in range(0,19,3):
        colors=torch.zeros((n,3),device=alpha.device,requires_grad=True)
        var=core.transformed_params2rendervar(detached,transformed);var['colors_precomp']=colors
        rendered=core.Renderer(raster_settings=data['cam'])(**var)[0]
        masks=torch.stack([(valid&(labels==k)).float() if k<19 else torch.zeros_like(alpha) for k in range(start,start+3)])
        grad=torch.autograd.grad(rendered,colors,grad_outputs=masks)[0]
        grads.append(grad.detach().cpu().numpy())
    counts=np.maximum(np.concatenate(grads,1)[:,:19],0);total=counts.sum(1);active=total>=.001
    observed=float((alpha*valid).sum(dtype=torch.float64))
    relative=abs(float(total.sum(dtype=np.float64))-observed)/max(observed,1.)
    if relative>1e-4:raise ValueError('Native semantic footprint conservation failed')
    distribution=counts/np.maximum(total[:,None],1e-8)
    return distribution,active.astype(np.float32),relative
