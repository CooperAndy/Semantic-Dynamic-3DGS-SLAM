"""GS-SLAM Eq.(12)-inspired selection, using measured same-frame pseudo-depth.

This module changes only tracking render input. It never deletes map points.
"""
import json
from pathlib import Path
import torch


class InsufficientTrackingSupport(RuntimeError):
    pass


@torch.no_grad()
def select_gaussians(xyz, c2w, depth, intrinsics, abs_tol=0.5, rel_tol=0.02):
    """Positive camera-z convention; nearest pixel centers; no GT or renderer depth.

    KITTI adaptation tolerance = 0.5 m + 2% of measured depth, frozen before
    this experiment. Not a threshold claimed by the original paper.
    """
    assert depth.ndim == 2 and xyz.ndim == 2 and xyz.shape[1] == 3
    assert c2w.shape == (4,4) and abs_tol >= 0 and rel_tol >= 0
    H,W=depth.shape
    fx,fy,cx,cy=intrinsics
    assert fx>0 and fy>0
    camera=(xyz-c2w[:3,3])@c2w[:3,:3]
    z=camera[:,2]
    valid=torch.isfinite(camera).all(-1)&(z>0)
    safe_z=torch.where(valid,z,torch.ones_like(z))
    u=fx*camera[:,0]/safe_z+cx;v=fy*camera[:,1]/safe_z+cy
    visible=valid&torch.isfinite(u)&torch.isfinite(v)&(u>=-0.5)&(u<W-0.5)&(v>=-0.5)&(v<H-0.5)
    # Clamp only for safe indexing. Out-of-image centers remain rejected.
    ui=torch.floor(torch.nan_to_num(u,nan=0,posinf=0,neginf=0)+0.5).clamp(0,W-1).long()
    vi=torch.floor(torch.nan_to_num(v,nan=0,posinf=0,neginf=0)+0.5).clamp(0,H-1).long()
    observed=depth[vi,ui]
    observed_valid=visible&torch.isfinite(observed)&(observed>=0.5)&(observed<40.0)
    residual=torch.abs(z-observed)
    mask=observed_valid&(residual<=abs_tol+rel_tol*observed)
    stats={'total':len(xyz),'visible_centers':int(visible.sum().item()),
           'valid_observation':int(observed_valid.sum().item()),'selected':int(mask.sum().item()),
           'abs_tol_m':abs_tol,'relative_tol':rel_tol,'depth_range_m':[0.5,40.0],
           'selection_fraction':float(mask.float().mean().item()) if len(xyz) else 0.0}
    return mask,stats


def install_selector(mapper, module, observations, output, min_gaussians=256, min_pixels=128):
    """Install hooks once; support guard is inserted in the isolated mapper source."""
    output=Path(output)
    base_tracking=mapper.tracking
    base_optimize=mapper.optimize_cam_in_batch
    base_render=module.render
    mapper.cfg['t04_depth_selection']=True
    mapper.cfg['t04_min_tracking_pixels']=min_pixels
    mapper._t04_render_selection_active=False
    mapper._t04_selection_mask=None
    mapper._t04_support=[]

    def selected_render(*args,**kwargs):
        if mapper._t04_render_selection_active:
            mask=mapper._t04_selection_mask
            assert mask is not None and int(mask.sum().item())>=min_gaussians
            assert all(x.shape[0]==len(mask) for x in args[:6])
            args=tuple(x[mask] for x in args[:6])+args[6:]
        return base_render(*args,**kwargs)

    def guarded_optimize(*args,**kwargs):
        mapper._t04_render_selection_active=True
        try:
            result=base_optimize(*args,**kwargs)
            mapper._t04_support.append(int(result[-1].sum().item()))
            return result
        finally:
            mapper._t04_render_selection_active=False

    def retain_coarse(idx,pose):
        mapper.estimate_c2w_list[int(idx)]=pose.detach().clone().cpu()
        mapper.pre_c2w=pose.detach().clone()
        mapper.idx[0]=idx

    def selected_tracking(idx,color,depth,coarse,seg_mask):
        key=int(idx);source_id=int(mapper._t04_source_id)
        stats={'keyframe':key,'source_frame':source_id,'fallback':False,'iterations_completed':0}
        mapper._t04_support=[]
        try:
            if key<=1:
                stats['reason']='author_first_two_keyframes_no_refinement'
                return base_tracking(idx,color,depth,coarse,seg_mask)
            observed=observations[source_id].to(device=coarse.device,dtype=coarse.dtype)
            assert observed.shape==(mapper.H,mapper.W)
            xyz=mapper.gaussians.get_xyz().detach()
            mask,selection=select_gaussians(xyz,coarse,observed,(mapper.fx,mapper.fy,mapper.cx,mapper.cy))
            mapper._t04_selection_mask=mask
            stats.update(selection)
            if selection['selected']<min_gaussians:
                stats.update(fallback=True,reason='too_few_consistent_gaussians')
                retain_coarse(idx,coarse)
                return None
            try:
                return base_tracking(idx,color,depth,coarse,seg_mask)
            except InsufficientTrackingSupport as error:
                stats.update(fallback=True,reason=str(error))
                retain_coarse(idx,coarse)
                return None
        finally:
            support=mapper._t04_support
            stats['iterations_completed']=len(support)
            if support:stats.update(pixel_support_min=min(support),pixel_support_max=max(support),pixel_support_fraction_min=min(support)/(mapper.H*mapper.W))
            with (output/'selection_history.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            print('T04_DEPTH_SELECTION',json.dumps(stats),flush=True)
            mapper._t04_render_selection_active=False
            mapper._t04_selection_mask=None

    mapper.tracking=selected_tracking
    mapper.optimize_cam_in_batch=guarded_optimize
    module.render=selected_render
