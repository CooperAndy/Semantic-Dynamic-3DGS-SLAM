"""C04 candidate selection and common per-frame sampling; native initialization retained."""
import json,hashlib,numpy as np,torch
from pathlib import Path
import torch.nn.functional as F

def add_new_gaussians(params,variables,data,sil_thres,index,mean_sq_dist_method,gaussian_distribution,
                      confidence_seed_threshold=None,*,core,mode,cap,seed,log_path,fb):
    assert mode in ['native','disp_valid'] and sil_thres==.5 and cap in (3000, 6000)
    g=core.transform_to_frame(params,index,gaussians_grad=False,camera_grad=False)
    ds,_,_=core.Renderer(raster_settings=data['cam'])(**core.transformed_params2depthplussilhouette(params,data['w2c'],g))
    observed=data['depth'][0];rendered=ds[0];sil=ds[1]
    valid=observed>0
    error=torch.abs(observed-rendered)*valid
    median=error.median()
    sil_candidate=sil<sil_thres
    disparity_error=fb*torch.abs(1/observed.clamp_min(1e-6)-1/rendered.clamp_min(1e-6))*valid
    used_median=median if mode=='native' else (disparity_error[valid].median() if valid.any() else disparity_error.new_tensor(0.))
    if mode=='native':raw=sil_candidate|((rendered>observed)&(error>50*median))
    else:raw=sil_candidate|(disparity_error>1.)|((rendered>observed)&(disparity_error>10*used_median))
    native_raw_any=bool(raw.any())
    if confidence_seed_threshold is not None:
        confidence=data.get('depth_confidence');assert confidence is not None
        valid=valid&(confidence[0]>=confidence_seed_threshold)
    candidate=raw&valid
    selected=candidate.reshape(-1).clone()
    eligible=np.flatnonzero(selected.detach().cpu().numpy())
    if len(eligible)>cap:
        # Common pixel ranking, independent of candidate count and global SLAM RNG.
        ranks=np.random.default_rng(np.random.SeedSequence([seed,index])).random(selected.numel(),dtype=np.float32)
        chosen=eligible[np.argpartition(ranks[eligible],cap-1)[:cap]]
        selected.zero_();selected[torch.tensor(np.sort(chosen),device=selected.device)]=True
    chosen_count=int(selected.sum())
    assert chosen_count<=cap
    depth_bands=[]
    for low,high in [(.5,10.),(10.,30.),(30.,80.0001)]:
        band=valid&(observed>=low)&(observed<high)
        depth_bands.append(dict(low_m=low,high_m=min(high,80.),valid=int(band.sum()),
                                candidates=int((candidate&band).sum()),selected=int((selected.reshape_as(band)&band).sum())))
    event=dict(append_quota=cap,frame=index,mode=mode,fb=fb,used_residual_median=float(used_median),disparity1_candidates=int(((disparity_error>1)&valid).sum()),valid_depth=int(valid.sum()),candidate_count=len(eligible),selected_count=chosen_count,
               full_image_depth_error_median=float(median),valid_only_depth_error_median=float(error[valid].median()) if valid.any() else None,
               sil_candidates=int((sil_candidate&valid).sum()),absolute05_candidates=int(((error>.05)&valid).sum()),depth_bands=depth_bands)
    mask_dir=Path(log_path).parent/'addition_masks';mask_dir.mkdir(exist_ok=True)
    mask_file=mask_dir/f'{index:06d}.npz'
    extra={}
    if index in [1,24,98,178]:
        extra=dict(observed=observed.detach().cpu().numpy(),rendered=rendered.detach().cpu().numpy(),silhouette=sil.detach().cpu().numpy(),valid=valid.detach().cpu().numpy())
    np.savez_compressed(mask_file,candidate_bits=np.packbits(candidate.detach().cpu().numpy().reshape(-1)),selected_ids=np.flatnonzero(selected.detach().cpu().numpy()).astype(np.int32),shape=np.array(candidate.shape),**extra)
    event['mask_file']=str(mask_file.relative_to(Path(log_path).parent))
    event['mask_sha256']=hashlib.sha256(mask_file.read_bytes()).hexdigest()
    with open(log_path,'a') as stream:stream.write(json.dumps(event)+'\n')
    # Preserve the native tail, including its empty-selection buffer reset behavior.
    if native_raw_any:
        q=F.normalize(params['cam_unnorm_rots'][...,index].detach())
        t=params['cam_trans'][...,index].detach()
        w2c=torch.eye(4,device='cuda');w2c[:3,:3]=core.build_rotation(q);w2c[:3,3]=t
        pointcloud,dist=core.get_pointcloud(data['im'],data['depth'],data['intrinsics'],w2c,
                                          mask=selected,compute_mean_sq_dist=True,mean_sq_dist_method=mean_sq_dist_method)
        new=core.initialize_new_params(pointcloud,dist,gaussian_distribution)
        for key,value in new.items():params[key]=torch.nn.Parameter(torch.cat((params[key],value),dim=0).requires_grad_(True))
        n=params['means3D'].shape[0]
        variables['means2D_gradient_accum']=torch.zeros(n,device='cuda').float()
        variables['denom']=torch.zeros(n,device='cuda').float()
        variables['max_2D_radius']=torch.zeros(n,device='cuda').float()
        variables['timestep']=torch.cat((variables['timestep'],index*torch.ones(pointcloud.shape[0],device='cuda').float()),dim=0)
    return params,variables
