"""Native channel renderer, using the shared SplaTAM rasterizer.

Shared native alpha/depth order, explicit unknown channel, zero background.
Uses SplaTAM's rasterizer (three feature channels per pass), no palette decode.
CPU contracts cannot certify GPU numerical behavior; see dated native audit.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,native_render_variables,UNKNOWN,NO_MAP

def semantic_features(state):
    if not np.isfinite(state.evidence).all() or (state.evidence<0).any():
        raise ValueError('invalid semantic evidence')
    total=state.evidence.sum(1);known=total>0
    features=np.zeros((len(state.ids),state.evidence.shape[1]+1),np.float32)
    features[known,:-1]=state.evidence[known]/total[known,None]
    features[~known,-1]=1
    return features

def decode_mass(mass,alpha_min=.05,unknown_fraction_max=.5,reference_alpha=None):
    """Unknown contributes opacity; low support is distinct from unknown."""
    if mass.ndim!=3 or mass.shape[0]<2 or not np.isfinite(mass).all() or (mass<0).any():
        raise ValueError('invalid rendered class mass')
    alpha=mass.sum(0);unknown_mass=mass[-1];classes=mass[:-1]
    if reference_alpha is not None:
        if reference_alpha.shape!=alpha.shape or not np.isfinite(reference_alpha).all() or (reference_alpha<0).any():
            raise ValueError('invalid shared geometry alpha')
        if not np.allclose(alpha,reference_alpha,atol=5e-5,rtol=1e-4):
            raise ValueError('semantic channel alpha differs from shared geometry')
        alpha=reference_alpha
    prediction=np.full(alpha.shape,NO_MAP,np.int16)
    supported=alpha>=alpha_min;prediction[supported]=UNKNOWN
    ratio=np.divide(unknown_mass,alpha,out=np.ones_like(alpha),where=alpha>0)
    labelled=supported&(classes.sum(0)>0)&(ratio<unknown_fraction_max)
    prediction[labelled]=classes.argmax(0)[labelled]
    score=np.divide(classes.max(0),alpha,out=np.zeros_like(alpha),where=alpha>0)
    return dict(label=prediction,score=score,alpha=alpha,unknown_mass=unknown_mass,class_mass=classes)

def render_mass(params,transformed,features,camera):
    import torch
    import torch.nn.functional as F
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    if not bool((camera.bg==0).all()):raise ValueError('numeric semantic channels require zero background')
    padded=np.pad(features,((0,0),(0,(-features.shape[1])%3)))
    gpu_features=torch.as_tensor(padded,device=params['means3D'].device)
    chunks=[]
    for start in range(0,padded.shape[1],3):
        var=native_render_variables(params,transformed,gpu_features[:,start:start+3],torch.zeros_like,
            lambda q:F.normalize(q,dim=1),torch.exp,torch.sigmoid)
        rendered,_,_=Renderer(raster_settings=camera)(**var)
        chunks.append(rendered.cpu().numpy())
    return np.concatenate(chunks,axis=0)[:features.shape[1]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--params',required=True);p.add_argument('--state',required=True)
    p.add_argument('--splatam-root',required=True);p.add_argument('--frames',required=True)
    p.add_argument('--out',required=True);p.add_argument('--alpha-min',type=float,default=.05)
    p.add_argument('--unknown-fraction-max',type=float,default=.5);a=p.parse_args()
    # Deliberately lazy: importing this file for static/CPU audit does not touch CUDA.
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,a.splatam_root)
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from utils.recon_helpers import setup_camera
    from utils.slam_helpers import transform_to_frame
    out=Path(a.out)
    if out.exists():raise FileExistsError('use a new output run')
    out.mkdir(parents=True)
    with np.load(a.params,allow_pickle=False) as raw:
        ids=raw['source_gaussian_index'];state=Evidence.load(a.state,ids)
        params={k:torch.as_tensor(raw[k],device='cuda',dtype=torch.float32) for k in
           ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
        camera=setup_camera(int(raw['org_width']),int(raw['org_height']),raw['intrinsics'],raw['w2c'])
        if not bool((camera.bg==0).all()):raise ValueError('numeric semantic channels require zero background')
        timestep=raw['timestep'].copy()
    features=semantic_features(state)
    with torch.no_grad():
        for frame in [int(v) for v in a.frames.split(',')]:
            if not 0<=frame<params['cam_trans'].shape[-1]:raise ValueError('frame out of range')
            transformed=transform_to_frame(params,frame,gaussians_grad=False,camera_grad=False)
            mass=render_mass(params,transformed,features,camera)
            shared_alpha=render_mass(params,transformed,np.ones((len(ids),3),np.float32),camera)[0]
            np.savez_compressed(out/f'{frame:06d}_semantic.npz',**decode_mass(mass,a.alpha_min,a.unknown_fraction_max,reference_alpha=shared_alpha))
    (out/'render_manifest.json').write_text(json.dumps({'frames':a.frames,'channel_count':features.shape[1],
       'unknown_preserves_opacity':True,'zero_background_required':True,'alpha_min':a.alpha_min,
       'unknown_fraction_max':a.unknown_fraction_max,'geometry_mode':'fixed final map; all base Gaussians active',
       'support_alpha':'shared independent native all-one feature render',
       'timestep_note':'stored for provenance; final-map rendering includes all Gaussians, not incremental-time reconstruction',
       'parameters':a.params,'state':a.state},indent=2))

if __name__=='__main__':main()
