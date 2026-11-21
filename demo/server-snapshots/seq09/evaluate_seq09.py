"""Common ORB-camera RGB comparison; S23 holdouts are seen by the old demo."""
from pathlib import Path
import json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer, GaussianRasterizationSettings as Camera
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim
R=Path('/root/autodl-tmp/seq09_300_499_demo_tuning_20261114'); D=Path('/root/autodl-tmp/demo_batch09_17_20261107/seq09_000300_000499')
out=R/'comparison_common_orb'; out.mkdir(exist_ok=False)
poses=np.loadtxt(D/'poses_estimated_camera2.txt').reshape(200,3,4)
frames=[0,49,99,149,199]+list(range(19,200,20))
results=[]
for label,path in [('Demo_F1',D/'runs/F1_appearance800/params.npz'),('B',R/'S0/runs/B/params.npz'),('TMD',R/'S0/runs/TMD/params.npz')]:
    with np.load(path,allow_pickle=False) as z: p={k:z[k] for k in z.files}
    means=torch.from_numpy(p['means3D']).cuda(); colors=torch.from_numpy(p['rgb_colors']).cuda(); rot=F.normalize(torch.from_numpy(p['unnorm_rotations']).cuda(),dim=1)
    scales=torch.exp(torch.from_numpy(p['log_scales']).cuda()); scales=scales.repeat(1,3) if scales.shape[1]==1 else scales
    op=torch.sigmoid(torch.from_numpy(p['logit_opacities']).cuda())
    for i in frames:
        c2w=np.eye(4); c2w[:3]=poses[i]
        cam=setup_camera(1226,370,p['intrinsics'],np.linalg.inv(c2w).astype(np.float32),near=.01,far=150)
        keys=['image_height','image_width','tanfovx','tanfovy','scale_modifier','viewmatrix','projmatrix','sh_degree','campos','prefiltered']
        cam=Camera(bg=torch.tensor([.055,.065,.08],device='cuda'),**{k:getattr(cam,k) for k in keys})
        with torch.no_grad(): rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=colors,rotations=rot,opacities=op,scales=scales)
        pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy(); target=np.asarray(Image.open(D/f'sequences/09/image_2/{i:06d}.png').convert('RGB'),dtype=np.float32)/255
        mse=float(np.mean((pred-target)**2)); score=float(-10*np.log10(max(mse,1e-12)))
        with torch.no_grad(): ssim=float(calc_ssim(torch.from_numpy(pred.transpose(2,0,1))[None],torch.from_numpy(target.transpose(2,0,1))[None]))
        row=dict(variant=label,local_frame=i,source_frame=300+i,cohort='S23_holdout' if i%20==19 else 'seen',psnr=score,ssim=ssim)
        results.append(row); print(json.dumps(row),flush=True)
        Image.fromarray(np.round(pred*255).astype(np.uint8)).save(out/f'{label}_{i:06}.png')
    del means,colors,rot,scales,op,rgb; torch.cuda.empty_cache()
(out/'results.json').write_text(json.dumps(dict(camera='common original ORB P2',background=[.055,.065,.08],scope='S23 holdout frames were seen by existing F1; not an equal held-out comparison',rows=results),indent=2))
