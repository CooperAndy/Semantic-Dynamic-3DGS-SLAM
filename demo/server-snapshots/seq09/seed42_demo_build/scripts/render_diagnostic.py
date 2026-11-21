from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer,GaussianRasterizationSettings as Camera
from utils.recon_helpers import setup_camera
from resource_guard import wait_for_headroom

R=Path(__file__).resolve().parent.parent

def main():
    out=R/'diagnostic';out.mkdir()
    results=[]
    for version,relative in [('F200','runs/F200_orb_raft_conf060/params.npz'),('F1','runs/F1_appearance800/params.npz')]:
        with np.load(R/relative,allow_pickle=False) as f:p={k:f[k] for k in f.files}
        means=torch.from_numpy(p['means3D']).cuda();colors=torch.from_numpy(p['rgb_colors']).cuda();rot=F.normalize(torch.from_numpy(p['unnorm_rotations']).cuda(),dim=1);scales=torch.exp(torch.from_numpy(p['log_scales']).cuda().repeat(1,3));op=torch.sigmoid(torch.from_numpy(p['logit_opacities']).cuda())
        poses=np.loadtxt(R/'poses_estimated_camera2.txt').reshape(200,3,4)
        for i in [0,49,99,149,199]:
            wait_for_headroom('diagnostic',i)
            c2w=np.eye(4);c2w[:3,:4]=poses[i]
            cam=setup_camera(1226,370,p['intrinsics'],np.linalg.inv(c2w).astype(np.float32),near=0.01,far=150)
            common={k:getattr(cam,k) for k in ['image_height','image_width','tanfovx','tanfovy','scale_modifier','viewmatrix','projmatrix','sh_degree','campos','prefiltered']}
            cam=Camera(bg=torch.tensor([.055,.065,.08],device='cuda'),**common)
            with torch.no_grad():rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=colors,rotations=rot,opacities=op,scales=scales)
            pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();target=np.asarray(Image.open(R/f'sequences/09/image_2/{i:06d}.png').convert('RGB'),dtype=np.float32)/255
            roi=np.load(R/f'reliable_roi/{i:06d}_valid.npy')
            errors=(pred-target)**2
            result={'variant':version,'frame':i,'full_rgb_psnr':float(-10*np.log10(errors.mean())),'reliable_roi_psnr':float(-10*np.log10(errors[roi].mean())),'scope':'seen training views; original-resolution offline renderer; no held-out or tracking claim'}
            Image.fromarray(np.round(pred*255).astype(np.uint8)).save(out/f'{version}_{i:06d}.png')
            if version=='F200':
                Image.fromarray(np.round(target*255).astype(np.uint8)).save(out/f'reference_{i:06d}.png')
            results.append(result);print(json.dumps(result),flush=True)
        del means,colors,rot,scales,op,rgb;torch.cuda.empty_cache()
    (out/'seen5.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
