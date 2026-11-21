from pathlib import Path
import argparse,json,numpy as np,torch,torch.nn.functional as F
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer,GaussianRasterizationSettings as Camera
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim

a=argparse.ArgumentParser(); a.add_argument('--name',required=True); a.add_argument('--map',type=Path,required=True); a.add_argument('--out',type=Path,required=True); args=a.parse_args()
D=Path('/root/autodl-tmp/seq13_000780_000979_tuning_20261115'); args.out.mkdir(exist_ok=True); images=args.out/args.name; images.mkdir(exist_ok=False)
with np.load(args.map,allow_pickle=False) as z:p={k:z[k] for k in z.files}
means=torch.from_numpy(p['means3D']).cuda(); colors=torch.from_numpy(p['rgb_colors']).cuda(); rot=F.normalize(torch.from_numpy(p['unnorm_rotations']).cuda(),dim=1); scales=torch.exp(torch.from_numpy(p['log_scales']).cuda()); scales=scales.repeat(1,3) if scales.shape[1]==1 else scales; op=torch.sigmoid(torch.from_numpy(p['logit_opacities']).cuda())
poses=np.load('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115/cell_O3_retry1/runs/B/estimated_trajectory.npz')['c2w'][:200,:3,:4]; frames=sorted(set([0,49,99,149,199]+list(range(19,200,20)))); rows=[]
for i in frames:
 c2w=np.eye(4); c2w[:3]=poses[i]; cam=setup_camera(1241,376,p['intrinsics'],np.linalg.inv(c2w).astype(np.float32),near=.01,far=150)
 keys=['image_height','image_width','tanfovx','tanfovy','scale_modifier','viewmatrix','projmatrix','sh_degree','campos','prefiltered']; cam=Camera(bg=torch.tensor([.055,.065,.08],device='cuda'),**{k:getattr(cam,k) for k in keys})
 with torch.no_grad(): rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=colors,rotations=rot,opacities=op,scales=scales)
 pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy(); target=np.asarray(Image.open(D/f'sequences/13/image_2/{i:06d}.png').convert('RGB'),dtype=np.float32)/255; mask=np.load(D/f'reliable_roi/{i:06d}_valid.npy').astype(bool); err=(pred-target)**2
 with torch.no_grad(): ssim=float(calc_ssim(torch.from_numpy(pred.transpose(2,0,1))[None],torch.from_numpy(target.transpose(2,0,1))[None]))
 rows.append(dict(frame=i,source_frame=780+i,psnr=float(-10*np.log10(max(err.mean(),1e-12))),roi_psnr=float(-10*np.log10(max(err[mask].mean(),1e-12))),ssim=ssim)); Image.fromarray(np.round(pred*255).astype(np.uint8)).save(images/f'{i:06}.png')
res=dict(name=args.name,map=str(args.map),gaussians=len(p['means3D']),mean_psnr=float(np.mean([x['psnr'] for x in rows])),mean_roi_psnr=float(np.mean([x['roi_psnr'] for x in rows])),mean_ssim=float(np.mean([x['ssim'] for x in rows])),rows=rows)
(args.out/f'{args.name}.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res),flush=True)
