"""Compare seq10 M3/F1 with the prior ORB-F1 map on identical 20 views."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer,GaussianRasterizationSettings as Camera
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim
R=Path('/root/autodl-tmp/seq10_m3f_20261116');O=Path('/root/autodl-tmp/seq10_o3f_1000_20261116/comparison20');O.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
baseline=R/'baseline/orb_f1_resume796.npz';assert sha(baseline)=='01b9873a1c96c555e830b2ce49288cc55edac6ed72fecca9d64646aad366cccc'
runtime=json.loads((R/'bundle/runtime.json').read_text());assert runtime['frame_count']==1000
pose=np.loadtxt(runtime['pose_file']).reshape(1000,3,4); c2w=np.tile(np.eye(4),(1000,1,1));c2w[:,:3,:4]=pose;orb=np.linalg.inv(c2w)
with np.load(R/'cell_full/runs/B/estimated_trajectory.npz') as z: selected=z['w2c'].copy()
assert selected.shape==(1000,4,4) and np.allclose(selected[0],orb[0],atol=1e-5)
frames=list(range(19,200,20))+list(range(819,1000,20)); cases=[('o3_common',Path('/root/autodl-tmp/seq10_o3f_1000_20261116/cell_full/runs/B/params.npz'),orb),('o3f_common',Path('/root/autodl-tmp/seq10_o3f_1000_20261116/cell_full/runs/O3F/params.npz'),orb)];results={}
for name,path,poses in cases:
 (O/name).mkdir()
 with np.load(path,allow_pickle=False) as z:p={k:z[k].copy() for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','intrinsics']}
 assert np.allclose(p['intrinsics'][:3,:3],runtime['K'],atol=1e-3)
 means=torch.from_numpy(p['means3D']).cuda();colors=torch.from_numpy(p['rgb_colors']).cuda();rot=F.normalize(torch.from_numpy(p['unnorm_rotations']).cuda(),dim=1);scales=torch.exp(torch.from_numpy(p['log_scales']).cuda());scales=scales.repeat(1,3) if scales.shape[1]==1 else scales;op=torch.sigmoid(torch.from_numpy(p['logit_opacities']).cuda());rows=[]
 for i in frames:
  row=runtime['rows'][i];cam=setup_camera(runtime['W'],runtime['H'],p['intrinsics'],poses[i].astype(np.float32),near=.01,far=150);keys=['image_height','image_width','tanfovx','tanfovy','scale_modifier','viewmatrix','projmatrix','sh_degree','campos','prefiltered'];cam=Camera(bg=torch.tensor([.055,.065,.08],device='cuda'),**{k:getattr(cam,k) for k in keys})
  with torch.no_grad():rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=colors,rotations=rot,opacities=op,scales=scales)
  pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();target=np.asarray(Image.open(row['rgb']).convert('RGB'),dtype=np.float32)/255;mask=np.load(row['depth'])>0;err=(pred-target)**2;assert mask.any()
  with torch.no_grad():ssim=float(calc_ssim(torch.from_numpy(pred.transpose(2,0,1))[None],torch.from_numpy(target.transpose(2,0,1))[None]))
  rows.append(dict(frame=i,psnr=float(-10*np.log10(max(float(err.mean()),1e-12))),roi_psnr=float(-10*np.log10(max(float(err[mask].mean()),1e-12))),ssim=ssim));Image.fromarray(np.round(pred*255).astype(np.uint8)).save(O/name/f'{i:06}.png')
 results[name]=dict(path=str(path),sha256=sha(path),gaussians=len(means),**{f'mean_{k}':float(np.mean([x[k] for x in rows])) for k in ('psnr','roi_psnr','ssim')},rows=rows);del means,colors,rot,op,scales,p;torch.cuda.empty_cache()
(O/'results.json').write_text(json.dumps(dict(status='COMPLETE',scope='20 seen views; common ORB or own selected cameras; same current confidence ROI and background',frames=frames,results=results),indent=2));print('COMPLETE',flush=True)
