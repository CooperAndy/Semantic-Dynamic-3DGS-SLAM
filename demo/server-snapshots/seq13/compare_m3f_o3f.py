"""Paired seen-view visual comparison of 260-frame O3F and M3F maps."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
P=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115');O=P/'cell_O3_retry1';M=P/'cell_M3';R=Path('/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1');OUT=P/'compare_m3f_o3f'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 sys.path.insert(0,str(O/'source'));sys.path.insert(0,str(R))
 import diff_gaussian_rasterization as raster
 assert Path(raster.__file__).resolve().is_relative_to(R.resolve())
 from diff_gaussian_rasterization import GaussianRasterizer as Renderer
 from utils.recon_helpers import setup_camera
 from utils.slam_external import calc_ssim
 runtime=json.loads((P/'bundle/runtime.json').read_text());assert runtime['frame_count']==260
 with np.load(O/'runs/B/estimated_trajectory.npz',allow_pickle=False) as z:ow=z['w2c'].copy();oc=z['c2w'].copy()
 with np.load(M/'runs/B/estimated_trajectory.npz',allow_pickle=False) as z:mw=z['w2c'].copy();mc=z['c2w'].copy()
 assert ow.shape==mw.shape==(260,4,4) and np.allclose(ow[0],mw[0],atol=1e-5)
 frames=[0,19,39,59,79,99,119,139,159,179,199,229,259];OUT.mkdir()
 cases=[('O3F_common',O/'runs/O3F/params.npz',ow),('M3F_common',M/'runs/M3F/params.npz',ow),('M3F_own',M/'runs/M3F/params.npz',mw)];results={}
 for name,path,poses in cases:
  directory=OUT/name;directory.mkdir()
  with np.load(path,allow_pickle=False) as z:p={k:z[k].copy() for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','intrinsics']}
  assert np.allclose(p['intrinsics'],runtime['K'],rtol=0,atol=1e-3)
  means=torch.from_numpy(p['means3D']).cuda();colors=torch.from_numpy(p['rgb_colors']).cuda();rot=F.normalize(torch.from_numpy(p['unnorm_rotations']).cuda(),dim=1);op=torch.sigmoid(torch.from_numpy(p['logit_opacities']).cuda());scale=torch.exp(torch.from_numpy(p['log_scales']).cuda());scale=scale.repeat(1,3) if scale.shape[1]==1 else scale;rows=[]
  for i in frames:
   row=runtime['rows'][i];target=np.asarray(Image.open(row['rgb']).convert('RGB'),dtype=np.float32)/255;depth=np.load(row['depth'],allow_pickle=False);mask=np.isfinite(depth)&(depth>0);assert mask.any()
   cam=setup_camera(runtime['W'],runtime['H'],p['intrinsics'],poses[i]);
   with torch.no_grad():rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=colors,rotations=rot,opacities=op,scales=scale)
   pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();err=(pred-target)**2
   with torch.no_grad():ssim=float(calc_ssim(torch.from_numpy(pred.transpose(2,0,1))[None],torch.from_numpy(target.transpose(2,0,1))[None]))
   rows.append(dict(frame=i,source_frame=row['source_frame_id'],psnr=float(-10*np.log10(max(float(err.mean()),1e-12))),roi_psnr=float(-10*np.log10(max(float(err[mask].mean()),1e-12))),ssim=ssim));Image.fromarray(np.round(pred*255).astype(np.uint8)).save(directory/f'{row["source_frame_id"]:06}.png')
  results[name]=dict(path=str(path),sha256=sha(path),gaussians=len(means),mean_psnr=float(np.mean([r['psnr'] for r in rows])),mean_roi_psnr=float(np.mean([r['roi_psnr'] for r in rows])),mean_ssim=float(np.mean([r['ssim'] for r in rows])),rows=rows);del means,colors,rot,op,scale,p;torch.cuda.empty_cache()
 delta=np.linalg.norm(oc[:,:3,3]-mc[:,:3,3],axis=1);angles=Rotation.from_matrix(np.einsum('nji,njk->nik',oc[:,:3,:3],mc[:,:3,:3])).magnitude()*180/np.pi
 events=[json.loads(x) for x in (M/'runs/B/backend_events.jsonl').read_text().splitlines()];accepted=[x['frame'] for x in events if x['accepted'] is True]
 result=dict(status='COMPLETE',scope='13 predetermined seen views; G19C depth-valid ROI; black background; no GT accuracy',source_frames=[780+i for i in frames],common_camera='O3 selected estimated ORB poses; first pose common, no post-hoc alignment',own_camera_note='M3F_own is not a paired-camera score',results=results,MASS=dict(accepted_count=len(accepted),attempts=len(events)-1,accepted_local_frames=accepted,translation_difference_mean_m=float(delta.mean()),translation_difference_max_m=float(delta.max()),rotation_difference_mean_deg=float(angles.mean()),rotation_difference_max_deg=float(angles.max())),limits='Additional MASS feedback changes later map and trajectory; no ground truth accuracy or isolated H3 claim')
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'COMPLETE','MASS_accepted':len(accepted),'means':{k:[v['mean_psnr'],v['mean_ssim'],v['mean_roi_psnr']] for k,v in results.items()}}),flush=True)
if __name__=='__main__':main()
