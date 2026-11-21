"""Seen-view visual evaluation of the one M3 map before and after F1."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
P=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116');M=P/'cell_m3only_full';O=M;R=Path('/root/autodl-tmp/s23_effects_20261105_p16247/raster_fp64_accum_v1');OUT=P/'m3only_evaluation'
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
 runtime=json.loads((P/'bundle/runtime.json').read_text());assert runtime['frame_count']==801
 with np.load(O/'runs/B/estimated_trajectory.npz',allow_pickle=False) as z:ow=z['w2c'].copy();oc=z['c2w'].copy()
 with np.load(M/'runs/B/estimated_trajectory.npz',allow_pickle=False) as z:mw=z['w2c'].copy();mc=z['c2w'].copy()
 assert ow.shape==mw.shape==(801,4,4) and np.allclose(ow[0],mw[0],atol=1e-5)
 relative=np.einsum('nji,njk->nik',oc[:-1,:3,:3],oc[1:,:3,:3]);yaw=np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(),axis=1);turn=int(np.argmax(yaw))+1;frames=sorted(set([0,80,160,240,320,400,480,560,640,720,800,turn]));OUT.mkdir()
 cases=[('M3',M/'runs/B/params.npz',mw),('M3F',M/'runs/M3F/params.npz',mw)];results={}
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
   with torch.no_grad(): alpha_rgb,*_=Renderer(raster_settings=cam)(means3D=means,means2D=torch.zeros_like(means),colors_precomp=torch.ones_like(colors),rotations=rot,opacities=op,scales=scale)
   alpha=alpha_rgb[0].cpu().numpy();coverage=float((alpha>=.5).mean());roi_coverage=float((alpha[mask]>=.5).mean());del alpha_rgb
   with torch.no_grad():ssim=float(calc_ssim(torch.from_numpy(pred.transpose(2,0,1))[None],torch.from_numpy(target.transpose(2,0,1))[None]))
   rows.append(dict(frame=i,source_frame=row['source_frame_id'],psnr=float(-10*np.log10(max(float(err.mean()),1e-12))),roi_psnr=float(-10*np.log10(max(float(err[mask].mean()),1e-12))),ssim=ssim,alpha_coverage_0_5=coverage,roi_alpha_coverage_0_5=roi_coverage));Image.fromarray(np.round(pred*255).astype(np.uint8)).save(directory/f'{row["source_frame_id"]:06}.png')
  results[name]=dict(path=str(path),sha256=sha(path),gaussians=len(means),mean_psnr=float(np.mean([r['psnr'] for r in rows])),mean_roi_psnr=float(np.mean([r['roi_psnr'] for r in rows])),mean_ssim=float(np.mean([r['ssim'] for r in rows])),mean_alpha_coverage_0_5=float(np.mean([r['alpha_coverage_0_5'] for r in rows])),mean_roi_alpha_coverage_0_5=float(np.mean([r['roi_alpha_coverage_0_5'] for r in rows])),rows=rows);del means,colors,rot,op,scale,p;torch.cuda.empty_cache()
 delta=np.linalg.norm(oc[:,:3,3]-mc[:,:3,3],axis=1);angles=Rotation.from_matrix(np.einsum('nji,njk->nik',oc[:,:3,:3],mc[:,:3,:3])).magnitude()*180/np.pi
 events=[json.loads(x) for x in (M/'runs/B/backend_events.jsonl').read_text().splitlines()];accepted=[x['frame'] for x in events if x['accepted'] is True]
 for v in results.values():
  v['segments']={}
  for label,lo,hi in [('early',0,266),('middle',267,533),('late',534,800)]:
   subset=[x for x in v['rows'] if lo<=x['source_frame']<=hi];assert subset;v['segments'][label]=dict(frames=[x['source_frame'] for x in subset],mean_psnr=float(np.mean([x['psnr'] for x in subset])),mean_ssim=float(np.mean([x['ssim'] for x in subset])),mean_alpha_coverage_0_5=float(np.mean([x['alpha_coverage_0_5'] for x in subset])))
  v['worst_psnr_frame']=min(v['rows'],key=lambda x:x['psnr'])['source_frame']
 result=dict(status='COMPLETE',scope='preselected every 80 frames plus largest MASS selected relative rotation; 801 seen frames; Demo .60 depth-valid ROI; black background; no GT accuracy',source_frames=frames,common_camera='MASS selected estimated poses, identical for M3 and M3F',own_camera_note='M3 and M3F share the M3 selected poses',results=results,MASS=dict(accepted_count=len(accepted),attempts=len(events)-1,accepted_frames=accepted),limits='Single M3F group; no ground truth accuracy, no isolated H3 or MASS comparison')
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'COMPLETE','MASS_accepted':len(accepted),'means':{k:[v['mean_psnr'],v['mean_ssim'],v['mean_roi_psnr']] for k,v in results.items()}}),flush=True)
if __name__=='__main__':main()
