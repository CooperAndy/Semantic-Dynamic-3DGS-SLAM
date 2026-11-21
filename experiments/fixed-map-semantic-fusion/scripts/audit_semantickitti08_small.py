"""Counts, RGB hash chain, official Tr/P2 and independent projection check."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np,cv2,yaml
from PIL import Image
from semantickitti_projection import calibration,project,PALETTE,CITY_MAP
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1';B=R/'00_输入与基图_GT曝光审计/semantickitti08_small_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 source=json.loads((D/'small_download_manifest.json').read_text());P,T=calibration(D/'calib.txt');official=yaml.safe_load((B/'official_semantic_kitti.yaml').read_text());inventory=list(csv.DictReader((R/'inputs/input_inventory_20261007.csv').open()));K=P[:,:3];t2=np.linalg.inv(K)@P[:,3]
 with np.load(R/'inputs/base_map/params.npz') as b:assert np.allclose(b['intrinsics'],K,atol=.0001);h,w=int(b['org_height']),int(b['org_width'])
 existing=R.parent/'slam_project/data/kitti/dataset/sequences/08/calib.txt';partial={}
 for l in existing.read_text().splitlines():
  k,v=l.split(':',1);partial[k]=np.array([float(x) for x in v.split()])
 assert np.allclose(partial['P2'],P.ravel(),atol=1e-10);shutil.copyfile(D/'calib.txt',B/'official_calib_with_Tr.txt');rows=[]
 for f in range(3000,3010):
  p=D/'velodyne'/f'{f:06d}.bin';lp=D/'labels'/f'{f:06d}.label';assert p.stat().st_size%16==0 and lp.stat().st_size%4==0 and p.stat().st_size//16==lp.stat().st_size//4
  points=np.fromfile(p,dtype='<f4').reshape(-1,4);packed=np.fromfile(lp,dtype='<u4');assert np.isfinite(points).all();sem=packed&0xffff;assert set(np.unique(sem).tolist())<=set(official['labels']);assert points[:,3].min()>=0 and points[:,3].max()<=1
  rec=next(x for x in inventory if x['kind']=='image_2' and int(x['original_frame'])==f);rgbpath=Path(rec['stage3_reference']);assert Path(rec['source']).stem==f'{f:06d}' and sha(rgbpath)==rec['sha256'];rgb=np.asarray(Image.open(rgbpath).convert('RGB'));assert rgb.shape==(h,w,3)
  gt,stats,uv=project(points,packed,P,T,h,w);assert stats['in_image_points']>1000 and stats['mapped_GT_pixels']>500
  # Independent OpenCV projection uses camera2 offset K^-1 P2[:,3].
  index=np.flatnonzero(np.isfinite(uv).all(1))[::10];reference=cv2.projectPoints(points[index,:3].astype(np.float64),cv2.Rodrigues(T[:3,:3])[0],T[:3,3]+t2,K,np.zeros(5))[0].reshape(-1,2);positive=((np.c_[points[index,:3],np.ones(len(index))]@T.T)[:,2]+t2[2])>1;error=float(np.max(np.abs(reference[positive]-uv[index][positive])));assert error<.02
  mask=gt['label']!=255;assert np.array_equal(gt['raw_semantic'][mask],sem[gt['point_index'][mask]])
  overlay=rgb.copy();y,x=np.nonzero(mask)
  for yy,xx in zip(y.tolist(),x.tolist()):cv2.circle(overlay,(xx,yy),1,tuple(int(v) for v in PALETTE[gt['label'][yy,xx]]),-1)
  canvas=np.vstack([rgb,overlay]);cv2.putText(canvas,f'KITTI08 original={f:06d}; LiDAR semantic circles r=1 display only; GT pixels not densified',(15,h+25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);Image.fromarray(canvas).save(B/f'{f:06d}_RGB_semantic_projection.png')
  depth=gt['depth'];image=cv2.applyColorMap(np.clip(depth/60*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);image[depth==0]=0;Image.fromarray(image[:,:,::-1]).save(B/f'{f:06d}_projected_depth.png');np.savez_compressed(B/f'{f:06d}_sparse_GT.npz',**gt)
  rows.append({'original_frame':f,'local_frame':f-3000,'sequence':'08','RGB_source':rec['source'],'RGB_sha256':sha(rgbpath),'point_sha256':sha(p),'label_sha256':sha(lp),'point_count_equals_label_count':len(points),'packing':'little-endian float32xyzi/uint32 low16semantic high16instance','projection_opencv_max_abs_px_error':error,'mapped_pixel_fraction':stats['mapped_GT_pixels']/(h*w),**stats})
 dump(B/'correspondence_audit.json',{'status':'AUTOMATED_CORRESPONDENCE_PASS_AWAIT_VISUAL_REVIEW','frames':rows,'official_calibration_sha256':sha(D/'calib.txt'),'official_classes_sha256':sha(B/'official_semantic_kitti.yaml'),'existing_P2_matches_official':True,'base_K_matches_official':True,'projection':'P2 @ Tr @ homogeneous LiDAR point; original rectified image_2; full P2 translation retained; no extra R0 rectification','P2_camera_offset_m':t2.tolist(),'nearest_all_label_zbuffer_before_GT_ignore':True,'moving_semantic_IDs_preserved':True,'main_mask':'sparse independent mapped GT pixels; no dilation or prediction masks','exposure':'HISTORICALLY_EXPOSED_DEV, not Test','visual_review_passed':False,'no_quality_metrics_or_selection_yet':True,'mapping':CITY_MAP})
 print(json.dumps({'status':'AUTOMATED_CORRESPONDENCE_PASS','frames':10,'image_size':[h,w],'visual_review_required':True}),flush=True)
if __name__=='__main__':main()
