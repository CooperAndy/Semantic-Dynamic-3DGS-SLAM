"""CHY exposed DEV adapter: official semantic GT never used by prediction path."""
import argparse,hashlib,io,json,tarfile,runpy
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--mapping-template',required=True);a=ap.parse_args()
 root=Path(a.root);out=root/'inputs/vkitti2_gt_dev_v2';archives=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/archives')
 gtarchive=out/'vkitti_2.0.3_classSegmentation.tar'
 md5=hashlib.md5(gtarchive.read_bytes()).hexdigest();assert md5=='e8658ab49250e61f1caa174625563263'
 if (out/'manifest.json').exists():raise FileExistsError('preserve prior adapter')
 handles={k:tarfile.open(p) for k,p in {'rgb':archives/'vkitti_2.0.3_rgb.tar','depth':archives/'vkitti_2.0.3_depth.tar','instance':archives/'vkitti_2.0.3_instanceSegmentation.tar','text':archives/'vkitti_2.0.3_textgt.tar.gz','class':gtarchive}.items()}
 def raw(k,n):return handles[k].extractfile(n).read()
 def img(k,n):return Image.open(io.BytesIO(raw(k,n)))
 mapping={'Terrain':9,'Sky':10,'Tree':8,'Vegetation':8,'Building':2,'Road':0,'TrafficSign':7,'TrafficLight':6,'Pole':5,'Truck':14,'Car':13,'Van':13,'GuardRail':255,'Misc':255,'Undefined':255}
 sequences=[]
 for scene,start in [('Scene01',330),('Scene02',110)]:
  name=f'{scene}_clone_{start}_{start+25}';d=out/name;d.mkdir()
  for n in ['image_2','image_3','depth_sgbm','gt_depth','gt_labels','gt_instances','gt_moving']: (d/n).mkdir()
  prefix=scene+'/clone/'
  text={n:raw('text',prefix+n).decode() for n in ['colors.txt','intrinsic.txt','extrinsic.txt','bbox.txt']}
  for n,t in text.items():(d/n).write_text(t)
  rows=lambda n:[l.split() for l in text[n].splitlines()[1:] if l.strip()]
  intr={(int(r[0]),int(r[1])):list(map(float,r[2:])) for r in rows('intrinsic.txt')}
  ext={(int(r[0]),int(r[1])):np.asarray(list(map(float,r[2:]))).reshape(4,4) for r in rows('extrinsic.txt')}
  colors={tuple(map(int,r[1:])):r[0] for r in rows('colors.txt')}
  box={}
  for r in rows('bbox.txt'):box.setdefault((int(r[0]),int(r[1])),[]).append(r)
  fx,fy,cx,cy=intr[start,0];K=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])
  stereo=ext[start,1]@np.linalg.inv(ext[start,0]);baseline=float(np.linalg.norm(stereo[:3,3]));assert abs(baseline-.532725)<.001
  assert np.allclose(stereo[:3,:3],np.eye(3),atol=1e-5) and abs(stereo[0,3]+baseline)<1e-4
  records=[];poses=[]
  for i,f in enumerate(range(start,start+26)):
   assert intr[f,0]==intr[start,0] and intr[f,1]==intr[start,1]
   left=np.asarray(img('rgb',prefix+f'frames/rgb/Camera_0/rgb_{f:05d}.jpg').convert('RGB'));right=np.asarray(img('rgb',prefix+f'frames/rgb/Camera_1/rgb_{f:05d}.jpg').convert('RGB'));h,w=left.shape[:2]
   assert left.shape==right.shape
   Image.fromarray(left).save(d/'image_2'/f'{i:06d}.png');Image.fromarray(right).save(d/'image_3'/f'{i:06d}.png')
   gd=np.asarray(img('depth',prefix+f'frames/depth/Camera_0/depth_{f:05d}.png'),dtype=np.uint16).astype(np.float32)/100
   inst_image=img('instance',prefix+f'frames/instanceSegmentation/Camera_0/instancegt_{f:05d}.png');assert inst_image.mode=='P';inst=np.asarray(inst_image)
   sem=np.asarray(img('class',prefix+f'frames/classSegmentation/Camera_0/classgt_{f:05d}.png').convert('RGB'))
   gt=np.full((h,w),255,np.uint8);matched=np.zeros((h,w),bool)
   for color,category in colors.items():
    m=(sem==color).all(2);matched|=m;gt[m]=mapping.get(category,255)
   assert matched.all(),'unknown official RGB color'
   moving=np.zeros((h,w),bool)
   for r in box.get((f,0),[]):
    m=inst==int(r[2])+1;assert int(m.sum())==int(r[7])
    if r[10]=='True':moving|=m
   disparity=cv2.StereoSGBM_create(minDisparity=0,numDisparities=256,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY).compute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)).astype(np.float32)/16
   depth=np.zeros((h,w),np.float32);valid=disparity>0;depth[valid]=fx*baseline/disparity[valid];depth[(depth<.5)|(depth>80)]=0
   np.save(d/'depth_sgbm'/f'{i:06d}_depth.npy',depth);np.save(d/'gt_depth'/f'{i:06d}.npy',gd);np.save(d/'gt_labels'/f'{i:06d}.npy',gt);np.save(d/'gt_instances'/f'{i:06d}.npy',inst);np.save(d/'gt_moving'/f'{i:06d}.npy',moving)
   c2w=np.linalg.inv(ext[f,0]);assert np.allclose(c2w@ext[f,0],np.eye(4),atol=1e-5);poses.append(c2w)
   # Independent pose/depth reprojection to the right camera checks sign/scale.
   y,x=np.mgrid[0:h:25,0:w:25];z=gd[y,x];xyz=np.stack([(x-cx)*z/fx,(y-cy)*z/fy,z,np.ones_like(z)],-1)
   transformed=xyz@stereo.T;u=fx*transformed[...,0]/transformed[...,2]+cx
   assert np.allclose(u,x-fx*baseline/z,atol=.01)
   right_gt_depth=np.asarray(img('depth',prefix+f'frames/depth/Camera_1/depth_{f:05d}.png'),dtype=np.uint16).astype(np.float32)/100
   ur=np.rint(u).astype(int);inside=(ur>=0)&(ur<w)&(z>=.5)&(z<=80);ur=np.clip(ur,0,w-1)
   relative=np.abs(right_gt_depth[y,ur]-z)/np.maximum(z,.5)
   assert int(inside.sum())>50 and float(np.median(relative[inside]))<.03, 'stereo GT reprojection discrepancy'
   paths={n:(d/n/(f'{i:06d}_depth.npy' if n=='depth_sgbm' else f'{i:06d}.png' if n.startswith('image') else f'{i:06d}.npy')) for n in ['image_2','image_3','depth_sgbm','gt_depth','gt_labels','gt_instances','gt_moving']}
   records.append({'local_frame':i,'original_frame':f,'files':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'valid_gt_pixels':int((gt!=255).sum()),'stereo_GT_depth_reprojection_median_relative_error':float(np.median(relative[inside]))})
  np.savetxt(d/'poses.txt',np.asarray(poses)[:,:3,:].reshape(26,12));np.save(d/'c2w.npy',np.asarray(poses))
  cfg={'dataset_name':'kitti','depth_folder':str(d/'depth_sgbm'),'pose_path':str(d/'poses.txt'),'camera_params':{'image_height':h,'image_width':w,'fx':fx,'fy':fy,'cx':cx,'cy':cy,'png_depth_scale':1.0}}
  import yaml
  (d/'dataset.yaml').write_text(yaml.safe_dump(cfg))
  conf=runpy.run_path(a.mapping_template)['config'];conf.update(workdir=str(d/'mapping'),run_name='fixed_sgbm_gtpose_10iter',eval_every=26,report_global_progress_every=26,use_wandb=False)
  conf['data'].update(basedir=str(out),sequence=name,gradslam_data_cfg=str(d/'dataset.yaml'),desired_image_height=h,desired_image_width=w,num_frames=26)
  conf['mapping']['num_iters']=10;conf['mapping']['prune_gaussians']=False;conf['tracking']['use_gt_poses']=True
  (d/'mapping_config.py').write_text('config = '+repr(conf)+'\n')
  evaluation=[4,9,14,19,24];fusion=[i for i in range(26) if i not in evaluation]
  sequences.append({'name':name,'scene':scene,'variation':'clone','path':str(d),'height':h,'width':w,'K':K.tolist(),'baseline_m':baseline,'frame_count':26,'records':records,'fusion':fusion,'evaluation':evaluation,'exposure':'HISTORICALLY_EXPOSED_DEV; geometry uses all26; current semantic excludes5','pose_source':'official VKITTI2 extrinsic inverted to c2w; not tracking result'})
 dump(out/'manifest.json',{'status':'CPU_ADAPTER_READY_GT_DEV_NOT_FINAL','owner':'CHY','sequences':sequences,'gt_archive_md5':md5,'gt_archive_sha256':sha(gtarchive),'mapping':mapping,'evaluation_classes':[0,2,5,6,7,8,9,10,13,14],'geometry_depth':'predicted SGBM; official depth evaluation only','GT_used_in_main_fusion':False,'small_vehicle_area_fraction':.0025,'boundary_radius_pixels':3,'final_split':[],'history':'OFF','map_iterations':10,'no_scientific_winner_declared':True})
 for t in handles.values():t.close()
 print(json.dumps({'status':'PASS_CPU_ADAPTER','scenes':2,'frames':52,'GT_archive_md5':md5}),flush=True)
if __name__=='__main__':main()
