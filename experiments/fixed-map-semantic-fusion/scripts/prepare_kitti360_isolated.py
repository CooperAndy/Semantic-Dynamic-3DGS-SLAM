"""Build only fusion images/depth; final views have pose+GT only."""
import io,json,hashlib,zipfile,runpy
from pathlib import Path
import numpy as np,cv2,yaml
from PIL import Image
from http_range_zip import RangeZip
from audit_kitti360_entry import ROOT,D,URLS,dump
CITY_IDS=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 selection=json.loads((D/'metadata_selection.json').read_text());protocol=ROOT/'protocol/KITTI360隔离评价预声明_20261007_v1.json'
 frozen=json.loads(protocol.read_text());assert frozen['metadata_selection_sha256']==sha(D/'metadata_selection.json')
 assert frozen['GT_pixels_read_before_freeze']==False
 labels=RangeZip(URLS['labels'],256*1024*1024);assert labels.etag==selection['labels_archive']['etag'];sequences=[];transfer=[]
 calib={}
 for line in (D/'calibration/perspective.txt').read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   try:calib[k]=np.array([float(x) for x in v.split()])
   except ValueError:pass
 P0=calib['P_rect_00'].reshape(3,4);P1=calib['P_rect_01'].reshape(3,4);K=P0[:,:3];baseline=abs(P1[0,3]/P1[0,0]-P0[0,3]/P0[0,0]);assert .4<baseline<.8
 h,w=int(calib['S_rect_00'][1]),int(calib['S_rect_00'][0]);assert (h,w)==(376,1408)
 with zipfile.ZipFile(D/'poses.zip') as pz:
  for s in selection['sequences']:
   name=s['sequence'];d=D/name;d.mkdir(exist_ok=True);fusion=s['fusion_local'];evals=s['evaluation_local']
   for n in ['image_2','image_3','depth_sgbm','gt_labels','gt_instances']:(d/n).mkdir(exist_ok=True)
   pose_member=next(n for n in pz.namelist() if n.endswith(f'{name}/cam0_to_world.txt'));raw=pz.read(pose_member);(d/'official_cam0_to_world.txt').write_bytes(raw)
   rows=np.loadtxt(io.BytesIO(raw));poses={int(r[0]):r[1:].reshape(4,4) for r in rows};c2ws=np.asarray([poses[f] for f in s['frames']]);assert np.allclose(c2ws[:,-1],[0,0,0,1])
   relative=np.linalg.inv(c2ws[0])@c2ws;assert np.allclose(relative[0],np.eye(4),atol=1e-6);np.save(d/'c2w.npy',relative);np.savetxt(d/'poses.txt',relative[fusion,:3,:].reshape(-1,12))
   images={cam:RangeZip(s['image_archives'][cam]['url'],160*1024*1024) for cam in ['00','01']}
   assert all(images[k].etag==s['image_archives'][k]['etag'] for k in images)
   records=[]
   for i,f in enumerate(s['frames']):
    files={};pi=fusion.index(i) if i in fusion else None
    if pi is not None:
     rgb={}
     for cam,folder in [('00','image_2'),('01','image_3')]:
      b=images[cam].read(s['image_members'][cam][str(f)]);p=d/folder/f'{pi:06d}.png';p.write_bytes(b);rgb[cam]=np.asarray(Image.open(io.BytesIO(b)).convert('RGB'));files[folder]={'path':str(p),'sha256':sha(p)}
     assert rgb['00'].shape==rgb['01'].shape==(h,w,3)
     disp=cv2.StereoSGBM_create(minDisparity=0,numDisparities=256,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY).compute(cv2.cvtColor(rgb['00'],cv2.COLOR_RGB2GRAY),cv2.cvtColor(rgb['01'],cv2.COLOR_RGB2GRAY)).astype(np.float32)/16
     dep=np.zeros((h,w),np.float32);v=disp>0;dep[v]=K[0,0]*baseline/disp[v];dep[(dep<.5)|(dep>80)]=0;p=d/'depth_sgbm'/f'{pi:06d}_depth.npy';np.save(p,dep);files['depth_sgbm']={'path':str(p),'sha256':sha(p)}
    # Evaluation GT only; fusion GT deliberately not extracted, no final Oracle.
    if i in evals:
     b=labels.read(s['semantic_members'][str(f)]);rawgt=np.asarray(Image.open(io.BytesIO(b)));gt=np.full(rawgt.shape,255,np.uint8)
     for trainid,semanticid in enumerate(CITY_IDS):gt[rawgt==semanticid]=trainid
     inst=np.asarray(Image.open(io.BytesIO(labels.read(s['instance_members'][str(f)]))));assert gt.shape==inst.shape==(h,w)
     for kind,value in [('gt_labels',gt),('gt_instances',inst)]:p=d/kind/f'{i:06d}.npy';np.save(p,value);files[kind]={'path':str(p),'sha256':sha(p)}
    records.append({'local_frame':i,'original_frame':f,'prediction_frame':pi,'role':'FUSION_GEOMETRY' if pi is not None else 'FINAL_EVALUATION','files':files})
   cfg={'dataset_name':'kitti','depth_folder':str(d/'depth_sgbm'),'pose_path':str(d/'poses.txt'),'camera_params':{'image_height':h,'image_width':w,'fx':float(K[0,0]),'fy':float(K[1,1]),'cx':float(K[0,2]),'cy':float(K[1,2]),'png_depth_scale':1.0}}
   (d/'dataset.yaml').write_text(yaml.safe_dump(cfg));conf=runpy.run_path(str(ROOT/'scripts/vkitti_mapping_template.py'))['config'];conf.update(workdir=str(d/'mapping'),run_name='fusion21_fixed_sgbm_pose_10iter',eval_every=21,report_global_progress_every=21,use_wandb=False)
   conf['data'].update(basedir=str(D),sequence=name,gradslam_data_cfg=str(d/'dataset.yaml'),desired_image_height=h,desired_image_width=w,num_frames=21);conf['mapping']['num_iters']=10;conf['mapping']['prune_gaussians']=False;conf['tracking']['use_gt_poses']=True;(d/'mapping_config.py').write_text('config = '+repr(conf)+'\n')
   sequences.append({'name':name,'scene':name,'path':str(d),'height':h,'width':w,'K':K.tolist(),'baseline_m':float(baseline),'frame_count':26,'records':records,'fusion':fusion,'evaluation':evals,'geometry_frames':fusion,'pose_source':'official optimized rectified cam0_to_world; first camera normalization; all-sequence external pose estimates, not tracking accuracy GT','exposure':'project-new small public train/val holdout; not official hidden benchmark Test'})
   transfer.extend([z.provenance() for z in images.values()])
 transfer.append(labels.provenance())
 dump(D/'manifest.json',{'status':'ISOLATED_FINAL_INPUT_READY','owner':'CHY','sequences':sequences,'protocol_sha256':sha(protocol),'class_mapping':{str(k):i for i,k in enumerate(CITY_IDS)},'evaluation_classes':list(range(19)),'GT_in_geometry_or_fusion':False,'evaluation_RGB_and_depth_extracted':False,'geometry_depth':'SGBM predicted fusion frames only','transfer_receipts':transfer,'history':'OFF','map_iterations':10})
 print(json.dumps({'status':'PASS_ISOLATED_CPU_ADAPTER','sequences':2,'fusion_RGB_pairs':42,'evaluation_GT_views':10,'final_RGB_in_map':False}),flush=True)
if __name__=='__main__':main()
