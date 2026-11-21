"""One fresh complete no-GT DG-SLAM window; raw outputs only, no scoring."""
from pathlib import Path
import argparse,csv,functools,hashlib,importlib,json,os,time,types
import numpy as np
from runtime_boot import F,R,DATA,P,BASE,bootstrap,guard_gt,sha,dump
def main():
 a=argparse.ArgumentParser();a.add_argument('--window',choices=['SMOKE200','DEV200'],required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--arm',choices=P['arms'],required=True);args0=a.parse_args()
 smoke=args0.window=='SMOKE200';assert args0.seed==(1219 if smoke else args0.seed)
 if not smoke:assert args0.seed in P['seeds'] and (F/'execution_lock.json').exists()
 if (F/'execution_lock.json').exists():
  lock=json.loads((F/'execution_lock.json').read_text());assert sha(F/'protocol.json')==lock['protocol_sha256']
  for path,h in lock['runtime_immutable_files'].items():assert sha(path)==h,path
 N=200
 native=bootstrap();audit=guard_gt()
 import torch,lietorch,cv2
 from lietorch import SE3
 from scipy.spatial.transform import Rotation
 from dg_slam.config import load_config
 from dg_slam.dg_model import dg_model
 from dg_slam.gaussian.common import setup_seed
 from live_refinement import local_tracking
 output=F/('smoke' if smoke else 'runs')/f'{args0.window}_s{args0.seed}_{args0.arm}'
 output.mkdir(parents=True,exist_ok=False)
 cfgpath='configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz.yaml';cfg=load_config(cfgpath,'configs/dg_slam.yaml')
 cfg['data'].update(n_img=N,output=str(output),exp_name='T17_'+args0.arm);cfg.update(wandb=False,gs_pose_refine=args0.arm!='NO_GS_REFINE',setup_seed=args0.seed)
 assert cfg['tracking']['iters']==50 and cfg['mapping']['iters_first']==500 and cfg['mapping']['iters']==100 and not cfg['mapping']['BA']
 assert cfg['tracking']['w_color_loss']==.9 and cfg['tracking']['w_geo_loss']==.1
 args=argparse.Namespace(weights=str(R/'01_source_env/weights/droid.pth'),buffer=232,image_size=[384,512],stereo=False,disable_vis=True,plot_curve=False,beta=.6,filter_thresh=1.75,warmup=12,keyframe_thresh=2.25,frontend_thresh=12.,frontend_window=25,frontend_radius=2,frontend_nms=1,backend_thresh=15.,backend_radius=2,backend_nms=3,config=cfgpath)
 setup_seed(args0.seed)
 counts={k:0 for k in ['input_frames','gs_tracking_calls','gs_mapping_calls','mapping_optimize_calls','gs_render_calls','camera_backward','camera_accepted_updates','support_fallbacks']}
 started=time.time();meta={'status':'RUNNING','window':args0.window,'seed':args0.seed,'arm':args0.arm,'formal':False,'registered_DEV':not smoke,'protocol_sha256':sha(F/'protocol.json'),'execution_lock_sha256':sha(F/'execution_lock.json') if not smoke else None,'cfg':cfg,'args':vars(args),'natives':native,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
 def save():
  meta.update(counts=counts,GT_guard=audit,seconds=time.time()-started,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated());dump(output/'run_manifest.json',meta)
 save()
 try:
  rows=list(csv.DictReader((F/f'inputs/{args0.window}_runtime.csv').open()));assert len(rows)==N
  assert all(not any('gt' in key.lower() for key in row) for row in rows)
  data=[];pixel_ids={}
  for t,row in enumerate(rows):
   rgb=cv2.cvtColor(cv2.resize(cv2.imread(str(DATA/row['rgb_path'])),(512,384)),cv2.COLOR_BGR2RGB)
   image=torch.from_numpy(rgb[None]).permute(0,3,1,2)
   d=cv2.imread(str(DATA/row['depth_path']),cv2.IMREAD_UNCHANGED);assert d is not None
   depth=torch.from_numpy(cv2.resize(d,(512,384)).astype(np.float32)/5000.)
   maskfile=(DATA/'seg_mask')/Path(row['rgb_path']).name
   labels=cv2.imread(str(maskfile),cv2.IMREAD_GRAYSCALE);assert labels is not None and labels.shape==(480,640) and labels.max()<150
   mask=cv2.dilate((labels==12).astype(np.uint8),np.ones((8,8),np.uint8),iterations=1);mask=torch.from_numpy(cv2.resize(mask,(512,384)).astype(np.uint8))
   K=.8*torch.tensor([535.4,539.2,320.1,247.6]);data.append((t,image,depth,K,None,mask))
   pixel_ids[hashlib.sha256(image[0].numpy().tobytes()).hexdigest()]=t
  assert len(pixel_ids)==N
  model=dg_model(cfg,args);m=model.tracking_mapping
  assert not hasattr(model.video,'poses_gt') and not hasattr(m,'gt_c2w_list')
  assert Path(importlib.import_module(type(model).__module__).__file__).resolve().is_relative_to((F/'source').resolve())
  frame_rows=[];active=None
  original_dispatch=m.run
  def dispatch(idx,*values,**kw):
   nonlocal active
   k=int(idx);h=hashlib.sha256(model.video.images[k].cpu().numpy().tobytes()).hexdigest();actual=pixel_ids[h];recorded=int(model.video.tstamp[k]);assert actual==recorded
   active={'keyframe':k,'local_input_frame':actual,'selected_frame':int(rows[actual]['selected_frame']),'rgb_sha256_resized':h,'coarse_c2w':values[2].detach().cpu().numpy().tolist()}
   result=original_dispatch(idx,*values,**kw)
   active.update(mapped_c2w=m.estimate_c2w_list[k].cpu().numpy().tolist(),gaussian_count=int(m.gaussians.get_xyz().shape[0]));assert active['gaussian_count']<=500000
   frame_rows.append(active)
   with (output/'keyframe_audit.jsonl').open('a') as f:f.write(json.dumps(active,allow_nan=False)+'\n')
   active=None;return result
  m.run=dispatch
  original_tracking=m.tracking
  def tracking(idx,color,depth,estimated,seg):
   counts['gs_tracking_calls']+=1
   if args0.arm=='NO_GS_REFINE':result=original_tracking(idx,color,depth,estimated,seg)
   else:result=local_tracking(m,idx,color,depth,estimated,seg,args0.arm,output,counts);active['local_refinement']=result
   active['tracked_c2w']=m.estimate_c2w_list[int(idx)].cpu().numpy().tolist();return result
  m.tracking=tracking
  for method,key in [('mapping','gs_mapping_calls'),('optimize_map','mapping_optimize_calls')]:
   fn=getattr(m,method)
   def wrapper(*values,_fn=fn,_key=key,**kw):counts[_key]+=1;return _fn(*values,**kw)
   setattr(m,method,wrapper)
  author_opt=m.optimize_cam_in_batch
  def opt(*values,**kw):counts['camera_backward']+=1;return author_opt(*values,**kw)
  m.optimize_cam_in_batch=opt
  mod=importlib.import_module(type(m).__module__);oldrender=mod.render
  def render(*values,**kw):counts['gs_render_calls']+=1;return oldrender(*values,**kw)
  mod.render=render
  for t,image,depth,K,none,mask in data:
   assert none is None;model.track(t,image,depth,K,mask);counts['input_frames']+=1
   if counts['input_frames']%10==0:save();print('FRAME',counts['input_frames'],'KF',model.video.counter.value,flush=True)
  n=int(model.video.counter.value);assert len(frame_rows)==n==model.mapping_counter and n>2 and model.frontend.is_initialized
  matrices=m.estimate_c2w_list[:n].cpu().numpy();assert np.allclose(matrices[0],np.eye(4),atol=1e-6)
  arrays={key:getattr(m.gaussians,method)().detach().cpu().numpy() for key,method in [('xyz','get_xyz'),('features_dc','get_features_dc'),('features_rest','get_features_rest'),('scaling','get_scaling'),('rotation','get_rotation'),('opacity','get_opacity')]}
  assert arrays['xyz'].shape[0]>0 and all(np.isfinite(v).all() for v in arrays.values())
  ids=model.video.tstamp[:n].cpu().numpy();assert (np.diff(ids)>0).all() and ids.min()>=0 and ids.max()<N
  np.savez_compressed(output/'gaussian_snapshot.npz',**arrays,estimated_keyframe_c2w=matrices,keyframe_input_indices=ids)
  seeds=torch.from_numpy(np.concatenate([matrices[:,:3,3],Rotation.from_matrix(matrices[:,:3,:3]).as_quat()],axis=1)).float()
  assert np.allclose(SE3(seeds).matrix().numpy(),matrices,atol=1e-5)
  before_fill=model.video.poses[:n].detach().clone();np.save(output/'droid_pose_seeds.npy',seeds.numpy())
  model.video.poses[:n]=lietorch.cat([SE3(seeds)],0).inv().data
  trajectory=model.terminate_woBA(data)
  assert trajectory.shape==(N,7) and np.isfinite(trajectory).all() and np.allclose(np.linalg.norm(trajectory[:,3:],axis=1),1,atol=1e-4)
  np.savez(output/'trajectories.npz',estimated=trajectory,rgb_timestamps=np.asarray([float(r['rgb_timestamp']) for r in rows]),selected_frames=np.asarray([int(r['selected_frame']) for r in rows]))
  np.savez(output/'keyframe_poses.npz',indices=ids,coarse=np.asarray([v['coarse_c2w'] for v in frame_rows]),tracked=np.asarray([v['tracked_c2w'] for v in frame_rows]),mapped=matrices,droid_before_injection=before_fill.cpu().numpy(),droid_after_filler=model.video.poses[:n].cpu().numpy())
  assert audit['blocked_GT_opens']==0 and counts['gs_mapping_calls']>0
  assert (counts['camera_backward']==0 if args0.arm=='NO_GS_REFINE' else counts['camera_backward']>0)
  if args0.arm=='NO_GS_REFINE':assert np.allclose(np.asarray([v['coarse_c2w'] for v in frame_rows]),np.asarray([v['tracked_c2w'] for v in frame_rows]),atol=1e-6)
  size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());assert size<=512*1024**2
  meta.update(status='COMPLETED',droid_keyframes=n,mapped_keyframes=model.mapping_counter,gaussian_count=int(arrays['xyz'].shape[0]),mapping_identity_audit='PASS',first_pose_identity=True,
   output_bytes=size,artifacts={p.name:sha(p) for p in output.glob('*') if p.is_file() and p.name!='run_manifest.json'})
 except Exception as e:
  meta.update(status='FAILED',error=repr(e))
  if 'model' in locals():meta.update(frontend_initialized=model.frontend.is_initialized,video_keyframes=int(model.video.counter.value))
  raise
 finally:save()
if __name__=='__main__':main()
