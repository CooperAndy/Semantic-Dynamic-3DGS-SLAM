"""Receipt-backed isolated pipeline; no reconstruction on final RGB views."""
import hashlib,json,subprocess,time,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/kitti360_isolated_v1';B=R/'03_方法冻结与隔离评价/kitti360_isolated_v1';B.mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipts=[]
def phase(name,cmd,cwd=None):
 dump(B/'pipeline_status.json',{'status':'RUNNING','phase':name,'completed':receipts})
 with (B/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 receipt={'name':name,'command':cmd,'exit_code':r.returncode,'log_sha256':sha(B/(name+'.log'))};receipts.append(receipt);dump(B/(name+'_receipt.json'),receipt)
 if r.returncode:raise RuntimeError(name+' failed; no metric-driven retry authorized')
try:
 deadline=time.time()+1800
 while not (D/'manifest.json').exists():
  dump(B/'pipeline_status.json',{'status':'WAITING','phase':'frozen_input_adapter'})
  if time.time()>deadline:raise TimeoutError('adapter manifest absent')
  time.sleep(5)
 assert not (B/'GT_final_evaluation').exists()
 names=['run_kitti360_pipeline.py','run_kitti360_fusion.py','evaluate_kitti360_isolated.py','prepare_kitti360_isolated.py','http_range_zip.py','semantic_cpu.py','native_semantic_render.py','evaluate_vkitti_gt_dev.py','vkitti_mapping_template.py']
 locks=[{'path':'scripts/'+n,'sha256':sha(R/'scripts'/n)} for n in names]
 dump(B/'source_freeze.json',{'status':'CODE_FROZEN_BEFORE_FIRST_GPU_OR_GT_METRIC','files':locks,'GT_metrics_existed':False,'protocol_sha256':sha(R/'protocol/KITTI360隔离评价预声明_20261007_v1.json'),'adapter_repair':'member path prefix corrected before any GT pixels decoded; frame/metric protocol unchanged'})
 man=json.loads((D/'manifest.json').read_text())
 for s in man['sequences']:
  d=Path(s['path']);name=s['name']
  phase(name+'_prediction',['/root/venvs/semantic_seg/bin/python','/root/autodl-tmp/slam_project/stage2_semantic/scripts/generate_mask2former_semantics_kitti.py','--image-dir',str(d/'image_2'),'--output-dir',str(d/'prediction'),'--model-id','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-semantic-91281d04','--local-files-only','--num-frames','21','--device','cuda','--preview-every','21'])
  phase(name+'_mapping',['/root/miniconda3/bin/python','/root/autodl-tmp/SplaTAM/scripts/splatam.py',str(d/'mapping_config.py')],'/root/autodl-tmp/SplaTAM')
  source=d/'mapping/fusion21_fixed_sgbm_pose_10iter/params.npz'
  with np.load(source,allow_pickle=False) as p:base={k:p[k] for k in p.files}
  assert base['cam_trans'].shape[-1]==21 and np.allclose(base['w2c'],np.eye(4),atol=1e-5)
  poses=np.load(d/'c2w.npy');w2cs=np.linalg.inv(poses);rot=np.zeros((1,4,26),np.float32);tran=np.zeros((1,3,26),np.float32)
  for i in range(26):
   if i in s['fusion']:
    j=s['fusion'].index(i);q=base['cam_unnorm_rots'][0,:,j];t=base['cam_trans'][0,:,j];assert np.allclose(Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),w2cs[i,:3,:3],atol=1e-5) and np.allclose(t,w2cs[i,:3,3],atol=1e-4)
   else:q=Rotation.from_matrix(w2cs[i,:3,:3]).as_quat()[[3,0,1,2]].astype(np.float32);t=w2cs[i,:3,3].astype(np.float32)
   rot[0,:,i]=q;tran[0,:,i]=t
  base['cam_unnorm_rots']=rot;base['cam_trans']=tran;base['gt_w2c_all_frames']=w2cs.astype(np.float32);base['source_gaussian_index']=np.arange(len(base['means3D']),dtype=np.int64);np.savez_compressed(d/'base_map.npz',**base)
  dump(d/'base_map_lock.json',{'source':str(source),'source_sha256':sha(source),'file_sha256':sha(d/'base_map.npz'),'gaussians':len(base['means3D']),'geometry_frames':s['fusion'],'semantic_frames':s['fusion'],'evaluation_frames_excluded_from_RGB_depth_and_geometry':s['evaluation'],'fusion_camera_parameters_copied_unchanged':True,'eval_camera_appended_from_external_pose_only':True,'pose_rotation_and_translation_check_passed':True})
  del base
  phase(name+'_fusion',['/root/miniconda3/bin/python',str(R/'scripts/run_kitti360_fusion.py'),'--root',str(R),'--sequence',name])
 phase('CPU_final_GT_evaluation',['/root/miniconda3/bin/python',str(R/'scripts/evaluate_kitti360_isolated.py')])
 dump(B/'pipeline_status.json',{'status':'COMPLETE_PROJECT_HOLDOUT_MEASUREMENT','completed':receipts,'scientific_PASS':False,'reference':'S0','no_postholdout_tuning':True})
except Exception as e:
 dump(B/'pipeline_status.json',{'status':'FAILED','error':str(e),'completed':receipts});print(str(e),flush=True);sys.exit(1)
