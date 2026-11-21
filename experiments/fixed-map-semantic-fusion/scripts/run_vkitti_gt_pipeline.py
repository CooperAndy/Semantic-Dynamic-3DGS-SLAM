"""Sequential, receipt-backed DEV execution. Never consumes old Scene20/Test."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/stage3_chy_20261006');out=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2';out.mkdir(parents=True,exist_ok=True)
status=out/'pipeline_status.json';receipts=[]
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def phase(name,command,cwd=None):
 dump(status,{'status':'RUNNING','phase':name,'completed':receipts});print(name,flush=True)
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
 receipt={'name':name,'command':command,'exit_code':r.returncode,'log':str(out/(name+'.log'))};receipts.append(receipt);dump(out/(name+'_receipt.json'),receipt)
 if r.returncode:raise RuntimeError(name+' failed; inspect log')
try:
 gt=root/'inputs/vkitti2_gt_dev_v2/vkitti_2.0.3_classSegmentation.tar'
 dump(status,{'status':'RUNNING','phase':'official_GT_download','completed':[]})
 deadline=time.time()+1800
 while True:
  if gt.exists() and gt.stat().st_size>=900000000 and hashlib.md5(gt.read_bytes()).hexdigest()=='e8658ab49250e61f1caa174625563263':break
  if time.time()>deadline:raise TimeoutError('official GT download incomplete')
  time.sleep(20)
 phase('01_cpu_adapter',['/root/miniconda3/bin/python',str(root/'scripts/prepare_vkitti_gt_dev.py'),'--root',str(root),'--mapping-template',str(root/'scripts/vkitti_mapping_template.py')])
 manifest=json.loads((root/'inputs/vkitti2_gt_dev_v2/manifest.json').read_text())
 for seq in manifest['sequences']:
  d=Path(seq['path']);name=seq['name']
  phase(name+'_02_prediction',['/root/venvs/semantic_seg/bin/python','/root/autodl-tmp/slam_project/stage2_semantic/scripts/generate_mask2former_semantics_kitti.py','--image-dir',str(d/'image_2'),'--output-dir',str(d/'prediction'),'--model-id','/root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-semantic-91281d04','--local-files-only','--num-frames','26','--device','cuda','--preview-every','13'])
  phase(name+'_03_mapping',['/root/miniconda3/bin/python','/root/autodl-tmp/SplaTAM/scripts/splatam.py',str(d/'mapping_config.py')],'/root/autodl-tmp/SplaTAM')
  import numpy as np
  source=d/'mapping/fixed_sgbm_gtpose_10iter/params.npz'
  with np.load(source,allow_pickle=False) as p:base={k:p[k] for k in p.files}
  base['source_gaussian_index']=np.arange(len(base['means3D']),dtype=np.int64)
  assert base['cam_trans'].shape[-1]==26 and int(base['org_height'])==seq['height'] and int(base['org_width'])==seq['width']
  np.savez_compressed(d/'base_map.npz',**base)
  dump(d/'base_map_lock.json',{'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'file_sha256':hashlib.sha256((d/'base_map.npz').read_bytes()).hexdigest(),'gaussians':len(base['means3D']),'geometry_frames':list(range(26)),'pose_source':seq['pose_source'],'depth':'SGBM predicted','mapping_iters':10})
  del base
  for mode in ['smoke','full']:phase(name+'_04_fusion_'+mode,['/root/miniconda3/bin/python',str(root/'scripts/run_vkitti_gt_fusion.py'),'--root',str(root),'--sequence',name,'--mode',mode])
 phase('05_cpu_GT_evaluation',['/root/miniconda3/bin/python',str(root/'scripts/evaluate_vkitti_gt_dev.py'),'--root',str(root)])
 dump(status,{'status':'COMPLETE_GT_DEV_NOT_ISOLATED_FINAL_TEST','completed':receipts,'scientific_PASS':False})
except Exception as e:
 dump(status,{'status':'FAILED','error':str(e),'completed':receipts});print(str(e),flush=True);sys.exit(1)
