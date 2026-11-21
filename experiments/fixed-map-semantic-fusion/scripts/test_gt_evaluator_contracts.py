"""Small ground-truth scoring/reprojection fixtures, no GPU/model execution."""
import json,tempfile
from pathlib import Path
import numpy as np
from evaluate_vkitti_gt_dev import confusion,metrics,temporal_masks
gt=np.array([[0,0,13,255]]);pred=np.array([[-1,-2,13,0]])
r=metrics(confusion(gt,pred,gt!=255))
assert r['GT_valid_pixels']==3 and r['coverage']==1/3 and r['mIoU_GT_supported_common_classes']==.5
assert r['unknown_pixels']==1 and r['no_map_pixels']==1
assert next(c for c in r['classes'] if c['class']==0)['FN_includes_unknown_and_no_map']==2
empty=metrics(confusion(gt,pred,np.zeros_like(gt,dtype=bool)))
assert empty['mIoU_GT_supported_common_classes'] is None and empty['coverage'] is None
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp)
 for kind in ['gt_labels','gt_depth','gt_moving']:(d/kind).mkdir()
 for f in [0,1]:
  np.save(d/'gt_labels'/f'{f:06d}.npy',np.zeros((5,5),np.uint8))
  np.save(d/'gt_depth'/f'{f:06d}.npy',np.full((5,5),10,np.float32))
  moving=np.zeros((5,5),bool)
  if f==1:moving[2,2]=True
  np.save(d/'gt_moving'/f'{f:06d}.npy',moving)
 np.save(d/'c2w.npy',np.stack([np.eye(4)]*2))
 stable,changed=temporal_masks(d,{'K':[[10,0,2],[0,10,2],[0,0,1]]},1)
 assert stable.sum()==24 and changed.sum()==1 and changed[2,2] and not (stable&changed).any()
print(json.dumps({'status':'PASS_GT_EVALUATOR_CONTRACTS','unknown_no_map_true_class_misses':True,'ignore_excluded':True,'empty_subgroup_NOT_EVALUABLE':True,'controlled_plane_and_moving_pixel_temporal_masks':True,'gpu_used':False}))
