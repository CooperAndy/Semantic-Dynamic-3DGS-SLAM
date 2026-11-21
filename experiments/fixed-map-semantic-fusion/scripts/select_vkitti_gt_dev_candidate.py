"""Apply predeclared practical DEV gates; no final-Test or statistical PASS."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('/root/autodl-tmp/stage3_chy_20261006');ev=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2/GT_evaluation_v1'
def read(p):return json.loads(p.read_text())
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
policy_path=root/'protocol/VKITTI2_GT_DEV判优预声明_20261007_v2.json';policy=read(policy_path)
assert hashlib.sha256(policy_path.read_bytes()).hexdigest()=='5d98a66aed7784f19552491de4a2c3362ff56a31ecb3479e080fe8f6759caef4'
man=read(root/'inputs/vkitti2_gt_dev_v2/manifest.json');rows=read(ev/'GT_main_results.json');sub=read(ev/'GT_subgroup_results.json');checks=[];small_instances=set()
for seq in man['sequences']:
 d=Path(seq['path'])
 for f in seq['evaluation']:
  gt=np.load(d/'gt_labels'/f'{f:06d}.npy');inst=np.load(d/'gt_instances'/f'{f:06d}.npy')
  for identity,count in zip(*np.unique(inst[inst>0],return_counts=True)):
   if count<=.0025*gt.size and np.isin(gt[inst==identity],[13,14]).any():small_instances.add((seq['scene'],int(identity)))
def r(scene,g):return next(x for x in rows if x['sequence']==scene and x['group']==g)
def s(scene,g,key):return next(x for x in sub if x['sequence']==scene and x['group']==g and x['subgroup']==key)
for g in policy['candidates']:
 gains=[];fail=[];missing=[]
 if len(small_instances)<policy['minimum_total_distinct_small_vehicle_instances']:missing.append('insufficient distinct small vehicle instances')
 for seq in man['sequences']:
  scene=seq['name'];baseline=r(scene,'S0');current=r(scene,g);gain=current['mIoU_GT_supported_common_classes']-baseline['mIoU_GT_supported_common_classes'];gains.append(gain)
  if gain < -policy['maximum_single_scene_mIoU_loss']:fail.append(scene+': whole mIoU loss')
  if baseline['coverage']-current['coverage']>policy['maximum_single_scene_coverage_loss']:fail.append(scene+': coverage loss')
  for key in policy['required_subgroups']:
   b=s(scene,'S0',key);c=s(scene,g,key)
   if b['GT_valid_pixels']<policy['minimum_subgroup_GT_pixels_each_scene'] or b['mIoU_GT_supported_common_classes'] is None:missing.append(scene+': '+key+' unsupported')
   elif b['mIoU_GT_supported_common_classes']-c['mIoU_GT_supported_common_classes']>policy['maximum_subgroup_mIoU_loss']:fail.append(scene+': '+key+' mIoU loss')
  bcar=next(x for x in baseline['classes'] if x['class']==13);ccar=next(x for x in current['classes'] if x['class']==13)
  if bcar['GT_pixels']<policy['minimum_car_GT_pixels_each_scene']:missing.append(scene+': car unsupported')
  elif bcar['IoU']-ccar['IoU']>policy['maximum_car_class_IoU_loss']:fail.append(scene+': car IoU loss')
  resource=read(ev.parent/(scene+'_full')/'result.json')
  if resource['peak_torch_GPU_bytes']>policy['maximum_batch_peak_GPU_bytes']:fail.append(scene+': batch memory budget')
  if resource['wall_seconds']>policy['maximum_full_fusion_render_seconds_each_scene']:fail.append(scene+': batch time budget')
 mean=float(np.mean(gains))
 if mean<policy['minimum_equal_scene_mean_mIoU_gain']:fail.append('mean mIoU gain below 1 percentage point')
 checks.append({'group':g,'equal_scene_mean_mIoU_gain':mean,'scene_gains':gains,'failed_gates':fail,'missing_support':missing,'decision':'NOT_EVALUABLE' if missing else 'DEV_ELIGIBLE' if not fail else 'DEV_NOT_ELIGIBLE'})
eligible=[c for c in checks if c['decision']=='DEV_ELIGIBLE']
if eligible:
 best=max(c['equal_scene_mean_mIoU_gain'] for c in eligible);tied=[c for c in eligible if c['equal_scene_mean_mIoU_gain']>=best-.001];rank={'S-G':0,'S1':1,'S2':2,'S2-H':3};selected=min(tied,key=lambda c:rank[c['group']])['group'];status='DEV_CANDIDATE_SELECTED_NOT_FINAL_ACCEPTANCE'
else:selected='S0';status='NO_WEIGHTED_CANDIDATE_MEETS_PREDECLARED_DEV_GATES_BASELINE_RETAINED'
dump(ev/'DEV_candidate_decision.json',{'status':status,'selected_DEV_reference':selected,'quality_improvement_accepted_on_final_test':False,'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'policy_defined_before_first_GT_score_read':True,'distinct_small_vehicle_instances':len(small_instances),'checks':checks,'O_Sem_excluded_from_selection':True,'final_split':[]})
print(json.dumps({'status':status,'selected_DEV_reference':selected,'checks':checks}),flush=True)
