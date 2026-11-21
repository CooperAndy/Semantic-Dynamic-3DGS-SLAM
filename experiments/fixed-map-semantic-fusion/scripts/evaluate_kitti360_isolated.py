"""One-shot frozen project holdout scorer; fixed comparisons, no selection."""
import json,hashlib,csv
from pathlib import Path
import numpy as np,cv2
import evaluate_vkitti_gt_dev as common
from run_kitti360_fusion import GROUPS,sha,dump
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/kitti360_isolated_v1';B=R/'03_方法冻结与隔离评价/kitti360_isolated_v1'
PRIMARY=[0,2,5,6,7,8,9,10,13,14]
def met(cm,classes=PRIMARY):
 original=common.CLASSES;common.CLASSES=classes
 try:return common.metrics(cm)
 finally:common.CLASSES=original
def main():
 protocol=R/'protocol/KITTI360隔离评价预声明_20261007_v1.json';frozen=json.loads(protocol.read_text());man=json.loads((D/'manifest.json').read_text());assert sha(protocol)==man['protocol_sha256'];out=B/'GT_final_evaluation';assert not out.exists();out.mkdir();dump(out/'frozen_protocol.json',frozen)
 source_lock=json.loads((B/'source_freeze.json').read_text());assert all(sha(R/r['path'])==r['sha256'] for r in source_lock['files']);assert source_lock['GT_metrics_existed']==False
 mainrows=[];secondary=[];subrows=[];riskrows=[];matrices={};masklocks=[];support=[]
 for s in man['sequences']:
  name=s['name'];d=Path(s['path']);run=B/name;assert json.loads((run/'result.json').read_text())['GT_opened_by_fusion']==False;assert not set(s['fusion'])&set(s['evaluation']);assert s['geometry_frames']==s['fusion'];cms={g:{} for g in GROUPS};packed={g:[] for g in GROUPS};smallids=set();pixelcount={}
  for f in s['evaluation']:
   for k in ['gt_labels','gt_instances']:assert sha(s['records'][f]['files'][k]['path'])==s['records'][f]['files'][k]['sha256']
   gt=np.load(d/'gt_labels'/f'{f:06d}.npy');inst=np.load(d/'gt_instances'/f'{f:06d}.npy');valid=gt!=255;vehicle=np.isin(gt,[13,14]);small=np.zeros_like(valid)
   for identity,count in zip(*np.unique(inst[inst>0],return_counts=True)):
    if identity%1000>0 and count<=.0025*gt.size and np.any((inst==identity)&vehicle):small|=inst==identity;smallids.add(int(identity))
   boundary=np.zeros_like(valid);kernel=np.ones((7,7),np.uint8)
   for c in range(19):m=(gt==c).astype(np.uint8);boundary|=cv2.dilate(m,kernel)!=cv2.erode(m,kernel)
   masks={'all':valid,'small_vehicle':valid&vehicle&small,'non_small_vehicle':valid&vehicle&~small,'boundary':valid&boundary,'interior':valid&~boundary,'pedestrian':valid&(gt==11)}
   for k,m in masks.items():masklocks.append({'sequence':name,'frame':f,'subgroup':k,'GT_pixels':int(m.sum()),'sha256':hashlib.sha256(m.tobytes()).hexdigest()});pixelcount[k]=pixelcount.get(k,0)+int(m.sum())
   alpha_reference=None
   for g in GROUPS:
    with np.load(run/'renders'/g/f'{f:06d}.npz') as p:pred=p['label'];score=p['score'];alpha=p['alpha']
    assert pred.shape==gt.shape and np.isfinite(score).all() and np.array_equal(pred==-2,alpha<.05)
    if alpha_reference is None:alpha_reference=alpha.copy()
    else:assert np.array_equal(alpha,alpha_reference)
    for k,m in masks.items():cms[g][k]=cms[g].get(k,np.zeros((19,21),np.int64))+common.confusion(gt,pred,m)
    accept=valid&(pred>=0);packed[g].append((score[accept],pred[accept]!=gt[accept],int(valid.sum())))
  support.append({'sequence':name,'distinct_small_vehicle_instances':len(smallids),'small_instance_ids':sorted(smallids),'GT_pixel_counts':pixelcount,'stable_dynamic':'NOT_EVALUABLE: no independent dynamic state/GT depth correspondence used'})
  curves={};maximum={}
  for g in GROUPS:
   cm=cms[g]['all'];matrices[name+'_'+g]=cm;mainrows.append({'sequence':name,'group':g,**met(cm)});secondary.append({'sequence':name,'group':g,**met(cm,list(range(19)))})
   for k,c in cms[g].items():subrows.append({'sequence':name,'group':g,'subgroup':k,**met(c)})
   score=np.concatenate([p[0] for p in packed[g]]);err=np.concatenate([p[1] for p in packed[g]]);den=sum(p[2] for p in packed[g]);order=np.argsort(-score,kind='stable');score=score[order];err=err[order];ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1] if len(score) else np.array([],int);accepted=ends+1;errors=np.cumsum(err,dtype=np.int64)[ends];coverage=accepted/den;risk=errors/accepted
   np.savez_compressed(out/f'{name}_{g}_risk_curve.npz',threshold=score[ends],accepted=accepted,errors=errors,coverage=coverage,risk=risk,GT_valid_denominator=np.array(den));curves[g]=(coverage,risk);maximum[g]=len(score)/den
  maximum_common=min(maximum.values());levels=[float(v) for v in np.arange(.05,1,.05) if v<=maximum_common]+[maximum_common]
  for g,(cover,risk) in curves.items():
   for target in levels:
    i=np.searchsorted(cover,target,side='right')-1;riskrows.append({'sequence':name,'group':g,'requested_common_coverage':target,'achieved_coverage':float(cover[i]) if i>=0 else 0.,'risk':float(risk[i]) if i>=0 else None,'common_max_deployable':maximum_common})
 dump(out/'GT_main_results.json',mainrows);dump(out/'GT_secondary_City19_results.json',secondary);dump(out/'GT_subgroup_results.json',subrows);dump(out/'GT_common_risk_coverage.json',riskrows);dump(out/'GT_support_and_unevaluable_groups.json',support);dump(out/'common_mask_locks.json',masklocks);np.savez_compressed(out/'GT_confusion_matrices.npz',**matrices)
 scalar=[{k:v for k,v in r.items() if k!='classes'} for r in mainrows]
 with (out/'GT_main_results.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(scalar[0]));w.writeheader();w.writerows(scalar)
 summary=[];checks=[];gate=frozen['candidate_claim_gates']
 for g in GROUPS:
  summary.append({'group':g,'equal_drive_primary_mIoU':float(np.mean([r['mIoU_GT_supported_common_classes'] for r in mainrows if r['group']==g])),'equal_drive_coverage':float(np.mean([r['coverage'] for r in mainrows if r['group']==g]))})
  if g=='S0':continue
  delta=[];fail=[];missing=[]
  for s in man['sequences']:
   name=s['name'];get=lambda method,key:next(r for r in mainrows if r['sequence']==name and r['group']==method)[key];diff=get(g,'mIoU_GT_supported_common_classes')-get('S0','mIoU_GT_supported_common_classes');delta.append(diff)
   if diff<-.02:fail.append(name+': primary loss>2pp')
   if get(g,'coverage')-get('S0','coverage')<-.02:fail.append(name+': coverage loss>2pp')
   for subgroup in ['small_vehicle','non_small_vehicle','boundary','interior']:
    sub=lambda method:next(r for r in subrows if r['sequence']==name and r['group']==method and r['subgroup']==subgroup);ref=sub('S0');candidate=sub(g)
    if ref['GT_valid_pixels']<100:missing.append(name+': '+subgroup+' <100 GT pixels')
    elif candidate['mIoU_GT_supported_common_classes']-ref['mIoU_GT_supported_common_classes']<-.02:fail.append(name+': '+subgroup+' loss>2pp')
   rows=[r for r in mainrows if r['sequence']==name and r['group'] in ['S0',g]];car={r['group']:next(c for c in r['classes'] if c['class']==13) for r in rows}
   if car['S0']['GT_pixels']<1000:missing.append(name+': car <1000 GT pixels')
   elif car[g]['IoU']-car['S0']['IoU']<-.02:fail.append(name+': carIoU loss>2pp')
   if next(x for x in support if x['sequence']==name)['distinct_small_vehicle_instances']<2:missing.append(name+': fewer than2 small vehicle instances')
  mean=float(np.mean(delta))
  if mean<.01:fail.append('equal-drive primary gain <1pp')
  checks.append({'group':g,'equal_drive_mIoU_difference':mean,'failed_gates':fail,'missing_required_support':missing,'confirmation_status':'FAIL' if fail else 'INCONCLUSIVE' if missing else 'CONFIRMED_FIXED_COMPARISON_ONLY','DEV_selected_reference_unchanged':'S0','post_holdout_selection':False})
 dump(out/'fixed_comparison_checks.json',checks);dump(out/'equal_drive_descriptive_summary.json',summary)
 dump(out/'result.json',{'status':'COMPLETE_PROJECT_ISOLATED_SMALL_WINDOW_EVALUATION','drives':2,'views':10,'methods':5,'GT_render_scores_complete':True,'reference':'S0 from DEV','final_quality_winner':None,'post_holdout_selection_or_tuning':False,'no_Oracle':True,'official_hidden_benchmark_Test':False,'stable_dynamic_evaluation':'NOT_EVALUABLE','global_scientific_PASS':False,'scope':'conditional fixed official-pose mapping on2 small project-new public train/val windows; not original KITTI08/Stage2 quality acceptance'})
 print(json.dumps({'status':'COMPLETE_PROJECT_HOLDOUT','summary':summary,'comparisons':checks}),flush=True)
if __name__=='__main__':main()
