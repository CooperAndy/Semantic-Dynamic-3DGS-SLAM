"""Frozen DEV paired evaluation, complete ties and shared GT subgroups."""
import json,hashlib
from pathlib import Path
import numpy as np
import cv2
from evaluate_vkitti_gt_dev import confusion,metrics,temporal_masks,CLASSES
R=Path(__file__).resolve().parents[1];O=R/'05_条件轻量候选与历史诊断/dev_diagnostics_20261007_v2'
BASE=['S0','S1','S-G','S2','S2-H']
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
 assert not (O/'GT_results.json').exists();man=json.loads((R/'inputs/vkitti2_gt_dev_v2/manifest.json').read_text());rows=[];subs=[];riskrows=[];masklocks=[];cms={};deltas=[]
 for seq in man['sequences']:
  name=seq['name'];d=R/'inputs/vkitti2_gt_dev_v2'/name;old=R/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{name}_full';folder=O/name
  new=sorted(p.name for p in (folder/'renders').iterdir() if p.is_dir());groups=BASE+new;mat={g:{} for g in groups};packs={g:[] for g in groups if g=='S2' or g in ['S2-V','S2-Z']}
  for f in seq['evaluation']:
   gt=np.load(d/'gt_labels'/f'{f:06d}.npy');inst=np.load(d/'gt_instances'/f'{f:06d}.npy');moving=np.load(d/'gt_moving'/f'{f:06d}.npy');valid=gt!=255;vehicle=np.isin(gt,[13,14]);small=np.zeros_like(valid)
   for identity,count in zip(*np.unique(inst[inst>0],return_counts=True)):
    if count<=.0025*gt.size:small|=inst==identity
   boundary=np.zeros_like(valid);kernel=np.ones((7,7),np.uint8)
   for c in CLASSES:m=(gt==c).astype(np.uint8);boundary|=cv2.dilate(m,kernel)!=cv2.erode(m,kernel)
   stable,changed=temporal_masks(d,seq,f)
   masks={'all':valid,'small_vehicle':valid&vehicle&small,'non_small_vehicle':valid&vehicle&~small,'boundary':valid&boundary,'interior':valid&~boundary,'GT_moving_vehicle':valid&vehicle&moving,'GT_nonmoving_vehicle':valid&vehicle&~moving,'temporal_stable_proxy':valid&stable,'temporal_change_proxy':valid&changed,'temporal_unclassified':valid&~stable&~changed}
   oldmasks=json.loads((R/'02_基础因子DEV与Oracle诊断/gt_dev_v2/GT_evaluation_v1'/f'{name}_common_subgroup_masks.json').read_text())
   for key,m in masks.items():h=hashlib.sha256(m.tobytes()).hexdigest();assert h==next(r['sha256'] for r in oldmasks if r['frame']==f and r['subgroup']==key);masklocks.append({'sequence':name,'frame':f,'subgroup':key,'sha256':h})
   with np.load(old/'renders/S2'/f'{f:06d}.npz') as b:alpha=b['alpha']
   for g in groups:
    p=(old if g in BASE else folder)/'renders'/g/f'{f:06d}.npz'
    with np.load(p) as a:pred=a['label'];score=a['score'];assert np.array_equal(a['alpha'],alpha)
    for key,m in masks.items():mat[g][key]=mat[g].get(key,np.zeros((19,21),np.int64))+confusion(gt,pred,m)
    if g in packs:take=valid&(pred>=0);packs[g].append((score[take],pred[take]!=gt[take],int(valid.sum())))
  oldrows=json.loads((R/'02_基础因子DEV与Oracle诊断/gt_dev_v2/GT_evaluation_v1/GT_main_results.json').read_text())
  for g in groups:
   row={'sequence':name,'group':g,**metrics(mat[g]['all'])};rows.append(row);cms[name+'_'+g]=mat[g]['all']
   if g in BASE:assert row['mIoU_GT_supported_common_classes']==next(r['mIoU_GT_supported_common_classes'] for r in oldrows if r['sequence']==name and r['group']==g)
   for k,v in mat[g].items():subs.append({'sequence':name,'group':g,'subgroup':k,**metrics(v)})
  curves={}
  for g,pack in packs.items():
   scores=np.concatenate([x[0] for x in pack]);errors=np.concatenate([x[1] for x in pack]);den=sum(x[2] for x in pack);order=np.argsort(-scores,kind='stable');scores=scores[order];errors=errors[order];ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1];accepted=ends+1;mistakes=np.cumsum(errors,dtype=np.int64)[ends];cover=accepted/den;risk=mistakes/accepted;curves[g]=(cover,risk)
   np.savez_compressed(folder/f'{g}_risk_curve.npz',threshold=scores[ends],accepted=accepted,errors=mistakes,coverage=cover,risk=risk,GT_valid_denominator=np.array(den))
  for g in ['S2-V','S2-Z']:
   if g not in groups:continue
   main=next(r for r in rows if r['sequence']==name and r['group']==g);baseline=next(r for r in rows if r['sequence']==name and r['group']=='S2');deltas.append({'sequence':name,'candidate':g,'baseline':'S2','mIoU_delta':main['mIoU_GT_supported_common_classes']-baseline['mIoU_GT_supported_common_classes'],'coverage_delta':main['coverage']-baseline['coverage']})
   common=min(curves[g][0][-1],curves['S2'][0][-1]);levels=[v for v in np.arange(.05,1,.05) if v<=common]+[common]
   for method in ['S2',g]:
    cover,risk=curves[method]
    for target in levels:i=np.searchsorted(cover,target,side='right')-1;riskrows.append({'sequence':name,'pair':g,'method':method,'target_coverage':float(target),'achieved_coverage':float(cover[i]) if i>=0 else 0.,'risk':float(risk[i]) if i>=0 else None,'whole_ties_only':True})
 policy=json.loads((O/'quality_policy.json').read_text());decision=[]
 for g in ['S2-V','S2-Z']:
  dr=[r for r in deltas if r['candidate']==g]
  if len(dr)!=2:decision.append({'candidate':g,'status':'NOT_EVALUABLE_TWO_WINDOW_GATE','triggered_windows':len(dr)});continue
  failures=[];mean=float(np.mean([r['mIoU_delta'] for r in dr]))
  if mean<.01:failures.append('equal_scene_mean_gain_below_1pp')
  if any(r['mIoU_delta']<-.02 for r in dr):failures.append('single_scene_mIoU_loss_over_2pp')
  if any(r['coverage_delta']<-.02 for r in dr):failures.append('single_scene_coverage_loss_over_2pp')
  for seq in man['sequences']:
   for key in ['small_vehicle','non_small_vehicle','boundary','interior']:
    b=next(r for r in subs if r['sequence']==seq['name'] and r['group']=='S2' and r['subgroup']==key);c=next(r for r in subs if r['sequence']==seq['name'] and r['group']==g and r['subgroup']==key)
    if c['GT_valid_pixels']<100:failures.append(seq['name']+':'+key+':insufficient_GT')
    elif c['mIoU_GT_supported_common_classes']-b['mIoU_GT_supported_common_classes']<-.02:failures.append(seq['name']+':'+key+':loss_over_2pp')
  decision.append({'candidate':g,'equal_scene_mIoU_delta_vs_S2':mean,'status':'DEV_GATE_PASS_REQUIRES_NEW_ISOLATED_ENTRY' if not failures else 'DEV_GATE_FAIL','failures':failures,'formal_acceptance':False})
 np.savez_compressed(O/'GT_confusion_matrices.npz',**cms);dump(O/'GT_results.json',rows);dump(O/'GT_subgroups.json',subs);dump(O/'GT_masks_identity.json',masklocks);dump(O/'risk_coverage_paired.json',riskrows);dump(O/'candidate_deltas.json',deltas);dump(O/'quality_decision.json',decision);dump(O/'evaluation_result.json',{'status':'COMPLETE_05_DEV_GT_EVALUATION','old_five_metrics_exactly_reproduced':True,'all_subgroup_masks_old_hash_identical':True,'shared_native_alpha_identical':True,'main_unknown_no_map_retained':True,'candidate_decisions':decision,'final_entry_used':False,'default_S0_unchanged':True})
 print(json.dumps(decision),flush=True)
if __name__=='__main__':run()
