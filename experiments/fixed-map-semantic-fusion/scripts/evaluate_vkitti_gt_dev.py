"""GT-only evaluator: common masks, unknown misses, exact tied risk ranking."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import cv2
GROUPS=['S0','S1','S-G','S2','S2-H','O-Sem']
CLASSES=[0,2,5,6,7,8,9,10,13,14]
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def confusion(gt,pred,mask):
 assert ((pred>=-2)&(pred<19)).all()
 return np.bincount(gt[mask].astype(np.int64)*21+pred[mask].astype(np.int64)+2,minlength=19*21).reshape(19,21)
def metrics(cm):
 n=int(cm.sum());known=int(cm[:,2:].sum());agree=int(sum(cm[c,c+2] for c in range(19)));rows=[]
 for c in CLASSES:
  tp=int(cm[c,c+2]);gs=int(cm[c].sum());ps=int(cm[:,c+2].sum());u=gs+ps-tp
  rows.append({'class':c,'GT_pixels':gs,'TP':tp,'FP':ps-tp,'FN_includes_unknown_and_no_map':gs-tp,'IoU':tp/u if gs else None,'status':'EVALUABLE' if gs else 'NOT_EVALUABLE'})
 ious=[r['IoU'] for r in rows if r['IoU'] is not None]
 return {'GT_valid_pixels':n,'labelled_pixels':known,'coverage':known/n if n else None,'conditional_accuracy':agree/known if known else None,'mIoU_GT_supported_common_classes':float(np.mean(ious)) if ious else None,'unknown_pixels':int(cm[:,1].sum()),'no_map_pixels':int(cm[:,0].sum()),'unknown_fraction':int(cm[:,1].sum())/n if n else None,'no_map_fraction':int(cm[:,0].sum())/n if n else None,'classes':rows}
def temporal_masks(d,seq,f):
 gt=np.load(d/'gt_labels'/f'{f:06d}.npy');depth=np.load(d/'gt_depth'/f'{f:06d}.npy');moving=np.load(d/'gt_moving'/f'{f:06d}.npy')
 pg=np.load(d/'gt_labels'/f'{f-1:06d}.npy');pd=np.load(d/'gt_depth'/f'{f-1:06d}.npy');pm=np.load(d/'gt_moving'/f'{f-1:06d}.npy')
 poses=np.load(d/'c2w.npy');K=np.asarray(seq['K']);h,w=gt.shape;y,x=np.indices(gt.shape);z=depth
 xyz=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z,np.ones_like(z)],-1);q=xyz@(np.linalg.inv(poses[f-1])@poses[f]).T
 zp=q[...,2];u=np.rint(K[0,0]*q[...,0]/np.maximum(zp,1e-8)+K[0,2]).astype(int);v=np.rint(K[1,1]*q[...,1]/np.maximum(zp,1e-8)+K[1,2]).astype(int)
 inside=(zp>.5)&(z>=.5)&(z<=80)&(u>=0)&(u<w)&(v>=0)&(v<h);u=np.clip(u,0,w-1);v=np.clip(v,0,h-1)
 common=inside&(pd[v,u]>=.5)&(pd[v,u]<=80)&(gt!=255)&(pg[v,u]!=255)
 agree_depth=np.abs(pd[v,u]-zp)<(.1+.01*zp);agree_class=pg[v,u]==gt
 stable=common&agree_depth&agree_class&~moving&~pm[v,u]
 changed=common&(~agree_depth|~agree_class|moving|pm[v,u])
 return stable,changed
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
 source=root/'inputs/vkitti2_gt_dev_v2/manifest.json';man=json.loads(source.read_text());base=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2';out=base/'GT_evaluation_v1'
 if out.exists():raise FileExistsError('preserve prior evaluation')
 out.mkdir();allrows=[];subrows=[];curves={};matrices={};supportrows=[]
 protocol={'status':'DEFINED_BEFORE_FIRST_GT_METRIC_READ','main':'map-rendered 2D semantic mIoU on GT-supported common classes; full valid-GT denominator','GT_classes':CLASSES,'ignore':'only frozen official class mapping 255','unknown':'FN for GT true class; never excluded from main','no_map':'FN for GT true class in main; separate coverage reported','risk_score':'max final class mass / independent native total alpha, uncalibrated','ties':'all equal-score samples accepted together, no GT tie-break','coverage':'accepted labelled valid-GT pixels / all valid-GT pixels; no-map/unknown not eligible','risk':'0-1 semantic error among accepted','subgroups':'GT vehicle instance area <=0.0025 image; boundary morphology radius3; temporal previous GT reprojection tolerance .1m+.01z','temporal_note':'stable and depth/class/motion-change proxies, unclassified explicitly retained in whole-image denominator; no claim perfect dynamic segmentation','all_subgroup_masks_shared':True,'O-Sem':'DEV-only diagnostic outside deployable ranking','main_quality_improvement_threshold':None,'coverage_loss_budget':None,'critical_subgroup_thresholds':None,'final_split':[],'no_final_PASS':True,'independent_samples':'two underlying scenes; views within scene correlated, no pixel-level CI'}
 dump(out/'GT_evaluation_protocol.json',protocol)
 for seq in man['sequences']:
  d=Path(seq['path']);name=seq['name'];run=base/(name+'_full');cfg=json.loads((run/'config.json').read_text());evals=cfg['evaluation_frames'];assert not set(evals)&set(cfg['fusion_frames'])
  cms={g:{} for g in GROUPS};packed={g:[] for g in GROUPS};support_reference={};maskhashes=[]
  for f in evals:
   record=seq['records'][f]
   for kind in ['gt_labels','gt_instances','gt_moving','gt_depth']:assert sha(record['files'][kind]['path'])==record['files'][kind]['sha256']
   gt=np.load(d/'gt_labels'/f'{f:06d}.npy');inst=np.load(d/'gt_instances'/f'{f:06d}.npy');moving=np.load(d/'gt_moving'/f'{f:06d}.npy');valid=gt!=255
   vehicle=np.isin(gt,[13,14]);small=np.zeros_like(valid)
   for identity,count in zip(*np.unique(inst[inst>0],return_counts=True)):
    if count<=.0025*gt.size:small|=inst==identity
   kernel=np.ones((7,7),np.uint8);boundary=np.zeros_like(valid)
   for c in CLASSES:
    m=(gt==c).astype(np.uint8);boundary|=(cv2.dilate(m,kernel)!=cv2.erode(m,kernel))
   stable,changed=temporal_masks(d,seq,f)
   masks={'all':valid,'small_vehicle':valid&vehicle&small,'non_small_vehicle':valid&vehicle&~small,'boundary':valid&boundary,'interior':valid&~boundary,'GT_moving_vehicle':valid&vehicle&moving,'GT_nonmoving_vehicle':valid&vehicle&~moving,'temporal_stable_proxy':valid&stable,'temporal_change_proxy':valid&changed,'temporal_unclassified':valid&~stable&~changed}
   for key,mask in masks.items():maskhashes.append({'frame':f,'subgroup':key,'sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'GT_pixels':int(mask.sum())})
   for g in GROUPS:
    with np.load(run/'renders'/g/f'{f:06d}.npz') as p:pred=p['label'];score=p['score'];alpha=p['alpha']
    assert np.isfinite(score).all() and pred.shape==gt.shape
    supported=alpha>=.05;assert np.array_equal(pred==-2,~supported)
    if f not in support_reference:support_reference[f]=supported.copy()
    else:assert np.array_equal(supported,support_reference[f])
    for key,mask in masks.items():cms[g][key]=cms[g].get(key,np.zeros((19,21),np.int64))+confusion(gt,pred,mask)
    accepted=valid&(pred>=0);packed[g].append((score[accepted],pred[accepted]!=gt[accepted],int(valid.sum())))
  dump(out/(name+'_common_subgroup_masks.json'),maskhashes)
  max_cover={}
  for g in GROUPS:
   cm=cms[g]['all'];row={'sequence':name,'scene':seq['scene'],'group':g,**metrics(cm)};allrows.append(row)
   matrices[name+'_'+g]=cm
   for key,cm_sub in cms[g].items():subrows.append({'sequence':name,'group':g,'subgroup':key,**metrics(cm_sub)})
   scores=np.concatenate([p[0] for p in packed[g]]);errors=np.concatenate([p[1] for p in packed[g]]);den=sum(p[2] for p in packed[g]);order=np.argsort(-scores,kind='stable');scores=scores[order];errors=errors[order]
   ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1] if len(scores) else np.array([],int)
   accepted=ends+1;mistakes=np.cumsum(errors,dtype=np.int64)[ends];coverage=accepted/den;risk=mistakes/accepted
   np.savez_compressed(out/(name+'_'+g+'_risk_curve.npz'),threshold=scores[ends],accepted=accepted,errors=mistakes,coverage=coverage,risk=risk,GT_valid_denominator=np.array(den))
   curves[name,g]=(coverage,risk);max_cover[g]=len(scores)/den
  common_max=min(max_cover[g] for g in GROUPS[:-1]);levels=[v for v in np.arange(.05,1,.05) if v<=common_max]+[common_max]
  for g in GROUPS:
   coverage,risk=curves[name,g]
   for target in levels:
    idx=np.searchsorted(coverage,target,side='right')-1
    supportrows.append({'sequence':name,'group':g,'requested_common_coverage':float(target),'achieved_coverage':float(coverage[idx]) if idx>=0 else 0.,'risk':float(risk[idx]) if idx>=0 else None,'common_max_deployable':common_max,'tie_policy':'largest complete score group at or below target'})
 np.savez_compressed(out/'GT_confusion_matrices.npz',**matrices)
 dump(out/'GT_main_results.json',allrows);dump(out/'GT_subgroup_results.json',subrows);dump(out/'GT_common_risk_coverage.json',supportrows)
 scalar=[{k:v for k,v in r.items() if k!='classes'} for r in allrows]
 with (out/'GT_main_results.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(scalar[0]));w.writeheader();w.writerows(scalar)
 pooled=[]
 for g in GROUPS:pooled.append({'group':g,**metrics(sum((matrices[s['name']+'_'+g] for s in man['sequences']),np.zeros((19,21),np.int64)))})
 dump(out/'GT_pooled_descriptive.json',pooled)
 oracle=[]
 for seq in man['sequences']:
  s0=next(r for r in allrows if r['sequence']==seq['name'] and r['group']=='S0');o=next(r for r in allrows if r['sequence']==seq['name'] and r['group']=='O-Sem')
  oracle.append({'sequence':seq['name'],'S0_mIoU':s0['mIoU_GT_supported_common_classes'],'O_Sem_mIoU':o['mIoU_GT_supported_common_classes'],'difference':o['mIoU_GT_supported_common_classes']-s0['mIoU_GT_supported_common_classes'],'interpretation':'Input-label contribution diagnostic only; residual errors may involve geometry, occlusion, association, temporal inconsistency and taxonomy; not strict upper bound'})
 dump(out/'O_Sem_diagnostic.json',oracle)
 result={'status':'COMPLETE_INDEPENDENT_GT_EXPOSED_DEV_EVALUATION','scenes':2,'evaluation_views':10,'methods':5,'oracle_methods':1,'shared_support_and_subgroup_masks':True,'unknown_and_no_map_retained_in_main':True,'formal_final_test':'NOT_EXECUTED_NO_ISOLATED_ENTRY','quality_winner':None,'GT_source':'official classSegmentation, independently authored from model outputs','input_prediction_GT_leakage':False,'resources':'per-run result files','no_training_or_Stage2_reopen':True}
 dump(out/'result.json',result)
 dump(out/'SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
