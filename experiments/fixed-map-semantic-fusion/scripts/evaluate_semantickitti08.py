"""All200 sparse independentGT evidence, fixed states, no method selection."""
import json,hashlib,csv
from pathlib import Path
import numpy as np
import evaluate_vkitti_gt_dev as common
from semantickitti_projection import PRIMARY,MOVING_IDS
R=Path('/root/autodl-tmp/stage3_chy_20261006');D=R/'inputs/semantickitti08_v1';B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1';GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def metrics(cm):
 old=common.CLASSES;common.CLASSES=PRIMARY
 try:return common.metrics(cm)
 finally:common.CLASSES=old
def main():
 man=json.loads((D/'evaluation_input_manifest.json').read_text());protocol=R/'protocol/SemanticKITTI08稀疏GT评价预声明_20261007_v1.json';assert sha(protocol)==man['protocol_sha256'];out=B/'GT_evaluation';assert not out.exists();out.mkdir();dump(out/'protocol.json',json.loads(protocol.read_text()));cms={g:{} for g in GROUPS+['2D_input_teacher']};packed={g:[] for g in GROUPS};frame_rows=[];masklocks=[];diag=[];supports=[];tinyids=set()
 for rec in man['records']:
  f=rec['original_frame'];local=rec['local_frame'];assert sha(rec['GT_file']['path'])==rec['GT_file']['sha256']
  with np.load(rec['GT_file']['path']) as gt:
   pi=np.flatnonzero(gt['point_index'].ravel()>=0);label=gt['label'].ravel()[pi];raw=gt['raw_semantic'].ravel()[pi];inst=gt['instance'].ravel()[pi];depth=gt['depth'].ravel()[pi]
  valid=label!=255;teacher=np.load(rec['input_locks']['labels']['path']).ravel()[pi];assert ((teacher>=0)&(teacher<19)).all();vehicle=np.isin(label,[13,14]);tiny=np.zeros_like(valid);width=man['image_width'];height=man['image_height'];y=pi//width;x=pi%width
  for identity in np.unique(inst[vehicle&(inst>0)]):
   v=vehicle&(inst==identity);count=int(v.sum());area=int((x[v].max()-x[v].min()+1)*(y[v].max()-y[v].min()+1))
   if count>=5 and area<=.0025*width*height:tiny|=v;tinyids.add(int(identity))
  moving=np.isin(raw,MOVING_IDS);shared=None;mapdepth=None
  for g in GROUPS:
   with np.load(B/'renders'/g/f'{f:06d}.npz') as p:assert np.array_equal(pi,p['pixel_index']);pred=p['label'];score=p['score'];alpha=p['alpha'];md=p['map_depth']
   if shared is None:shared=alpha;mapdepth=md
   else:assert np.array_equal(shared,alpha) and np.array_equal(md,mapdepth)
   assert np.array_equal(pred==-2,alpha<.05);agree_depth=(alpha>=.05)&np.isfinite(md)&(md>0)&(np.abs(md-depth)<=1+.02*depth)
   masks={'all200':valid,'fusion180':valid&(local%10!=9),'excluded20':valid&(local%10==9),'GT_moving':valid&moving,'GT_nonmoving':valid&~moving,'small_vehicle_visible_bbox_proxy':valid&vehicle&tiny,'other_vehicle_or_insufficient_bbox':valid&vehicle&~tiny,'map_GT_depth_agrees_proxy':valid&agree_depth,'map_GT_depth_disagrees_proxy':valid&(alpha>=.05)&~agree_depth,'no_map_support':valid&(alpha<.05)}
   for k,m in masks.items():cms[g][k]=cms[g].get(k,np.zeros((19,21),np.int64))+common.confusion(label,pred,m)
   cm=common.confusion(label,pred,valid);frame_rows.append({'original_frame':f,'local_frame':local,'group':g,'role':'fusion180' if local%10!=9 else 'excluded20','GT_pixel_fraction':int(valid.sum())/(height*width),**metrics(cm)})
   accept=valid&(pred>=0);packed[g].append((score[accept],pred[accept]!=label[accept],int(valid.sum())))
   tcorrect=teacher==label;mcorrect=pred==label;known=pred>=0
   diag.append({'original_frame':f,'group':g,'valid_GT_pixels':int(valid.sum()),'both_correct':int((valid&tcorrect&mcorrect).sum()),'teacher_wrong_map_correct':int((valid&~tcorrect&mcorrect).sum()),'teacher_correct_map_wrong_known':int((valid&tcorrect&known&~mcorrect).sum()),'both_wrong_map_known':int((valid&~tcorrect&known&~mcorrect).sum()),'teacher_correct_map_unknown':int((valid&tcorrect&(pred==-1)).sum()),'teacher_correct_map_no_map':int((valid&tcorrect&(pred==-2)).sum()),'map_unknown':int((valid&(pred==-1)).sum()),'map_no_map':int((valid&(pred==-2)).sum()),'depth_disagrees_known_wrong':int((valid&known&~mcorrect&~agree_depth).sum()),'depth_agrees_known_wrong':int((valid&known&~mcorrect&agree_depth).sum())})
   if g=='S0':
    for k,m in masks.items():cms['2D_input_teacher'][k]=cms['2D_input_teacher'].get(k,np.zeros((19,21),np.int64))+common.confusion(label,teacher,m);masklocks.append({'frame':f,'subgroup':k,'pixels':int(m.sum()),'sha256':hashlib.sha256(m.tobytes()).hexdigest()})
  supports.append({'original_frame':f,'raw_points':rec['points'],'in_image_points':rec['in_image_points'],'unique_projected_pixels':len(pi),'mapped_GT_pixels':int(valid.sum()),'mapped_GT_image_fraction':int(valid.sum())/(height*width),'moving_GT_pixels':int((valid&moving).sum()),'raw_semantic_counts':{str(i):int(n) for i,n in zip(*np.unique(raw,return_counts=True))}})
 rows=[{'group':g,'subgroup':k,**metrics(cm)} for g,sub in cms.items() for k,cm in sub.items()];dump(out/'GT_results.json',rows);dump(out/'GT_per_frame.json',frame_rows);dump(out/'input_vs_map_diagnostics.json',diag);dump(out/'GT_support.json',supports);dump(out/'shared_subgroup_mask_locks.json',masklocks);np.savez_compressed(out/'GT_confusion_matrices.npz',**{g+'__'+k:cm for g,sub in cms.items() for k,cm in sub.items()});curves={};maxima={}
 for g in GROUPS:
  score=np.concatenate([a[0] for a in packed[g]]);error=np.concatenate([a[1] for a in packed[g]]);den=sum(a[2] for a in packed[g]);order=np.argsort(-score,kind='stable');score=score[order];error=error[order];ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1];accepted=ends+1;errors=np.cumsum(error,dtype=np.int64)[ends];coverage=accepted/den;risk=errors/accepted;np.savez_compressed(out/(g+'_risk_curve.npz'),threshold=score[ends],accepted=accepted,errors=errors,coverage=coverage,risk=risk,GT_valid_denominator=np.array(den));curves[g]=(coverage,risk);maxima[g]=len(score)/den
 levels=[float(v) for v in np.arange(.05,1,.05) if v<=min(maxima.values())]+[min(maxima.values())];riskrows=[]
 for g,(cov,risk) in curves.items():
  for target in levels:i=np.searchsorted(cov,target,side='right')-1;riskrows.append({'group':g,'requested_common_coverage':target,'achieved_coverage':float(cov[i]) if i>=0 else 0.,'risk':float(risk[i]) if i>=0 else None})
 dump(out/'common_risk_coverage.json',riskrows);dump(out/'result.json',{'status':'COMPLETE_REAL_SPARSE_GT_DEV_ALL200','frames':200,'methods':5,'renders':1000,'GT_source':'official SemanticKITTI raw28 labels projected using KITTI P2/Tr','GT_in_fusion':False,'dense_image_GT':False,'Gaussian3D_accuracy':False,'historically_exposed_DEV':True,'new_Test':False,'quality_winner':None,'scientific_PASS':False,'small_vehicle_proxy_instances':len(tinyids),'moving_nonmoving_subgroups':'official explicitmovingrawlabels; nonmovingnottemporalstabilityGT','no_Oracle_or_retraining':True});print(json.dumps({'status':'COMPLETE_SPARSE_GT_ALL200','main':[{k:v for k,v in r.items() if k!='classes'} for r in rows if r['subgroup']=='all200']}),flush=True)
if __name__=='__main__':main()
