"""Bounded archive extraction, full hashes and separate Boolean GT recount."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];B=R/'03_方法冻结与隔离评价/kitti360_isolated_v1';A=B.parent/'kitti360_isolated_v1_complete_20261007.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 receipt=json.loads((B/'package_result.json').read_text(encoding='utf-8'));assert A.stat().st_size==receipt['bytes'] and sha(A)==receipt['sha256'];allowed=['inputs/kitti360_isolated_v1/','03_方法冻结与隔离评价/kitti360_isolated_v1/','scripts/','protocol/KITTI360隔离评价预声明_20261007_v1.json']
 with tarfile.open(A) as t:
  members=t.getmembers();assert len(members)==receipt['files']
  for m in members:
   target=(R/m.name).resolve();assert target.is_relative_to(R.resolve()) and m.isfile() and any(m.name.startswith(v) for v in allowed)
   if target.exists():assert sha(target)==hashlib.sha256(t.extractfile(m).read()).hexdigest(),str(target)
   else:t.extract(m,R,filter='data')
 manifest=json.loads((B/'artifact_SHA256.json').read_text(encoding='utf-8'));assert all(sha(R/p)==h for p,h in manifest.items());D=R/'inputs/kitti360_isolated_v1';man=json.loads((D/'manifest.json').read_text(encoding='utf-8'));E=B/'GT_final_evaluation';rows=json.loads((E/'GT_main_results.json').read_text(encoding='utf-8'));secondary=json.loads((E/'GT_secondary_City19_results.json').read_text(encoding='utf-8'));checks=[]
 for s in man['sequences']:
  d=D/s['name'];run=B/s['name'];support={};poses=np.load(d/'c2w.npy')
  with np.load(d/'base_map.npz') as p:assert np.allclose(p['gt_w2c_all_frames'],np.linalg.inv(poses),atol=1e-4)
  assert len(list((d/'image_2').glob('*.png')))==21 and len(list((d/'gt_labels').glob('*.npy')))==5
  for g in ['S0','S1','S-G','S2','S2-H']:
   n=known=unknown=nomap=correct=0;TP=np.zeros(19,np.int64);GS=TP.copy();PS=TP.copy();scores=[];errors=[]
   for f in s['evaluation']:
    gt=np.load(d/'gt_labels'/f'{f:06d}.npy');valid=gt!=255
    with np.load(run/'renders'/g/f'{f:06d}.npz') as p:pred=p['label'];alpha=p['alpha'];score=p['score']
    if f not in support:support[f]=alpha
    else:assert np.array_equal(support[f],alpha)
    n+=int(valid.sum());known+=int(((pred>=0)&valid).sum());unknown+=int(((pred==-1)&valid).sum());nomap+=int(((pred==-2)&valid).sum());correct+=int(((pred==gt)&valid).sum())
    for c in range(19):TP[c]+=int(((pred==c)&(gt==c)&valid).sum());GS[c]+=int(((gt==c)&valid).sum());PS[c]+=int(((pred==c)&valid).sum())
    accept=valid&(pred>=0);scores.append(score[accept]);errors.append((pred!=gt)[accept])
   row=next(r for r in rows if r['sequence']==s['name'] and r['group']==g);second=next(r for r in secondary if r['sequence']==s['name'] and r['group']==g)
   assert row['GT_valid_pixels']==n and row['labelled_pixels']==known and row['unknown_pixels']==unknown and row['no_map_pixels']==nomap;assert abs(row['coverage']-known/n)<1e-12 and abs(row['conditional_accuracy']-correct/known)<1e-12
   for stored,classes in [(row,[0,2,5,6,7,8,9,10,13,14]),(second,list(range(19)))]:
    values=[TP[c]/(GS[c]+PS[c]-TP[c]) for c in classes if GS[c]>0];assert abs(stored['mIoU_GT_supported_common_classes']-float(np.mean(values)))<1e-12
    for entry in stored['classes']:c=entry['class'];assert entry['TP']==TP[c] and entry['FP']==PS[c]-TP[c] and entry['FN_includes_unknown_and_no_map']==GS[c]-TP[c]
   score=np.concatenate(scores);error=np.concatenate(errors);order=np.argsort(-score,kind='stable');score=score[order];error=error[order];ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1];accepted=ends+1;mistakes=np.cumsum(error,dtype=np.int64)[ends]
   with np.load(E/f"{s['name']}_{g}_risk_curve.npz") as curve:
    assert np.array_equal(curve['threshold'],score[ends]) and np.array_equal(curve['accepted'],accepted) and np.array_equal(curve['errors'],mistakes);assert np.array_equal(curve['coverage'],accepted/n) and np.array_equal(curve['risk'],mistakes/accepted)
   checks.append({'sequence':s['name'],'group':g,'Boolean_counts_exact':True,'primary_and_City19_mIoU':True,'risk_ties_and_errors_exact':True})
 taxroot=B.parent/'kitti360_taxonomy_audit_v1';tax=json.loads((taxroot/'taxonomy_validation.json').read_text(encoding='utf-8'));assert tax['GT_metrics_or_masks_changed']==False
 for s in man['sequences']:
  for f in s['evaluation']:
   folder=taxroot/s['name'];p=folder/f'{f:06d}_official_semantic.png';pin=folder/f'{f:06d}_official_instance.png';rec=next(v for v in tax['checks'] if v['sequence']==s['name'] and v['local_frame']==f);assert sha(p)==rec['raw_GT_semantic_sha256'] and sha(pin)==rec['raw_instance_sha256'];sem=np.asarray(Image.open(p));gt=np.full(sem.shape,255,np.uint8)
   for label in tax['official_label_definitions']:
    if not label['ignoreInEval']:gt[sem==label['semanticID']]=label['trainID']
   assert np.array_equal(gt,np.load(D/s['name']/'gt_labels'/f'{f:06d}.npy')) and np.array_equal(np.asarray(Image.open(pin)),np.load(D/s['name']/'gt_instances'/f'{f:06d}.npy'))
 result={'status':'PASS_LOCAL_FULL_HASH_AND_INDEPENDENT_GT_AND_RISK_RECOUNT','artifact_files':len(manifest),'archive_files':len(members),'archive_sha256':receipt['sha256'],'renders':50,'pairs':10,'GT_and_risk_recount':checks,'GPU_used':False,'metric_tolerance':1e-12,'evaluation_RGB_depth_excluded':True,'official_raw_GT_mapping_recount':True,'global_scientific_PASS':False};(B/'local_integrity_and_independent_recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='GT_and_risk_recount'}))
if __name__=='__main__':main()
