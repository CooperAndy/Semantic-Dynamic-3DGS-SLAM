"""Download integrity and independent Boolean GT recount on the local copy."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];step=root/'02_基础因子DEV与Oracle诊断';archive=step/'vkitti_gt_dev_v2_complete_20261007.tar.gz'
receipt=json.loads(json.loads((root/'vkitti_package_completion_20261007.json').read_text(encoding='utf-8'))['stdout'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
allowed=['02_基础因子DEV与Oracle诊断/gt_dev_v2/','inputs/vkitti2_gt_dev_v2/','03_方法冻结与隔离评价/gt_dev_rules_freeze_20261007_v2/','scripts/','protocol/VKITTI2_GT_DEV判优预声明_20261007_v2.json']
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():
  target=(root/m.name).resolve();assert target.is_relative_to(root.resolve()) and m.isfile() and any(m.name.startswith(p) for p in allowed)
  if target.exists():
   assert target.read_bytes()==t.extractfile(m).read(),str(target)
  else:t.extract(m,root,filter='data')
freeze=root/'03_方法冻结与隔离评价/gt_dev_rules_freeze_20261007_v2';manifest=json.loads((freeze/'artifact_SHA256.json').read_text(encoding='utf-8'))
assert all(sha(root/p)==h for p,h in manifest.items())
data=root/'inputs/vkitti2_gt_dev_v2';man=json.loads((data/'manifest.json').read_text(encoding='utf-8'));ev=step/'gt_dev_v2/GT_evaluation_v1';rows=json.loads((ev/'GT_main_results.json').read_text(encoding='utf-8'));checks=[]
for seq in man['sequences']:
 d=data/seq['name'];run=step/'gt_dev_v2'/(seq['name']+'_full');support={}
 poses=np.load(d/'c2w.npy')
 with np.load(d/'base_map.npz') as p:assert np.allclose(p['gt_w2c_all_frames'],[np.linalg.inv(q)@poses[0] for q in poses],atol=1e-4)
 for g in ['S0','S1','S-G','S2','S2-H','O-Sem']:
  n=known=unknown=nomap=correct=0;TP=np.zeros(19,np.int64);GS=TP.copy();PS=TP.copy()
  for f in seq['evaluation']:
   gt=np.load(d/'gt_labels'/f'{f:06d}.npy');valid=gt!=255
   with np.load(run/'renders'/g/f'{f:06d}.npz') as p:prediction=p['label'];alpha=p['alpha']
   supported=alpha>=.05
   if f not in support:support[f]=supported
   else:assert np.array_equal(support[f],supported)
   n+=int(valid.sum());known+=int(((prediction>=0)&valid).sum());unknown+=int(((prediction==-1)&valid).sum());nomap+=int(((prediction==-2)&valid).sum());correct+=int(((prediction==gt)&valid).sum())
   for c in range(19):
    TP[c]+=int(((prediction==c)&(gt==c)&valid).sum());GS[c]+=int(((gt==c)&valid).sum());PS[c]+=int(((prediction==c)&valid).sum())
  old=next(r for r in rows if r['sequence']==seq['name'] and r['group']==g)
  ious=[TP[c]/(GS[c]+PS[c]-TP[c]) for c in man['evaluation_classes'] if GS[c]>0];miou=float(np.mean(ious))
  assert old['GT_valid_pixels']==n and old['labelled_pixels']==known and old['unknown_pixels']==unknown and old['no_map_pixels']==nomap
  assert abs(old['coverage']-known/n)<1e-12 and abs(old['conditional_accuracy']-correct/known)<1e-12 and abs(old['mIoU_GT_supported_common_classes']-miou)<1e-12
  for row in old['classes']:
   c=row['class'];assert row['TP']==TP[c] and row['FP']==PS[c]-TP[c] and row['FN_includes_unknown_and_no_map']==GS[c]-TP[c]
  checks.append({'sequence':seq['name'],'group':g,'independent_boolean_GT_recount':True,'unknown_no_map_main_misses':True,'shared_support':True})
result={'status':'PASS_LOCAL_FULL_HASH_AND_INDEPENDENT_GT_RECOUNT','archive_sha256':receipt['sha256'],'artifact_files_verified':len(manifest),'methods_scene_pairs':len(checks),'render_views':60,'counts_exact':True,'metrics_tolerance':1e-12,'checks':checks,'GPU_used':False,'isolated_final_evaluation':False}
(step/'gt_dev_v2/local_integrity_and_GT_recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
