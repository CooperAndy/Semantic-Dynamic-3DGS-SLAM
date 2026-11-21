"""Independent Boolean GT recount and local CPU stress/V identity verification."""
import json,hashlib,tarfile,io
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence
R=Path(__file__).resolve().parents[1];D=R/'05_条件轻量候选与历史诊断';O=D/'dev_diagnostics_20261007_v2';A=D/'stage3_05_raw_complete_20261007_v2.tar.gz'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt=json.loads(json.loads((D/'raw_archive_receipt.json').read_text(encoding='utf-8-sig'))['stdout']);assert sha(A)==receipt['sha256'];statechecks=0
 with tarfile.open(A) as t:
  for m in t.getmembers():
   p=R/m.name;assert m.isfile() and p.resolve().is_relative_to(R.resolve())
   if '/states/' in m.name and 'S2-Z' not in m.name and p.exists():
    with np.load(p) as a,np.load(io.BytesIO(t.extractfile(m).read())) as b:
     for k in a.files:assert np.array_equal(a[k],b[k]),(m.name,k)
    statechecks+=1
  for name in ['CPU_result.json','V_trigger_diagnostic.json','history_recovery_trace.json']:
   remote=json.loads(t.extractfile('05_条件轻量候选与历史诊断/dev_diagnostics_20261007_v2/'+name).read());assert remote==json.loads((O/name).read_text(encoding='utf-8-sig'))
  t.extractall(R,filter='data')
 hashes=json.loads((D/'raw_05_file_hashes.json').read_text(encoding='utf-8-sig'));assert all(sha(R/p)==v for p,v in hashes.items())
 man=json.loads((R/'inputs/vkitti2_gt_dev_v2/manifest.json').read_text(encoding='utf-8-sig'));results=json.loads((O/'GT_results.json').read_text(encoding='utf-8-sig'));saved=np.load(O/'GT_confusion_matrices.npz');comparisons=0;curvechecks=0;poses=[]
 for seq in man['sequences']:
  name=seq['name'];d=R/'inputs/vkitti2_gt_dev_v2'/name;old=R/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{name}_full';groups=[r['group'] for r in results if r['sequence']==name];den=0;packed={}
  for g in groups:
   matrix=np.zeros((19,21),np.int64);chunks=[]
   for f in seq['evaluation']:
    gt=np.load(d/'gt_labels'/f'{f:06d}.npy');valid=gt!=255;p=(old if g in ['S0','S1','S-G','S2','S2-H'] else O/name)/'renders'/g/f'{f:06d}.npz'
    with np.load(p) as z:pred=z['label'];score=z['score']
    for c in range(19):
     labels,counts=np.unique(pred[valid&(gt==c)],return_counts=True);matrix[c,labels.astype(int)+2]+=counts
    if g in ['S2','S2-V','S2-Z']:take=valid&(pred>=0);chunks.append((score[take],pred[take]!=gt[take],int(valid.sum())))
    comparisons+=1
   assert np.array_equal(matrix,saved[name+'_'+g]);row=next(r for r in results if r['sequence']==name and r['group']==g);ious=[]
   # Independent literal TP/FN/FP reductions, rather than calling the production evaluator.
   for c in [0,2,5,6,7,8,9,10,13,14]:
    support=int(matrix[c].sum());tp=int(matrix[c,c+2]);union=support+int(matrix[:,c+2].sum())-tp
    if support:ious.append(tp/union)
   assert float(np.mean(ious))==row['mIoU_GT_supported_common_classes'];assert int(matrix[:,2:].sum())/int(matrix.sum())==row['coverage']
   if chunks:
    scores=np.concatenate([x[0] for x in chunks]);errors=np.concatenate([x[1] for x in chunks]);order=np.argsort(-scores,kind='stable');scores=scores[order];errors=errors[order];ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    with np.load(O/name/f'{g}_risk_curve.npz') as q:assert np.array_equal(q['threshold'],scores[ends]);assert np.array_equal(q['accepted'],ends+1);assert np.array_equal(q['errors'],np.cumsum(errors,dtype=np.int64)[ends]);assert int(q['GT_valid_denominator'])==sum(x[2] for x in chunks)
    curvechecks+=1
  with np.load(d/'base_map.npz') as base:ids=base['source_gaussian_index']
  if (O/name/'states/S2-Z_final.npz').exists():
   state=Evidence(ids,19)
   for f in seq['fusion']:
    with np.load(old/'observations'/f'{f:06d}_prediction.npz') as c:i=c['gaussian_index'];dist=c['distribution'];sem=c['semantic_reliability'];version=str(c['input_version'])
    with np.load(O/name/'Z_geometry_observations'/f'{f:06d}.npz') as z:assert np.array_equal(z['gaussian_id'],ids[i]);geo=z['geometry_reliability']
    full=np.zeros_like(state.evidence);full[i]=dist;weight=np.zeros(len(ids),float);weight[i]=sem.astype(float)*geo.astype(float);source={'dataset':'VKITTI2','sequence':seq['scene']+'/clone','camera':'Camera_0','original_frame':seq['records'][f]['original_frame']};state.update_frame(source,version+':Z_abs1_rel02_v1',full,weight)
   savedstate=Evidence.load(O/name/'states/S2-Z_final.npz',ids)
   for k in ['evidence','support','total_weight']:assert np.array_equal(getattr(state,k),getattr(savedstate,k))
   assert state.processed==savedstate.processed;statechecks+=1
  cp=np.load(d/'c2w.npy')
  for block in sorted({f//5 for f in seq['fusion']}):
   fs=[f for f in seq['fusion'] if f//5==block];p=cp[fs];centres=p[:,:3,3];directions=p[:,:3,2];directions/=np.linalg.norm(directions,axis=1,keepdims=True);distance=float(np.linalg.norm(centres[:,None]-centres[None,:],axis=-1).max());angle=float(np.degrees(np.arccos(np.clip(directions@directions.T,-1,1))).max());poses.append({'sequence':name,'block':block,'allowed_fusion_frames':fs,'max_camera_baseline_m':distance,'max_view_direction_change_deg':angle})
 dump(O/'view_group_pose_audit.json',poses);dump(O/'local_independent_verification.json',{'status':'PASS_LOCAL_05_INDEPENDENT_RECOUNT','raw_files_hash_verified':len(hashes),'local_remote_CPU_states_array_equal':statechecks,'local_remote_CPU_diagnostics_equal':True,'native_view_method_comparisons':comparisons,'all_main_integer_confusion_and_mIoU_recounted':True,'full_tied_risk_curves_recounted':curvechecks,'original_CPU_baseline_states_reproduced':True,'final_Test_accessed':False})
 print('PASS_LOCAL_05',comparisons,statechecks,curvechecks,flush=True)
if __name__=='__main__':main()
