"""Independent local raw reprojection, integer metrics and whole-tie risk recount."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root);D=R/'inputs/semantickitti08_v1';B=R/'02_基础因子DEV与Oracle诊断/semantickitti08_gt_v1';E=B/'GT_evaluation';manifest=json.loads((D/'evaluation_input_manifest.json').read_text(encoding='utf-8'));proto=json.loads((E/'protocol.json').read_text(encoding='utf-8'));mapping={int(k):v for k,v in proto['semantic_mapping_raw_to_City19'].items()};primary=proto['evaluation_classes'];hashes=json.loads((B/'evidence_file_hashes.json').read_text(encoding='utf-8'))
 for n,digest in hashes.items():assert sha(R/n)==digest,n
 assert sha(R/'inputs/base_map/params.npz')==proto['base_sha256']
 for g,digest in proto['state_sha256'].items():assert sha(R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/states'/(g+'_final.npz'))==digest
 calib={}
 for line in (D/'calib.txt').read_text().splitlines():
  if ':' in line:k,v=line.split(':',1);calib[k]=np.fromstring(v,sep=' ')
 P=calib['P2'].reshape(3,4);T=calib['Tr'].reshape(3,4);h=manifest['image_height'];w=manifest['image_width'];size=h*w;cms={g:{} for g in proto['groups']+['2D_input_teacher']};curves={g:[] for g in proto['groups']};masklocks={ (x['frame'],x['subgroup']):x for x in json.loads((E/'shared_subgroup_mask_locks.json').read_text(encoding='utf-8'))}
 def add(g,k,gt,pred,mask):
  mat=np.zeros((19,21),np.int64);np.add.at(mat,(gt[mask],pred[mask]+2),1);cms[g][k]=cms[g].get(k,np.zeros_like(mat))+mat
 for rec in manifest['records']:
  f=rec['original_frame'];pts=np.fromfile(D/'velodyne'/f'{f:06d}.bin',dtype='<f4').reshape(-1,4);packed=np.fromfile(D/'labels'/f'{f:06d}.label',dtype='<u4');assert len(pts)==len(packed)==rec['points'];xyz=pts[:,:3].astype(np.float64);camera=np.einsum('ij,nj->ni',T[:,:3],xyz)+T[:,3];q=np.einsum('ij,nj->ni',P[:,:3],camera)+P[:,3];z=q[:,2];positive=np.isfinite(q).all(1)&np.isfinite(pts).all(1)&(z>0);uv=np.zeros((len(pts),2));uv[positive]=q[positive,:2]/z[positive,None];uv=np.rint(uv).astype(np.int64);inside=positive&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);idx=np.flatnonzero(inside);pixel=uv[idx,1]*w+uv[idx,0];depth_order=np.argsort(z[idx],kind='stable');ordered=idx[depth_order];opixel=pixel[depth_order];pi,first=np.unique(opixel,return_index=True);chosen=ordered[first]
  with np.load(D/'sparse_gt'/f'{f:06d}.npz') as p:
   identity=np.full(size,-1,np.int32);identity[pi]=chosen;assert np.array_equal(identity,p['point_index'].ravel()),f
   label=p['label'].ravel()[pi];raw=p['raw_semantic'].ravel()[pi];inst=p['instance'].ravel()[pi];depth=p['depth'].ravel()[pi];expected=np.array([mapping.get(int(v),255) for v in (packed[chosen]&65535)],np.uint8);assert np.array_equal(label,expected);assert np.array_equal(raw,packed[chosen]&65535) and np.array_equal(inst,packed[chosen]>>16);assert np.allclose(depth,z[chosen],rtol=1e-7,atol=1e-5)
  teacherpath=Path(rec['input_locks']['labels']['path']);remote='/root/autodl-tmp/stage3_chy_20261006/';rel=str(teacherpath).replace('\\','/').split(remote)[-1];localteacher=R/rel
  if not localteacher.exists():localteacher=Path(__file__).resolve().parents[1]/rel
  assert sha(localteacher)==rec['input_locks']['labels']['sha256'];teacher=np.load(localteacher).ravel()[pi];valid=label!=255;vehicle=np.isin(label,[13,14]);tiny=np.zeros_like(valid);x=pi%w;y=pi//w
  for ident in np.unique(inst[vehicle&(inst>0)]):
   v=vehicle&(inst==ident)
   if v.sum()>=5 and (x[v].max()-x[v].min()+1)*(y[v].max()-y[v].min()+1)<=size*.0025:tiny|=v
  moving=(raw>=252)&(raw<=259);shared=None
  for g in proto['groups']:
   with np.load(B/'renders'/g/f'{f:06d}.npz') as p:assert np.array_equal(pi,p['pixel_index']);pred=p['label'].astype(int);score=p['score'];alpha=p['alpha'];md=p['map_depth']
   if (f-3000)%10==9:
    with np.load(R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/renders'/g/f'{f-3000:06d}.npz') as old:assert np.array_equal(pred,old['label'].ravel()[pi]) and np.array_equal(score,old['score'].ravel()[pi]) and np.array_equal(alpha,old['alpha'].ravel()[pi])
   if shared is None:shared=(alpha.copy(),md.copy())
   else:assert np.array_equal(shared[0],alpha) and np.array_equal(shared[1],md)
   assert np.array_equal(pred==-2,alpha<.05);agree=(alpha>=.05)&np.isfinite(md)&(md>0)&(np.abs(md-depth)<=1+.02*depth);local=f-3000;masks={'all200':valid,'fusion180':valid&(local%10!=9),'excluded20':valid&(local%10==9),'GT_moving':valid&moving,'GT_nonmoving':valid&~moving,'small_vehicle_visible_bbox_proxy':valid&vehicle&tiny,'other_vehicle_or_insufficient_bbox':valid&vehicle&~tiny,'map_GT_depth_agrees_proxy':valid&agree,'map_GT_depth_disagrees_proxy':valid&(alpha>=.05)&~agree,'no_map_support':valid&(alpha<.05)}
   for k,m in masks.items():add(g,k,label,pred,m);assert hashlib.sha256(m.tobytes()).hexdigest()==masklocks[f,k]['sha256']
   if g=='S0':
    for k,m in masks.items():add('2D_input_teacher',k,label,teacher.astype(int),m)
   accept=valid&(pred>=0);curves[g].append((score[accept],pred[accept]!=label[accept],int(valid.sum())))
  if f%20==19:print(json.dumps({'verified_through_frame':f}),flush=True)
 with np.load(E/'GT_confusion_matrices.npz') as saved:
  for g,sub in cms.items():
   for k,cm in sub.items():assert np.array_equal(cm,saved[g+'__'+k])
 for r in json.loads((E/'GT_results.json').read_text(encoding='utf-8')):
  cm=cms[r['group']][r['subgroup']];n=int(cm.sum());known=int(cm[:,2:].sum());correct=sum(cm[c,c+2] for c in range(19));ious=[]
  for c in primary:
   gt=int(cm[c].sum());tp=int(cm[c,c+2]);fp=int(cm[:,c+2].sum())-tp
   if gt:ious.append(tp/(gt+fp))
  assert n==r['GT_valid_pixels'] and known==r['labelled_pixels'];assert int(cm[:,0].sum())==r['no_map_pixels'] and int(cm[:,1].sum())==r['unknown_pixels']
  for actual,expected in [(np.mean(ious) if ious else None,r['mIoU_GT_supported_common_classes']),(known/n if n else None,r['coverage']),(correct/known if known else None,r['conditional_accuracy'])]:assert actual is None and expected is None or actual is not None and expected is not None and abs(actual-expected)<1e-12
 for g,parts in curves.items():
  scores=np.concatenate([p[0] for p in parts]);error=np.concatenate([p[1] for p in parts]);den=sum(p[2] for p in parts);order=np.argsort(scores,kind='stable')[::-1];scores=scores[order];error=error[order];end=np.flatnonzero(np.r_[scores[1:]!=scores[:-1],True]);accepted=end+1;wrong=np.cumsum(error,dtype=np.int64)[end]
  with np.load(E/(g+'_risk_curve.npz')) as p:assert np.array_equal(scores[end],p['threshold']);assert np.array_equal(accepted,p['accepted']) and np.array_equal(wrong,p['errors']);assert np.array_equal(accepted/den,p['coverage']) and np.array_equal(wrong/accepted,p['risk'])
 receipt={'status':'PASS_INDEPENDENT_LOCAL_RAW_PROJECTION_GT_INTEGER_METRICS_AND_FULL_RISK','verifier_version':'v2_extra_old100_sample_reproduction','verifier_source_sha256':sha(Path(__file__)),'old100_sampled_renders_identical':True,'frames':200,'renders':1000,'evidence_files_verified':len(hashes),'source_and_states_immutable_server_receipt':json.loads((B/'final_integrity.json').read_text(encoding='utf-8'))['status'],'GT_dense':False,'exposed_DEV':True};(B/'local_independent_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
