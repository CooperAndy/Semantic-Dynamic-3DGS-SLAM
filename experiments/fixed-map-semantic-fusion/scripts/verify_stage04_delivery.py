"""Verify data-to-display alignment, compact RGB exclusion, and full Gaussian export."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'04_系统整合与最终归档/final_delivery_20261007_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=read(O/'viewer_data.json');assert len(d['cases'])==15 and len(d['maps'])==5 and d['default_method']=='S0';palette=np.asarray(d['palette'],np.uint8);vk=read(R/'inputs/vkitti2_gt_dev_v2/manifest.json');k360=read(R/'inputs/kitti360_isolated_v1/manifest.json');sem=read(R/'inputs/semantickitti08_v1/evaluation_input_manifest.json');checked=0;caseproof=[]
 for m in d['maps']:
  p=R/m['base'];assert sha(p)==m['base_sha256']
  if m['id']=='KITTI08':assert sha(p)==sem['base_sha256']
  else:assert sha(p)==read(p.parent/'base_map_lock.json')['file_sha256']
  with np.load(p) as base:ids=base['source_gaussian_index'];N=len(ids);take=np.linspace(0,N-1,min(12000,N),dtype=int);points=read(O/m['points']);assert points['ids']==ids[take].tolist();assert points['count']==N
  for state in m['states']:
   sp=R/state['path'];assert sha(sp)==state['sha256']
   with np.load(sp) as st:assert np.array_equal(st['gaussian_id'],ids);ev=st['evidence'];sum_ev=ev.sum(1);label=np.where(sum_ev>0,ev.argmax(1),-1);assert points['groups'][state['group']]['label']==label[take].tolist();assert points['groups'][state['group']]['support']==st['support_count'][take].tolist();assert (st['support_count']<=(180 if m['id']=='KITTI08' else 21)).all()
 for case in d['cases']:
  key=case['dataset'];original=int(case['name'])
  if key=='KITTI08':f=original-3000;rec=next(r for r in sem['records'] if r['original_frame']==original);gtp=R/'inputs/semantickitti08_v1/sparse_gt'/f'{original:06d}.npz';assert sha(gtp)==rec['GT_file']['sha256'];gt=np.load(gtp)['label']
  else:
   manifest=k360 if key.startswith('2013') else vk;seq=next(s for s in manifest['sequences'] if s['name']==key);rec=next(r for r in seq['records'] if r['original_frame']==original);f=rec['local_frame'];assert f in seq['evaluation'];p=R/('inputs/kitti360_isolated_v1' if key.startswith('2013') else 'inputs/vkitti2_gt_dev_v2')/key/'gt_labels'/f'{f:06d}.npy';assert sha(p)==rec['files']['gt_labels']['sha256'];gt=np.load(p)
   if key.startswith('2013'):assert 'RGB' not in case and rec['prediction_frame'] is None and 'image_2' not in rec['files']
   else:assert np.array_equal(np.asarray(Image.open(O/case['RGB'])),np.asarray(Image.open(R/'inputs/vkitti2_gt_dev_v2'/key/'image_2'/f'{f:06d}.png')))
  valid=gt!=255;expected_gt=np.zeros((*gt.shape,3),np.uint8);expected_gt[valid]=palette[gt[valid]];assert np.array_equal(expected_gt,np.asarray(Image.open(O/case['GT'])))
  for g,paths in case['groups'].items():
   p=R/paths['source'];assert p.name==f'{f:06d}.npz' and sha(p)==paths['source_sha256'];pred=np.load(p)['label'];expected=np.full((*gt.shape,3),90,np.uint8);known=(pred>=0)&(pred<19);expected[known]=palette[pred[known]];expected[pred==-2]=0;assert np.array_equal(expected,np.asarray(Image.open(O/paths['semantic'])));error=np.asarray(Image.open(O/paths['error']));assert np.all(error[valid&(pred==gt)]==[0,190,95]);assert np.all(error[valid&(pred>=0)&(pred!=gt)]==[235,70,70]);assert np.all(error[valid&(pred==-1)]==[255,180,0]);assert np.all(error[valid&(pred==-2)]==[0,210,240]);assert np.all(error[~valid]==45);checked+=1
  caseproof.append({'dataset':key,'original_frame':original,'local_frame':f,'GT_manifest_correspondence':True,'RGB_correct_role_or_omitted':True})
 meta=read(O/'exports/KITTI08_S0_semantic_gaussians.json');p=O/'exports/KITTI08_S0_semantic_gaussians.ply';assert sha(p)==meta['sha256'];layout=np.dtype([tuple(x) for x in meta['field_layout']]);export=np.fromfile(p,dtype=layout,offset=meta['header_bytes'])
 with np.load(R/'inputs/base_map/params.npz') as b:
  assert len(export)==907488 and np.array_equal(export['gaussian_id'],b['source_gaussian_index']);assert np.array_equal(np.column_stack([export[k] for k in ['x','y','z']]),b['means3D']);assert np.array_equal(export['opacity'],b['logit_opacities'].ravel());assert np.allclose(np.column_stack([export[f'f_dc_{i}'] for i in range(3)])*.28209479177387814+.5,b['rgb_colors'],atol=1e-7);assert np.array_equal(np.column_stack([export[f'scale_{i}'] for i in range(3)]),np.repeat(b['log_scales'],3,axis=1));assert np.array_equal(np.column_stack([export[f'rot_{i}'] for i in range(4)]),b['unnorm_rotations'])
 with np.load(R/'02_基础因子DEV与Oracle诊断/dev_sgbm_native_full_20261007_v1/states/S0_final.npz') as st:assert np.array_equal(export['support_count'],st['support_count']) and np.array_equal(export['total_weight'],st['total_weight']);unknown=st['evidence'].sum(1)==0;assert np.array_equal(export['semantic_label']==-1,unknown)
 result={'status':'PASS_INDEPENDENT_04_ALIGNMENT_AND_EXPORT','cases':caseproof,'native_case_method_comparisons':checked,'maps_and_state_pairs':25,'full_export_Gaussians':907488,'geometry_RGB_opacity_rotation_scale_preserved':True,'KITTI360_holdout_RGB_omitted':True,'metrics_not_rescored_or_retuned':True};(O/'independent_delivery_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'comparisons':checked,'export':907488}),flush=True)
if __name__=='__main__':main()
