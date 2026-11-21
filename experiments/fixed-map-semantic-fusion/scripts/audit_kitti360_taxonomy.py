"""Independent official ignoreInEval validation, no metric/method changes."""
import ast,json,hashlib,io
from pathlib import Path
import requests,numpy as np
from PIL import Image
from http_range_zip import RangeZip
from audit_kitti360_entry import ROOT as R,D,URLS,dump
B=R/'03_方法冻结与隔离评价/kitti360_taxonomy_audit_v1';CITY_IDS=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
def main():
 B.mkdir(parents=True,exist_ok=True);url='https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/helpers/labels.py'
 if (B/'official_labels_source.txt').exists():raw=(B/'official_labels_source.txt').read_bytes()
 else:
  r=requests.get(url,stream=True,timeout=45)
  try:r.raise_for_status();raw=r.raw.read(262145);assert len(raw)<=262144
  finally:r.close()
 assert hashlib.sha256(raw).hexdigest()=='5fd33f3742fff242d01bcecb77d9f25c0db9646938ca2265911647f52eb82e86'
 (B/'official_labels_source.txt').write_bytes(raw);tree=ast.parse(raw.decode());labels=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='labels' for t in n.targets));defs=[]
 for call in labels.elts:
  assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='Label';args=[ast.literal_eval(x) for x in call.args];defs.append({'name':args[0],'semanticID':args[1],'trainID':args[3],'ignoreInEval':args[7]})
 assert sorted(x['semanticID'] for x in defs if not x['ignoreInEval'])==CITY_IDS;lookup={x['semanticID']:x for x in defs};man=json.loads((D/'manifest.json').read_text());selection=json.loads((D/'metadata_selection.json').read_text());z=RangeZip(URLS['labels'],80*1024*1024);assert z.etag==selection['labels_archive']['etag'];checks=[]
 for s in man['sequences']:
  selected=next(x for x in selection['sequences'] if x['sequence']==s['name']);folder=B/s['name'];folder.mkdir(exist_ok=True)
  for f in s['evaluation']:
   orig=str(s['records'][f]['original_frame']);b=z.read(selected['semantic_members'][orig]);p=folder/f'{f:06d}_official_semantic.png';p.write_bytes(b);sem=np.asarray(Image.open(io.BytesIO(b)));gt=np.full(sem.shape,255,np.uint8)
   for label in defs:
    if not label['ignoreInEval']:gt[sem==label['semanticID']]=label['trainID']
   assert np.array_equal(gt,np.load(D/s['name']/'gt_labels'/f'{f:06d}.npy'))
   b=z.read(selected['instance_members'][orig]);pinst=folder/f'{f:06d}_official_instance.png';pinst.write_bytes(b);inst=np.asarray(Image.open(io.BytesIO(b)));assert np.array_equal(inst,np.load(D/s['name']/'gt_instances'/f'{f:06d}.npy'))
   checks.append({'sequence':s['name'],'local_frame':f,'original_frame':orig,'mapped_GT_matches_official_nonignored_classes':True,'instance_original_unchanged':True,'raw_GT_semantic_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_instance_sha256':hashlib.sha256(pinst.read_bytes()).hexdigest(),'raw_semantic_ID_counts':{str(i):int(n) for i,n in zip(*np.unique(sem,return_counts=True))}})
 dump(B/'taxonomy_validation.json',{'status':'PASS_OFFICIAL_TAXONOMY_AND_RAW_GT_MAPPING','official_source':url,'official_source_sha256':hashlib.sha256(raw).hexdigest(),'official_label_definitions':defs,'frozen_semantic_IDs_nonignored':CITY_IDS,'checks':checks,'GT_metrics_or_masks_changed':False,'transfer':z.provenance(),'documentation_clarification':'Frozen protocol text says City19 trainID255 only; implementation frozen before GT uses exactly the19 official ignoreInEval=False IDs. Garage34/gate35/smallpole37 have suggested valid trainIDs but official ignoreInEval=True, and are correctly ignored. Clarification only; sealed protocol and masks unchanged.'})
 (B/'类别映射审计与预声明文字澄清.md').write_text('# KITTI-360类别映射独立审计\n\n官方ignoreInEval=False的19个semanticID与首次GT读取前固定的CITY_IDS完全一致；十份原始语义PNG按官方表独立转换后逐像素等于已有GT数组，实例PNG也一致。未改评价分母、结果、方法或选窗。原始PNG/官方表及其SHA256已补存。\n\n预声明中“仅City19 trainID255排除”的文字不够准确：garage(34)、gate(35)、smallpole(37)的建议trainID虽有效，官方ignoreInEval仍为True。此次实现已按冻结19个非ignore类正确排除；这里只补充文字澄清，原预声明哈希和历史结果保留。\n',encoding='utf-8');print(json.dumps({'status':'PASS_TAXONOMY_AND_RAW_GT','views':10,'changed_masks_or_results':False}),flush=True)
if __name__=='__main__':main()
