"""Validate real masks and compare every input tensor with the author loader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tum_window import ROOT,DATA,rows_for,check_masks,load_window,sha256

source=ROOT/'01_source_env/DG-SLAM/run_tum.py'
tree=ast.parse(source.read_text())
names=['parse_list','associate_frames','loadtum','image_stream']
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
scope={'os':os,'np':np,'cv2':cv2,'torch':torch}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
full_loader=scope['loadtum']
def prefix_loader(*args,**kwargs):
    return tuple(x[:30] for x in full_loader(*args,**kwargs))
scope['loadtum']=prefix_loader
author=scope['image_stream'](str(DATA))
actual=load_window(30)
assert len(author)==len(actual)==30
for i,(a,b) in enumerate(zip(author,actual)):
    assert a[0]==b[0]==i
    for field,x,y in zip(['rgb','depth','intrinsics','pose','mask'],a[1:],b[1:]):
        assert x.dtype==y.dtype and torch.equal(x,y),(i,field)
rows,index=rows_for(30)
assert check_masks(rows,required=True)==30
manifest=DATA/'seg_mask/manifest.json'
meta=json.loads(manifest.read_text())
fractions=[x['person_fraction'] for x in meta['masks']]
report={'status':'PASS','frames':30,'author_tensor_comparison':'EXACT_EQUAL_ALL_30',
        'compared':['RGB','depth','intrinsics','GT pose input','dilated/resized person mask'],
        'mask_provenance_and_hashes':'PASS','person_fraction_min_max':[min(fractions),max(fractions)],
        'mask_manifest_sha256':sha256(manifest),'index_sha256':sha256(index),
        'cuda_initialized':torch.cuda.is_initialized()}
assert not report['cuda_initialized']
out=ROOT/'05_reports/smoke30_20261011'
(out/'mask_and_loader_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
