"""CPU-only exact comparison with the author loader; separate window receipts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import ast
import json
import cv2
import numpy as np
import torch
from tum_window import ROOT,DATA,rows_for,check_masks,load_window,sha256

p=argparse.ArgumentParser()
p.add_argument('--frames',type=int,choices=[30,200],required=True)
a=p.parse_args()
source=ROOT/'01_source_env/DG-SLAM/run_tum.py'
tree=ast.parse(source.read_text())
names=['parse_list','associate_frames','loadtum','image_stream']
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
scope={'os':os,'np':np,'cv2':cv2,'torch':torch}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
full_loader=scope['loadtum']
scope['loadtum']=lambda *args,**kwargs: tuple(x[:a.frames] for x in full_loader(*args,**kwargs))
author=scope['image_stream'](str(DATA))
actual=load_window(a.frames)
assert len(author)==len(actual)==a.frames
for i,(expected,received) in enumerate(zip(author,actual)):
    assert expected[0]==received[0]==i
    for field,x,y in zip(['rgb','depth','intrinsics','pose','mask'],expected[1:],received[1:]):
        assert x.dtype==y.dtype and torch.equal(x,y),(i,field)
rows,index=rows_for(a.frames)
assert check_masks(rows,required=True)==a.frames
manifest=DATA/'seg_mask/manifest.json'
mask_meta=json.loads(manifest.read_text())
fractions=[x['person_fraction'] for x in mask_meta['masks']]
report={'status':'PASS','frames':a.frames,'author_tensor_comparison':f'EXACT_EQUAL_ALL_{a.frames}',
        'compared':['RGB','depth','intrinsics','GT pose input','dilated/resized person mask'],
        'mask_provenance_and_hashes':'PASS','person_fraction_min_max':[min(fractions),max(fractions)],
        'mask_manifest_sha256':sha256(manifest),'index_sha256':sha256(index),
        'cuda_initialized':torch.cuda.is_initialized()}
assert not report['cuda_initialized']
out=ROOT/'05_reports'/('dev200_20261011' if a.frames==200 else 'smoke30_20261011')
(out/'mask_and_loader_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
