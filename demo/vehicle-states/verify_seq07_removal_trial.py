from pathlib import Path
import json,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent.parent
DEMO=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本'
BASE=DEMO/'scene-data/KITTI07_C60S42'; OUT=BASE/'VehicleRemovalTrial'
mask=np.load(OUT/'selected_mask.npz',allow_pickle=False)['selected']
def load(p):
 with p.open('rb') as f:
  header=b''
  while not header.endswith(b'end_header\n'):header+=f.readline()
 n=int(re.search(rb'element vertex (\d+)',header).group(1))
 return np.memmap(p,dtype='<f4',mode='r',offset=len(header),shape=(n,17))
for v in ['F1','F200']:
 a=load(BASE/f'KITTI07_{v}/KITTI07_{v}_RGB.ply');b=load(OUT/f'{v}_RGB_removed.ply');c=load(OUT/f'{v}_RGB_highlight.ply')
 assert len(b)==len(a)-int(mask.sum());offset=0
 for i in range(0,len(a),200000):
  part=a[i:i+200000];m=mask[i:i+len(part)];n=int((~m).sum());q=c[i:i+len(part)]
  assert np.array_equal(b[offset:offset+n],part[~m]) and np.array_equal(q[~m],part[~m]) and np.array_equal(q[m,:6],part[m,:6]) and np.array_equal(q[m,9:],part[m,9:]);offset+=n
 assert offset==len(b);print(v,'PASS retained',offset,'selected',int(mask.sum()))
cfg=json.loads((DEMO/'源码/scene-config.kitti07-v009-track20-trial.json').read_text(encoding='utf8'));q=cfg['dynamicVehicleRemoval'];assert q['removedCount']==int(mask.sum()) and all((DEMO/q['variants'][v]['rgb']).is_file() and (DEMO/q['highlights'][v]).is_file() for v in ['F1','F200'])
print('config PASS')
