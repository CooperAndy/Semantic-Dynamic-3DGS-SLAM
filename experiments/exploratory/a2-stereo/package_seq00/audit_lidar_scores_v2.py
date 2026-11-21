import json,hashlib,sys
from pathlib import Path
import numpy as np
P=Path('/root/autodl-tmp/A2_union_20261029');seq=sys.argv[1];R=P if seq=='00' else P/f'seq{seq}'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
checked=[]
for seed in (0,1,2):
 for b,a in (('B','A'),('F','AF'),('B','AF'),('B','F')):
  cfg=json.loads((R/f'lidar_config_s{seed}_{b}_{a}.json').read_text());sc=json.loads((R/f'lidar_score_s{seed}_{b}_{a}.json').read_text());assert cfg['arms']==[b,a] and sc['status']=='DESCRIPTIVE_SCORE'; by={x['source_frame']:x for x in sc['frame_results']}
  for row in cfg['frames']:
   assert sha(row['reference']['path'])==row['reference']['sha256'];d=np.load(row['reference']['path'])['depth'].astype('float64');ref=np.isfinite(d)&(d>=.5)&(d<80);vs={};render={}
   for name,item in row['renders'].items():
    assert sha(item['path'])==item['sha256'];z=np.load(item['path']);v=z['depth'].astype('float64');alpha=z['alpha'].astype('float64');render[name]=v;vs[name]=ref&np.isfinite(v)&np.isfinite(alpha)&(alpha>.99)&(v>=.5)&(v<80)
   common=vs[a]&vs[b];buckets={'all':ref,'0.5-10':ref&(d<10),'10-30':ref&(d>=10)&(d<30),'30-80':ref&(d>=30)};old=by[row['source_frame']]['all_points']
   for key,mask in buckets.items():
    cm=mask&common;cell=old[key];assert int(mask.sum())==cell['reference_count'] and int(cm.sum())==cell['common_count']
    for name in (a,b):
     v=render[name];vm=vs[name]&mask;err=np.abs(v-d);x=cell['arms'][name];assert int(vm.sum())==x['covered_count']
     if cm.any():assert np.isclose(err[cm].mean(),x['common_mae_m'],rtol=0,atol=1e-10)
     if mask.any():assert np.isclose((np.minimum(err[vm],5).sum()+5*(mask.sum()-vm.sum()))/mask.sum(),x['capped_mae_missing_penalty_m'],rtol=0,atol=1e-10)
  checked.append({'seed':seed,'comparison':b+'-'+a,'frames':len(cfg['frames'])})
(R/'independent_lidar_score_audit.json').write_text(json.dumps({'status':'PASS_SCORE_RECOMPUTE','sequence':seq,'checked':checked,'scope':'arithmetic and locked projected reference; no timing or occlusion verification'},indent=2));print('PASS',seq,len(checked),'comparisons')
