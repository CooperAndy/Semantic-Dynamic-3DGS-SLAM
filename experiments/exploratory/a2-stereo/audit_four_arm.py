import numpy as np,json
from pathlib import Path
r=Path('/root/autodl-tmp/A2_union_20261029');arms=('B','A','F','AF'); out=[]
for s in range(3):
 cfg=json.loads((r/f'lidar_config_s{s}_all4.json').read_text());sc=json.loads((r/f'lidar_score_s{s}_all4.json').read_text());records={x['source_frame']:x for x in sc['frame_results']}
 for x in cfg['frames']:
  d=np.load(x['reference']['path'])['depth'].astype('float64');m=np.isfinite(d)&(d>=.5)&(d<80);vals={};v={}
  for a in arms:
   z=np.load(x['renders'][a]['path']);v[a]=z['depth'].astype('float64');alpha=z['alpha'].astype('float64');vals[a]=m&np.isfinite(v[a])&np.isfinite(alpha)&(alpha>.99)&(v[a]>=.5)&(v[a]<80)
  common=m.copy()
  for a in arms:common &= vals[a]
  ref=records[x['source_frame']]['all_points']['all'];assert ref['common_count']==int(common.sum())
  for a in arms:
   assert ref['arms'][a]['covered_count']==int(vals[a].sum());assert np.isclose(np.abs(v[a][common]-d[common]).mean(),ref['arms'][a]['common_mae_m'],atol=1e-10,rtol=0)
  out.append([s,x['source_frame'],int(common.sum())])
(r/'lidar_all4_independent_audit.json').write_text(json.dumps({'status':'PASS','frames':out},indent=2));print('PASS',len(out))
