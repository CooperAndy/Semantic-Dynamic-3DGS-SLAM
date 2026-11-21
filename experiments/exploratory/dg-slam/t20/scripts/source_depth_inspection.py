"""Reproduce the saved first-iteration frame158 depth description on CPU."""
import json
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parents[1]
def quantile(x,w,p):
    ix=np.argsort(x);x=x[ix];w=w[ix];c=np.cumsum(w)
    return float(x[np.searchsorted(c,p*c[-1])])
rows=[]
for arm,name in [('NO_GS_REFINE','N_NATIVE_FIXED'),('LOCAL_ID','I_MAP_FIXED')]:
    root=F/f'runs/DEV200_s2718_{arm}'/name/'trace'
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['stage']=='native_before_ba' and e['iteration']==0 and 158 in e['frames'])
    with np.load(root/event['file']) as z:
        edge=z['jj']==24+event['frames'].index(158)
        weight=z['effective_weight'][edge].sum(axis=1).ravel().astype(float)
        sens=z['source_disps_sens'][edge].ravel();disp=z['source_disps'][edge].ravel()
        for label,mask in [('sensor_missing',sens<=0),('sensor_valid',sens>0)]:
            w=weight[mask];d=disp[mask]
            rows.append({'case':name,'frame':158,'group':label,'effective_weight_percent':100*float(w.sum()/weight.sum()),
                'estimated_disparity_at_floor_weight_percent':100*float(w[d<=.00100001].sum()/w.sum()),
                'estimated_depth_weighted_p10_m':quantile(1/d,w,.1),'estimated_depth_weighted_median_m':quantile(1/d,w,.5),
                'estimated_depth_weighted_p90_m':quantile(1/d,w,.9)})
previous=json.loads((F/'receipts/source_depth_inspection.json').read_text())
for A,B in zip(previous['rows'],rows):
    for key in A:
        if isinstance(A[key],(float,int)):assert abs(A[key]-B[key])<=1e-9
        else:assert A[key]==B[key]
print(json.dumps({'status':'PASS','groups_reproduced':len(rows),'GT_used':False}))
