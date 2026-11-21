from pathlib import Path
import numpy as np,csv,collections,json
outputs=[]
for scene,r in [('Scene06',Path('/root/autodl-tmp/stage2_round16_creation_audit_v1')),('Scene02',Path('/root/autodl-tmp/stage2_round16_scene02_v1'))]:
 ins=list(csv.DictReader((r/'gt_vehicle_instances.csv').open()));counts=list(csv.DictReader((r/'gt_state_counts.csv').open()));g=collections.defaultdict(lambda:{'instances':0,'unmatched':0,'gt_pixels':0,'invalid_depth_pixels':0,'candidate_valid_pixels':0,'Static':0,'Uncertain':0,'Dynamic':0,'Ignore':0})
 for x in ins:
  area=int(x['area_pixels_p5']);depth=float(x['median_pred_depth_m']) if x['median_pred_depth_m'] else None
  for dimension,bin in [('all','all'),('frame',x['frame']),('track',x['track_id']),('area', '1-31' if area<32 else '32-127' if area<128 else '128-1023' if area<1024 else '1024+'),('pred_depth_m','unknown' if depth is None else '0-20' if depth<20 else '20-40' if depth<40 else '40+')]:
   key=(scene,x['GT_group'],dimension,bin);o=g[key];o['instances']+=1;o['unmatched']+=x['matched_iou_0_5']=='False';o['gt_pixels']+=area;o['invalid_depth_pixels']+=int(x['invalid_depth_pixels']);o['candidate_valid_pixels']+=int(x['candidate_valid_pixels']);
   for s in ['Static','Uncertain','Dynamic','Ignore']:o[s]+=int(x['candidate_'+s])
 for key,v in g.items():outputs.append(dict(zip(['scene','GT_group','group_by','bin'],key),**v))
 print(scene,'counts',json.dumps(json.loads((r/'gt_summary.json').read_text())['counts']))
out=Path('/root/autodl-tmp/stage2_round16_creation_audit_v1/grouped_stats.csv')
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=outputs[0]);w.writeheader();w.writerows(outputs)
