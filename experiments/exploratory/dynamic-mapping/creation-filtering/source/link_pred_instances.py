from pathlib import Path
import numpy as np,json,cv2,csv
base=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';rows=[]
for scene,start,end,root,pred in [('Scene06',130,137,Path('/root/autodl-tmp/stage2_round16_creation_audit_v1'),base/'dev_jointtrust_selection_100_150_20261002_v7/prediction_root/Scene06/clone/instances'),('Scene02',206,231,Path('/root/autodl-tmp/stage2_round16_scene02_v1/D3'),base/'dev_online_scene02_clone_full_v1/instances')]:
 for frame in range(start,end+1):
  path=root/f'candidate_records_{frame}.npz';z=np.load(path);x=z['x'];y=z['y'];inst=json.loads((pred/f'{frame:06}.json').read_text())['instances'];m=np.load(pred/f'{frame:06}_masks.npz')['masks'];assignment=np.full(len(x),-1,np.int32);overlap=np.zeros(len(x),np.uint8)
  for item in inst:
   mask=cv2.resize(m[item['mask_index']].astype(np.uint8),(640,192),interpolation=cv2.INTER_NEAREST)>0;hit=mask[y,x];overlap+=hit.astype(np.uint8);assignment[hit]=int(item['track_id'])
  assignment[overlap>1]=-2;np.savez_compressed(root/f'candidate_prediction_links_{frame}.npz',pred_track=assignment,pred_instance_overlap_count=overlap);rows.append({'scene':scene,'frame':frame,'candidates':len(x),'no_pred_instance':int((overlap==0).sum()),'one_pred_instance':int((overlap==1).sum()),'ambiguous_overlap':int((overlap>1).sum())})
with Path('/root/autodl-tmp/stage2_round16_creation_audit_v1/prediction_links_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('frames',len(rows),'overlap',sum(x['ambiguous_overlap'] for x in rows))
