from pathlib import Path
import csv,json,hashlib,numpy as np,cv2
from PIL import Image
r=Path('/root/autodl-tmp/stage2_round16_creation_audit_v1');gtroot=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene06/clone');b=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';pred=b/'dev_jointtrust_selection_100_150_20261002_v7/prediction_root/Scene06/clone/instances';labels={int(x['trackID']):x['label'] for x in csv.DictReader((gtroot/'info.txt').open(),delimiter=' ')};ann={}
with (gtroot/'bbox.txt').open() as f:
 for x in csv.DictReader(f,delimiter=' '):
  if x['cameraID']=='0' and 130<=int(x['frame'])<=137:ann[(int(x['frame']),int(x['trackID']))]=x
rows=[];instances=[];state_names={0:'Static',1:'Uncertain',2:'Dynamic',255:'Ignore'}
for i in range(8):
 frame=130+i;z=np.load(r/f'frame{i:02}.npz');gm=cv2.resize(np.asarray(Image.open(gtroot/'frames/instanceSegmentation/Camera_0'/f'instancegt_{frame:05d}.png')),(640,192),interpolation=cv2.INTER_NEAREST).astype(int);cand=z['candidate'];valid=z['valid_depth'];state=z['state'];depth=z['depth_m'];j=json.loads((pred/f'{frame:06d}.json').read_text());pm=np.load(pred/f'{frame:06d}_masks.npz')['masks'];occupied=cv2.resize((pm.any(0) if len(pm) else np.zeros((375,1242),bool)).astype(np.uint8),(640,192),interpolation=cv2.INTER_NEAREST)>0;gt_type=np.full(gm.shape,'other_unknown',dtype='<U16');gt_track=np.full(gm.shape,-1,np.int32)
 for code in np.unique(gm):
  tid=int(code)-1;x=ann.get((frame,tid));label=labels.get(tid)
  if x and label in ('Car','Van','Truck'):
   m=gm==code;gt_track[m]=tid;gt_type[m]='moving' if x['isMoving']=='True' else 'static'
   gtfull=np.asarray(Image.open(gtroot/'frames/instanceSegmentation/Camera_0'/f'instancegt_{frame:05d}.png'))==code;best=max([int((gtfull&m0).sum())/int((gtfull|m0).sum()) if (gtfull|m0).any() else 0 for m0 in pm],default=0);instances.append({'frame':frame,'track_id':tid,'category':label,'GT_isMoving':x['isMoving'],'gt_pixels_full':int(gtfull.sum()),'gt_pixels_p5':int(m.sum()),'candidate_valid_pixels':int((m&cand&valid).sum()),'invalid_depth_pixels':int((m&~valid).sum()),'best_prediction_iou':best,'matched_iou_0_5':best>=.5})
 for group in ['moving','static','other_unknown']:
  region=gt_type==group
  for code,name in state_names.items():
   m=region&cand&valid&(state==code);rows.append({'scene':'Scene06','frame':frame,'GT_group':group,'pred_state':name,'valid_candidate_pixels':int(m.sum()),'candidate_without_predmask':int((m&~occupied).sum())})
 # Per-candidate records: coordinates/depth/GT-track/state/coverage retained columnar.
 y,x=np.nonzero(cand&valid);np.savez_compressed(r/f'candidate_records_{frame}.npz',x=x.astype('uint16'),y=y.astype('uint16'),depth_m=depth[y,x],GT_track=gt_track[y,x],GT_group=gt_type[y,x],pred_state=state[y,x],pred_instance_coverage=occupied[y,x])
 for tid in np.unique(gt_track[gt_track>=0]):
  rr=gt_track==tid;coverage={'track':int(tid),'frame':frame,'GT_group':str(gt_type[rr][0]),'area_pixels_p5':int(rr.sum()),'median_pred_depth_m':float(np.median(depth[rr&valid])) if (rr&valid).any() else None}
  for code,name in state_names.items():coverage['candidate_'+name]=int((rr&cand&valid&(state==code)).sum())
  next(x for x in instances if x['frame']==frame and x['track_id']==tid).update(coverage)
for filename,data in [('gt_state_counts.csv',rows),('gt_vehicle_instances.csv',instances)]:
 with (r/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in data for k in x)));w.writeheader();w.writerows(data)
summary={g:{n:sum(x['valid_candidate_pixels'] for x in rows if x['GT_group']==g and x['pred_state']==n) for n in state_names.values()} for g in ['moving','static','other_unknown']};print(json.dumps({'counts':summary,'instances':len(instances),'unmatched_instances':sum(not x['matched_iou_0_5'] for x in instances)},indent=2));(r/'gt_summary.json').write_text(json.dumps({'counts':summary,'vehicle_instances':len(instances),'unmatched_iou_0_5':sum(not x['matched_iou_0_5'] for x in instances),'definition':'candidate before D3 state AND depth>0; predicted mask IoU is per-GT best, nonexclusive; state map generated from predicted T2.3','caveat':'posthoc candidate reconstruction matched historical scalar counts only'},indent=2))
