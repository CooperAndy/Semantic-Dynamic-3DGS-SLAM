from pathlib import Path
import numpy as np,json,csv,hashlib,cv2
from PIL import Image
from collections import defaultdict
base=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';dataset=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene06/clone');g2=base/'g2_online_dev_full_v2';depth=g2/'depth/Scene06__clone';orb=g2/'orb_run/Scene06__clone';pred=base/'dev_jointtrust_selection_100_150_20261002_v7/prediction_root/Scene06/clone';out=Path('/root/autodl-tmp/stage2_round14_scene06_20261031_v1');out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();traj=np.loadtxt(orb/'CameraTrajectory.txt');assert traj.shape==(270,12);times=np.loadtxt(g2/'orb_input/Scene06__clone/times.txt');assert len(times)==270
rows=[];records=[];bbox=defaultdict(dict)
with (dataset/'bbox.txt').open() as f:
 for x in csv.DictReader(f,delimiter=' '):
  if x['cameraID']=='0' and x['trackID']=='12':bbox[int(x['frame'])]=x
for i in range(130,138):
 gt=dataset/'frames/instanceSegmentation/Camera_0'/f'instancegt_{i:05}.png';dl=depth/f'{i:06}_depth.npy';left=dataset/'frames/rgb/Camera_0'/f'rgb_{i:05}.jpg';right=dataset/'frames/rgb/Camera_1'/f'rgb_{i:05}.jpg';j=pred/'instances'/f'{i:06}.json';n=pred/'instances'/f'{i:06}_masks.npz';g=np.asarray(Image.open(gt))==13;z=np.load(dl);assert z.shape==g.shape;ann=bbox[i];assert int(g.sum())==int(ann['number_pixels']);valid=g&np.isfinite(z)&(z>0);g_small=cv2.resize(g.astype(np.uint8),(640,192),interpolation=cv2.INTER_NEAREST).astype(bool);z_small=cv2.resize(z,(640,192),interpolation=cv2.INTER_NEAREST);validsmall=g_small&np.isfinite(z_small)&(z_small>0);js=json.loads(j.read_text());m=np.load(n)['masks'];best=None
 for inst in js['instances']:
  x=m[inst['mask_index']];inter=int((g&x).sum());union=int((g|x).sum());score=inter/union if union else 0
  if best is None or score>best[0]:best=(score,inst)
 rows.append({'frame':i,'GT_track':12,'GT_isMoving':ann['isMoving'],'gt_mask_pixels':int(g.sum()),'gt_mask_pixels_p5':int(g_small.sum()),'prediction_depth_valid_gt_pixels':int(valid.sum()),'p5_valid_gt_pixels':int(validsmall.sum()),'pred_depth_median_on_gt':float(np.median(z[valid])) if valid.any() else None,'best_pred_iou':best[0] if best else None,'best_pred_track':best[1]['track_id'] if best else None,'best_pred_confidence':best[1]['confidence'] if best else None,'orb_timestamp':float(times[i])})
 for typ,p in [('left_rgb',left),('right_rgb',right),('gt_mask',gt),('pred_depth',dl),('pred_json',j),('pred_masks',n)]:records.append({'frame':i,'kind':typ,'path':str(p),'sha256':sha(p)})
for typ,p in [('orb_trajectory',orb/'CameraTrajectory.txt'),('orb_run_manifest',orb/'orb_run_manifest.json'),('orb_times',g2/'orb_input/Scene06__clone/times.txt'),('camera_intrinsic',dataset/'intrinsic.txt'),('camera_extrinsic_GT_not_pose_input',dataset/'extrinsic.txt'),('gt_bbox',dataset/'bbox.txt'),('gt_info',dataset/'info.txt')]:records.append({'frame':None,'kind':typ,'path':str(p),'sha256':sha(p)})
run=json.loads((orb/'orb_run_manifest.json').read_text());assert sha(orb/'CameraTrajectory.txt')==run['trajectory_sha256'] and run['official_world_pose_used'] is False;assert all(np.isfinite(traj[130:138]).ravel());(out/'input_audit.json').write_text(json.dumps({'scene':'Scene06/clone','frames':list(range(130,138)),'scope':'exposed DEV; GT only oracle and diagnostics','trajectory_shape':list(traj.shape),'orb_time_index':'row index aligned with times.txt; verify manifest generation','records':records,'rows':rows},indent=2));
with (out/'vehicle_gt_and_prediction.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
