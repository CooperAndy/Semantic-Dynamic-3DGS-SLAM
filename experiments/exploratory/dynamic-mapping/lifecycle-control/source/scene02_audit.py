from pathlib import Path
import json,csv,hashlib,numpy as np
from PIL import Image
r=Path('/root/autodl-tmp/stage2_round12_20261031_v1');b=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';bridge=b/'dev_scene02_clone_p5_bridge_0_232_static_bg_v1';manifest=json.loads((bridge/'bridge_manifest.json').read_text());seq=bridge/'replica_layout/vkitti2_Scene02_clone_0_232';gt=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene02/clone');pred=b/'dev_online_scene02_clone_full_v1/instances';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
for i in range(206,232):
 rec=manifest['frame_records'][i];rgb=seq/'results'/f'frame{i:06d}.jpg';depth=seq/'results'/f'depth{i:06d}.png';state=bridge/'sidecars'/f'{i:06d}_state.npy';side=bridge/'sidecars'/f'{i:06d}_sidecar.json';assert sha(rgb)==rec['rgb_sha256'] and sha(depth)==rec['depth_png_sha256'] and sha(side)==rec['sidecar_sha256'];assert sha(state)==json.loads(side.read_text())['state_map']['sha256']
 gtp=gt/'frames/instanceSegmentation/Camera_0'/f'instancegt_{i:05d}.png';g=np.array(Image.open(gtp))==10;pjson=pred/f'{i:06d}.json';npz=pred/f'{i:06d}_masks.npz';d=json.loads(pjson.read_text());m=np.load(npz)['masks'];best=None
 for inst in d['instances']:
  mask=m[inst['mask_index']];inter=int((g&mask).sum());union=int((g|mask).sum());v=inter/union if union else 0
  if best is None or v>best[0]:best=(v,inst,inter)
 rows.append({'frame':i,'GT_track':9,'gt_pixels':int(g.sum()),'gt_sha256':sha(gtp),'prediction_json_sha256':sha(pjson),'prediction_masks_sha256':sha(npz),'best_iou':best[0] if best else None,'pred_track':best[1]['track_id'] if best else None,'pred_confidence':best[1]['confidence'] if best else None,'intersection':best[2] if best else 0,'prediction_count':len(d['instances']),'bridge_rgb_sha256':sha(rgb),'bridge_depth_sha256':sha(depth),'state_sha256':sha(state)})
out=r/'analysis';out.mkdir(exist_ok=True)
with (out/'scene02_track9_206_231.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps({'frames':len(rows),'best_iou_at_least_0_5':sum((x['best_iou'] or 0)>=.5 for x in rows),'predicted_track_ids':sorted(set(x['pred_track'] for x in rows if x['pred_track'] is not None)),'event_214_215':[x for x in rows if x['frame'] in [214,215]],'bridge_trajectory_sha256':sha(seq/'traj.txt'),'bridge_trajectory_expected':manifest['trajectory_bridge_sha256'],'pose_source':manifest['pose_source'],'depth_source':manifest['depth_source']},indent=2))
assert sha(seq/'traj.txt')==manifest['trajectory_bridge_sha256']
