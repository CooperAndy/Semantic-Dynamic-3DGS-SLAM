from pathlib import Path
import json,tarfile,hashlib
root=Path('/root/autodl-tmp/stage2_round17_delivery_v1');b=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';base=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted');paths=[]
for scene,a,c in [('Scene06',130,137),('Scene02',206,231)]:
 d=base/scene/'clone'
 for name in ['bbox.txt','info.txt','colors.txt','intrinsic.txt','extrinsic.txt','pose.txt']:
  p=d/name
  if p.is_file():paths.append(p)
 for f in range(a,c+1):
  for x in [d/'frames/instanceSegmentation/Camera_0'/f'instancegt_{f:05}.png',d/'frames/depth/Camera_0'/f'depth_{f:05}.png']:
   if x.is_file():paths.append(x)
 if scene=='Scene06':
  pred=b/'dev_jointtrust_selection_100_150_20261002_v7/prediction_root/Scene06/clone';geo=b/'g2_online_dev_full_v2';paths += [geo/'orb_run/Scene06__clone/CameraTrajectory.txt',geo/'orb_run/Scene06__clone/orb_run_manifest.json',geo/'orb_input/Scene06__clone/prepare_manifest.json',geo/'orb_input/Scene06__clone/times.txt',geo/'depth/Scene06__clone/generation_manifest.json']
  for f in range(a,c+1):paths += [d/'frames/rgb/Camera_0'/f'rgb_{f:05}.jpg',d/'frames/rgb/Camera_1'/f'rgb_{f:05}.jpg',geo/'depth/Scene06__clone'/f'{f:06}_depth.npy']
  paths += sorted((pred/'instances').glob('*.json'))+sorted((pred/'instances').glob('*_masks.npz'))+sorted((pred/'evidence').glob('*.json'))+[pred/'run_manifest.json']
 else:
  pred=b/'dev_online_scene02_clone_full_v1/instances';paths += [pred/f'{f:06}.json' for f in range(a,c+1)]+[pred/f'{f:06}_masks.npz' for f in range(a,c+1)]
assert all(p.is_file() for p in paths);assert all('Scene20' not in str(p) for p in paths);paths=list(dict.fromkeys(paths));manifest=[]
for p in paths:
 h=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append({'source':str(p),'bytes':p.stat().st_size,'sha256':h,'archive_path':str(p.relative_to('/root/autodl-tmp'))})
out=root/'DEV_gt_prediction_sources.tar.gz'
with tarfile.open(out,'w:gz',compresslevel=1,dereference=True) as t:
 for p in paths:t.add(p,arcname=str(p.relative_to('/root/autodl-tmp')),recursive=False)
with tarfile.open(out,'r:gz') as t:
 seen=0
 for m in t:
  if m.isfile():
   f=t.extractfile(m)
   while f.read(4*1024*1024):pass
   seen+=1
assert seen==len(paths);h=hashlib.sha256(out.read_bytes()).hexdigest();(root/'DEV_gt_prediction_sources_manifest.json').write_text(json.dumps({'archive':str(out),'archive_bytes':out.stat().st_size,'archive_sha256':h,'files':manifest,'scope':'Scene06 130-137 and historical 100-150 prediction inputs; Scene02 206-231 GT and predictions; not full training dataset'},indent=2));print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':h,'files':len(paths)},indent=2))
