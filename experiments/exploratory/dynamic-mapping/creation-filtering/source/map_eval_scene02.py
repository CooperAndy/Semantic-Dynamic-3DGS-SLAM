from pathlib import Path
import numpy as np,csv,json,cv2
from PIL import Image
r=Path('/root/autodl-tmp/stage2_round16_scene02_v1');d=Path('/root/autodl-tmp/datasets/vkitti2_2.0.3/extracted/Scene02/clone');labels={int(x['trackID']):x['label'] for x in csv.DictReader((d/'info.txt').open(),delimiter=' ')};ann={}
with (d/'bbox.txt').open() as f:
 for x in csv.DictReader(f,delimiter=' '):
  if x['cameraID']=='0' and 206<=int(x['frame'])<=231:ann[(int(x['frame']),int(x['trackID']))]=x
rows=[];region=[]
for i in range(26):
 frame=206+i;gm=cv2.resize(np.asarray(Image.open(d/'frames/instanceSegmentation/Camera_0'/f'instancegt_{frame:05d}.png')),(640,192),interpolation=cv2.INTER_NEAREST).astype(int);groups={name:np.zeros(gm.shape,bool) for name in ['moving','static']}
 for code in np.unique(gm):
  tid=int(code)-1;a=ann.get((frame,tid));
  if a and labels.get(tid) in ('Car','Van','Truck'):groups['moving' if a['isMoving']=='True' else 'static']|=gm==code
 for mode in ['D0','D3']:
  z=np.load(r/'maps'/mode/'eval'/f'audit_float_{i:04d}.npz');valid=z['valid_depth'].astype(bool)
  for name,m in groups.items():
   keep=m&valid;n=int(keep.sum());region.append({'frame':frame,'policy':mode,'GT_vehicle':name,'valid_pixels':n,'rgb_mse':float(((z['rgb']-z['reference_rgb'])[keep]**2).mean()) if n else None,'silhouette_coverage':float((z['silhouette'][keep]>.5).mean()) if n else None})
for mode in ['D0','D3']:
 audit=[json.loads(x) for x in (r/mode/'creation_audit.jsonl').read_text().splitlines()];met={k:np.loadtxt(r/'maps'/mode/'eval'/f'{k}.txt') for k in ['psnr','ssim','lpips']};
 for i in range(26):
  x=audit[i];rows.append({'policy':mode,'frame':206+i,'created':x['actual_created'],'total':x['total_gaussians'],'rejected_valid_dynamic':x['rejected_valid_dynamic'],'rejected_valid_uncertain':x['rejected_valid_uncertain'],'seconds':x['frame_elapsed_seconds'],'torch_peak_allocated_bytes':x['torch_peak_allocated_bytes'],**{k:float(v[i]) for k,v in met.items()}})
for name,data in [('map_per_frame.csv',rows),('map_regions.csv',region)]:
 with (r/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
print(json.dumps({mode:{'final':next(x['total'] for x in rows if x['policy']==mode and x['frame']==231),'mean_psnr':float(np.mean([x['psnr'] for x in rows if x['policy']==mode])),'mean_ssim':float(np.mean([x['ssim'] for x in rows if x['policy']==mode])),'mean_lpips':float(np.mean([x['lpips'] for x in rows if x['policy']==mode]))} for mode in ['D0','D3']},indent=2))
