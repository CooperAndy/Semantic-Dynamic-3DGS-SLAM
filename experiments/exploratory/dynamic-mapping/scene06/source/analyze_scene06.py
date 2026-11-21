from pathlib import Path
import numpy as np,json,csv,cv2
r=Path('/root/autodl-tmp/stage2_round14_scene06_20261031_v2');rows=[];report={};events=json.loads((r/'ORACLE/active_events.json').read_text());eid=next(x for x in events if x['event']=='deactivate');ids=np.array(eid['ids'],dtype=int);assert len(ids)==296
params=[np.load(r/'maps/ORACLE'/f'params{i}.npz') for i in range(8)]
for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales']:
 diffs=[float(np.max(np.abs(params[i+1][k][ids]-params[i][k][ids]))) for i in (4,5,6)];assert max(diffs)==0,(k,diffs);report[k]=diffs
for pol in ['D0','D3','ORACLE']:
 audit=[json.loads(x) for x in (r/pol/'creation_audit.jsonl').read_text().splitlines()];p=r/'maps'/pol;metrics={k:np.atleast_1d(np.loadtxt(p/'eval'/f'{k}.txt')) for k in ['psnr','ssim','lpips']};assert all(len(a)==8 for a in metrics.values())
 for i in range(8):
  z=np.load(p/f'params{i}.npz');n=len(z['means3D']);active=z['gaussian_active'];assert len(active)==n
  rows.append({'policy':pol,'frame':130+i,'window_frame':i,'total':n,'active':int(active.sum()),'inactive':int((~active).sum()),'created':audit[i]['actual_created'],'rejected_valid_dynamic':audit[i]['rejected_valid_dynamic'],'rejected_valid_uncertain':audit[i]['rejected_valid_uncertain'],'oracle_attributed_count':int((z['gaussian_object_id']==12).sum()),'psnr':float(metrics['psnr'][i]),'ms_ssim':float(metrics['ssim'][i]),'lpips':float(metrics['lpips'][i])})
with (r/'per_frame.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
region=[]
for i in range(8):
 st=np.load(r/'bridge/sidecars'/f'{i:06d}_state.npy');gt=np.load(r/'bridge/oracle_masks'/f'{i:06d}_track12.npy');raw={pol:np.load(r/'maps'/pol/'eval'/f'audit_float_{i:04d}.npz') for pol in ['D0','D3','ORACLE']}
 assert all(np.array_equal(raw['D0']['reference_rgb'],raw[p]['reference_rgb']) and np.array_equal(raw['D0']['valid_depth'],raw[p]['valid_depth']) for p in ['D3','ORACLE'])
 for name,m in [('gt_vehicle',gt),('pred_static_excluding_gt_vehicle',(st==0)&~gt),('all_valid_depth',np.ones_like(gt,bool))]:
  mask=m&raw['D0']['valid_depth'].astype(bool);n=int(mask.sum())
  for pol,z in raw.items():
   mse=float(((z['rgb']-z['reference_rgb'])[mask]**2).mean()) if n else None;cov=float((z['silhouette'][mask]>.5).mean()) if n else None
   region.append({'frame':130+i,'region':name,'policy':pol,'valid_depth_pixels':n,'rgb_mse':mse,'silhouette_coverage':cov})
with (r/'region.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=region[0]);w.writeheader();w.writerows(region)
(r/'frozen_parameter_check.json').write_text(json.dumps(report,indent=2));print(json.dumps({'inactive':len(ids),'frozen_parameter_maxdiff':report,'oracle_events':[(x['frame'],x['event'],x.get('assigned_track12'),len(x.get('ids',[])),x.get('max_reprojection_error')) for x in events]},indent=2))
