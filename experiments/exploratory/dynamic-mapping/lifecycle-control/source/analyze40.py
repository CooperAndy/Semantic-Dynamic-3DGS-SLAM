from pathlib import Path
import json,csv,numpy as np
r=Path('/root/autodl-tmp/stage2_round12_20261031_v1');out=r/'analysis';out.mkdir(exist_ok=True);freeze=json.loads((r/'input_freeze.json').read_text());rows=[];summary={}
for mode in ['D0','D3']:
 aud=[json.loads(l) for l in (r/mode/'creation_audit.jsonl').read_text().splitlines()];assert len(aud)==40
 for a in aud:a['source_frame']=freeze['source_frames'][a['frame']];rows.append(a)
 ev=r/'maps'/mode/'eval';metrics={k:np.atleast_1d(np.loadtxt(ev/(k+'.txt'))) for k in ['psnr','ssim','lpips']};assert all(len(x)==40 for x in metrics.values())
 summary[mode]={'gaussians_final':aud[-1]['total_gaussians'],'total_created':sum(x['actual_created'] for x in aud),'rejected_valid_depth_dynamic':sum(x['rejected_valid_dynamic'] for x in aud),'rejected_valid_depth_uncertain':sum(x['rejected_valid_uncertain'] for x in aud),'rejected_valid_depth_ignore':sum(x['rejected_valid_ignore'] for x in aud),'psnr_mean_all_40':float(metrics['psnr'].mean()),'ms_ssim_mean_all_40':float(metrics['ssim'].mean()),'lpips_mean_all_40':float(metrics['lpips'].mean()),'wall_seconds':json.loads((r/mode/'run_metrics.json').read_text())['elapsed_seconds'],'max_frame_torch_allocated_mib':max(x['torch_peak_allocated_bytes'] for x in aud)/1048576}
for name,rs in [('creation.csv',rows)]:
 with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rs[0]);w.writeheader();w.writerows(rs)
regional=[]
for i in range(40):
 state=np.load(r/'p5_window/sidecars/D3'/f'{i:06d}_state.npy');z={mode:np.load(r/'maps'/mode/'eval'/f'audit_float_{i:04d}.npz') for mode in ['D0','D3']};assert np.array_equal(z['D0']['reference_rgb'],z['D3']['reference_rgb']) and np.array_equal(z['D0']['valid_depth'],z['D3']['valid_depth'])
 for label,code in [('static',0),('uncertain',1),('dynamic',2),('ignore',255)]:
  sel=(state==code)&z['D0']['valid_depth'].astype(bool);n=int(sel.sum())
  for mode in ['D0','D3']:
   q=z[mode];regional.append({'frame':i,'source_frame':freeze['source_frames'][i],'region':label,'policy':mode,'valid_depth_pixels':n,'silhouette_coverage':float((q['silhouette'][sel]>.5).mean()) if n else None,'rgb_mse':float(((q['rgb']-q['reference_rgb'])[sel]**2).mean()) if n else None})
with (out/'regional.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=regional[0]);w.writeheader();w.writerows(regional)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
