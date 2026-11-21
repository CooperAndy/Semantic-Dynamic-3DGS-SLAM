from common import *
import sys,os,importlib.util,time,numpy as np,torch
for x in json.loads((R/'receipts/raw_seal.json').read_text())['files']:assert sha(R/x['path'])==x['sha256']
def guard(e,a):
 if e=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(a[0])
  if '/poses/' in p or p.endswith('groundtruth.txt') or '/evaluation_only/' in p:raise RuntimeError('GT blocked')
sys.addaudithook(guard);sys.path.insert(0,str(R/'source'))
from external_tracking import render
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim
import lpips
from PIL import Image
torch.set_num_threads(4);spec=importlib.util.spec_from_file_location('eval_core',R/'source/scripts/splatam_external_tm.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
model=lpips.LPIPS(net='alex',version='0.1',spatial=False,verbose=False).cpu().eval();m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());rows=[];maps=[];start=time.monotonic()
with np.load(R/'runs/B00C0_S0/params.npz') as z:q=z['cam_unnorm_rots'].copy();t=z['cam_trans'].copy();K=z['intrinsics'].copy();w=z['w2c'].copy()
data=dict(cam=setup_camera(m['W'],m['H'],K,w),w2c=torch.tensor(w,device='cuda'));keys=['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales']
for v in VARIANTS:
 for snap,ids in [('199',HOLD[:10]),('398',HOLD[:10]),('final',HOLD[:10]+HOLD[-10:])]:
  dest=R/'evaluation'/v/snap;dest.mkdir(parents=True);file=R/'runs'/v/(f'params{snap}.npz' if snap!='final' else 'params.npz')
  with np.load(file) as z:
   count=len(z['means3D']);color=z['rgb_colors'];maps.append(dict(variant=v,snapshot=snap,gaussians=count,RGB_channel_outside=float(((color<0)|(color>1)).mean()),RGB_min=float(color.min()),RGB_max=float(color.max())));p={k:torch.tensor(z[k],device='cuda') for k in keys}
  p.update(cam_unnorm_rots=torch.tensor(q,device='cuda'),cam_trans=torch.tensor(t,device='cuda'))
  for i in ids:
   row=m['rows'][i];assert sha(row['rgb'])==row['hashes']['rgb'] and sha(row['depth'])==row['hashes']['depth']
   with Image.open(row['rgb']) as im:target=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
   dep=np.load(row['depth']);rgb,ds=render(p,data,i,core);assert np.isfinite(rgb).all() and np.isfinite(ds).all();se=np.square(rgb.astype(np.float64)-target.astype(np.float64)).sum();clipse=np.square(np.clip(rgb,0,1).astype(np.float64)-target.astype(np.float64)).sum()
   aa=torch.tensor(rgb)[None];bb=torch.tensor(target)[None]
   with torch.no_grad():ss=float(calc_ssim(aa,bb));lp=float(model(aa.clamp(0,1),bb,normalize=True))
   r=dict(variant=v,seed=definition(v)['seed'],snapshot=snap,frame=i,source_frame=i+3000,group='early' if i<200 else 'recent',PSNR_dB=float(-10*np.log10(se/target.size)),clipped_PSNR_dB=float(-10*np.log10(clipse/target.size)),SSIM=ss,LPIPS=lp,coverage099=float((ds[1]>.99).mean()),depth_proxy_MAE_m=float(np.abs(ds[0][dep>0]-dep[dep>0]).mean()),RGB_outside=float(((rgb<0)|(rgb>1)).mean()),rgb_sse=float(se),clipped_sse=float(clipse),channels=target.size)
   Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255)).save(dest/f'{i:06d}.png');r['rgb_array_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest();r['depth_silhouette_array_sha256']=hashlib.sha256(ds.tobytes()).hexdigest()
   if i in [19,199,999]:np.savez_compressed(dest/f'{i:06d}.npz',rgb=rgb,depth_silhouette=ds)
   rows.append(r)
   with (R/'evaluation/per_view_metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  del p;torch.cuda.empty_cache();print('SCORED',v,snap,len(rows),flush=True)
names=['PSNR_dB','SSIM','LPIPS','coverage099','depth_proxy_MAE_m','RGB_outside','clipped_PSNR_dB'];summaries=[]
for v in VARIANTS:
 for snap,groups in [('199',['early']),('398',['early']),('final',['early','recent'])]:
  for g in groups:
   rr=[r for r in rows if r['variant']==v and r['snapshot']==snap and r['group']==g];assert len(rr)==10
   summaries.append(dict(variant=v,snapshot=snap,group=g,means={n:float(np.mean([r[n] for r in rr])) for n in names}))
def mean(v,g):return next(x['means'] for x in summaries if x['variant']==v and x['snapshot']=='final' and x['group']==g)
pairs=[]
for seed in range(3):
 v0,v3=f'B00C0_S{seed}',f'B03C0_S{seed}';de={n:mean(v3,'early')[n]-mean(v0,'early')[n] for n in names};dr={n:mean(v3,'recent')[n]-mean(v0,'recent')[n] for n in names}
 ww={v:json.loads((R/'runs'/v/'smoke_receipt.json').read_text()) for v in [v0,v3]};pr=ww[v3]['gaussians']/ww[v0]['gaussians'];mr=ww[v3]['peak_torch_allocated_bytes']/ww[v0]['peak_torch_allocated_bytes']
 checks=dict(early_PSNR=de['PSNR_dB']>=.5,early_LPIPS=de['LPIPS']<=0,early_SSIM=de['SSIM']>=-.01,early_coverage=de['coverage099']>=-.02,recent_PSNR=dr['PSNR_dB']>=-.2,recent_LPIPS=dr['LPIPS']<=.01,point_ratio=pr<=1.2,memory_ratio=mr<=1.2)
 view_deltas=[]
 for i in HOLD:
  a=next(x for x in rows if x['variant']==v0 and x['snapshot']=='final' and x['frame']==i);b=next(x for x in rows if x['variant']==v3 and x['snapshot']=='final' and x['frame']==i)
  view_deltas.append(dict(frame=i,source_frame=i+3000,group=a['group'],delta={n:b[n]-a[n] for n in names}))
 early=[x for x in view_deltas if x['group']=='early'];recent=[x for x in view_deltas if x['group']=='recent']
 tail=dict(early_worst_PSNR_dB=min(x['delta']['PSNR_dB'] for x in early),early_max_LPIPS=max(x['delta']['LPIPS'] for x in early),recent_max_LPIPS=max(x['delta']['LPIPS'] for x in recent))
 tc=dict(early_worst_PSNR=tail['early_worst_PSNR_dB']>=-1,early_max_LPIPS=tail['early_max_LPIPS']<=.10,recent_max_LPIPS=tail['recent_max_LPIPS']<=.03)
 pairs.append(dict(seed=seed,early_delta=de,recent_delta=dr,checks=checks,tail=tail,tail_checks=tc,point_ratio=pr,memory_ratio=mr,per_view=view_deltas,mean_gate_pass=all(checks.values()),tail_gate_pass=all(tc.values())))
stats={g:{n:dict(mean=float(np.mean([p[g+'_delta'][n] for p in pairs])),sample_SD=float(np.std([p[g+'_delta'][n] for p in pairs],ddof=1)),min=float(min(p[g+'_delta'][n] for p in pairs)),max=float(max(p[g+'_delta'][n] for p in pairs))) for n in names} for g in ['early','recent']}
assert len(rows)==240
dump(R/'evaluation/summary.json',dict(status='COMPLETE240_VIEW3_SEED_PAIRED_EVALUATION',at=now(),rows=rows,maps=maps,summaries=summaries,pairs=pairs,paired_seed_statistics=stats,all_seed_mean_gates_pass=all(p['mean_gate_pass'] for p in pairs),tracking_feedback_eligible=all(p['mean_gate_pass'] and p['tail_gate_pass'] for p in pairs),native_forwards=480,backward=0,Adam=0,seconds=time.monotonic()-start,quality_adoption=False,scope='ExposedKITTI08 DEV;1 shared frontend,3 backend seeds,within-pair H0 donated frozen births/targets,not strictly online or tracking gain'))
print('EVALUATION_COMPLETE',flush=True)
