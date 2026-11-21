from common import *
import numpy as np
from PIL import Image
j=json.loads((R/'evaluation/summary.json').read_text());assert len(j['rows'])==240;m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());checks=[]
for x in j['rows']:
 if x['frame'] not in [19,199,999]:continue
 p=R/'evaluation'/x['variant']/x['snapshot']/f'{x["frame"]:06d}.npz'
 with np.load(p) as z:
  a=z['rgb'];ds=z['depth_silhouette'];assert hashlib.sha256(a.tobytes()).hexdigest()==x['rgb_array_sha256'] and hashlib.sha256(ds.tobytes()).hexdigest()==x['depth_silhouette_array_sha256']
  with Image.open(m['rows'][x['frame']]['rgb']) as im:b=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255
  se=np.square(a.astype(np.float64)-b.astype(np.float64)).sum();cs=np.square(np.clip(a,0,1).astype(np.float64)-b.astype(np.float64)).sum();err=max(abs(-10*np.log10(se/b.size)-x['PSNR_dB']),abs(-10*np.log10(cs/b.size)-x['clipped_PSNR_dB']),abs((ds[1]>.99).mean()-x['coverage099']));assert err<1e-8
  checks.append(dict(variant=x['variant'],snapshot=x['snapshot'],frame=x['frame'],max_arithmetic_error=float(err)))
assert len(checks)==42
for s in j['summaries']:
 a=[x for x in j['rows'] if all(x[k]==s[k] for k in ['variant','snapshot','group'])];assert len(a)==10
 for k,v in s['means'].items():assert abs(np.mean([x[k] for x in a])-v)<1e-10
# Recheck seed deltas, pass/fail conditions and aggregate independently of rendering.
for p in j['pairs']:
 for g in ['early','recent']:
  a=next(x['means'] for x in j['summaries'] if x['variant']==f'B00C0_S{p["seed"]}' and x['snapshot']=='final' and x['group']==g)
  b=next(x['means'] for x in j['summaries'] if x['variant']==f'B03C0_S{p["seed"]}' and x['snapshot']=='final' and x['group']==g)
  for k in a:assert abs(p[g+'_delta'][k]-(b[k]-a[k]))<1e-10
for g,st in j['paired_seed_statistics'].items():
 for k,v in st.items():
  vals=[p[g+'_delta'][k] for p in j['pairs']];assert abs(np.mean(vals)-v['mean'])<1e-10 and abs(np.std(vals,ddof=1)-v['sample_SD'])<1e-10
locks(json.loads((R/'registration.json').read_text()))
dump(R/'receipts/score_audit.json',dict(status='PASS42_FLOAT_SAMPLES24_GROUP_MEANS_AND_PAIRED_STATS',at=now(),checks=checks,GT_reads=0,native_GPU_calls=0,SSIM_LPIPS_scope='Single implementation with scalar/PNG/array hash evidence; not independently reimplemented'))
print('SCORE_AUDIT_PASS',flush=True)
