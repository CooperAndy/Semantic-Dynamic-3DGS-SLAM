"""CPU diagnostic on sealed pre-addition renderer snapshots (4 selected frames)."""
import hashlib,json
from pathlib import Path
import numpy as np
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200');frames=[1,24,98,178]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 out=O/'diagnostic_20261028'/'training_residual_audit.json';assert not out.exists()
 seal=json.loads((R/'A2_DEV_evidence_seal_20261028.json').read_text());locks={x['path']:x['sha256'] for x in seal['files']};cal=O/'gradient_calibration_20261028.json';assert sha(cal)==locks[str(cal)]
 c=json.loads(cal.read_text());fb=c['stereo_08']['fb_px_m'];gamma=c['gamma']['B3'];m=json.loads((B/'inputs/runtime/input_manifest.json').read_text());result=[]
 histories={}
 for arm in ('B0','B3'):
  p=R/'runs'/f'{arm}_200'/'addition_history.jsonl';assert sha(p)==locks[str(p)];histories[arm]={x['frame']:x for x in map(json.loads,p.read_text().splitlines())}
 for f in frames:
  row=m['rows'][f]
  for k in ('depth','confidence'):assert sha(row[k])==row['hashes'][k]
  depth=np.load(row['depth']);conf=np.load(row['confidence']);frame={'frame':f,'groups':{}}
  rendered={}
  for arm in ('B0','B3'):
   info=histories[arm][f];path=R/'runs'/f'{arm}_200'/info['mask_file'];assert sha(path)==info['mask_sha256']
   with np.load(path) as z:
    assert np.array_equal(z['observed'],depth) and np.array_equal(z['valid'],depth>0)
    rendered[arm]=z['rendered'].copy()
  for lo,hi in ((.5,10.),(10.,30.),(30.,80.)):
   key=f'{lo:g}-{hi:g}';mask=(depth>=lo)&(depth<hi)&np.isfinite(conf);n=int(mask.sum())
   if n==0:frame['groups'][key]={'pixels':0};continue
   z=depth[mask].astype(np.float64);cval=conf[mask].astype(np.float64);scale=np.clip(np.clip(z*z/fb,.1,3.)*(1+2*(1-cval)),.1,3.)
   cell={'pixels':n,'B3_scale_mean':float(scale.mean()),'B3_scale_max_clip_fraction':float((scale>=2.999999).mean())}
   for arm in ('B0','B3'):
    residual=rendered[arm][mask].astype(np.float64)-z
    slope=gamma*.3/scale*np.minimum(np.abs(residual)/scale,1.)
    cell[arm]={'mean_abs_residual_m':float(np.mean(np.abs(residual))),'signed_residual_mean_m':float(np.mean(residual)),
               'B3_loss_grad_abs_mean_relative_L1':float(slope.mean()),'huber_quadratic_fraction':float((np.abs(residual)<=scale).mean())}
   frame['groups'][key]=cell
  result.append(frame)
 report={'status':'COMPLETE','frames':result,'scope':'four actual pre-addition snapshots from training frames; analytic per-pixel gradient coefficient, no claim of full-map optimizer gradients, geometry GT, or causal proof','quality_adoption':False}
 out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
