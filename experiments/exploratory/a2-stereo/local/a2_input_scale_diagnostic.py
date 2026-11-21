"""CPU descriptive loss-scale distribution on frozen E6a holdout input."""
import hashlib,json
from pathlib import Path
import numpy as np
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry');R=O/'mapper_v1';B=Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/B200')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 out=O/'diagnostic_20261028'/'input_scale_diagnostic.json';assert not out.exists()
 seal=json.loads((R/'A2_DEV_evidence_seal_20261028.json').read_text());locks={x['path']:x['sha256'] for x in seal['files']}
 cal=O/'gradient_calibration_20261028.json';assert sha(cal)==locks[str(cal)];c=json.loads(cal.read_text());fb=c['stereo_08']['fb_px_m'];gamma=c['gamma']['B3'];m=json.loads((B/'inputs/runtime/input_manifest.json').read_text())
 bins=[(.5,10.),(10.,30.),(30.,80.)];values={f'{lo:g}-{hi:g}':[] for lo,hi in bins};confidence={k:[] for k in values};tot={k:0 for k in values}
 for frame in range(19,200,20):
  row=m['rows'][frame]
  for key in ('depth','confidence'):assert sha(row[key])==row['hashes'][key]
  depth=np.load(row['depth']);conf=np.load(row['confidence']);valid=np.isfinite(depth)&(depth>.5)&(depth<80)&np.isfinite(conf)&(conf>=.8)&(conf<=1)
  for lo,hi in bins:
   key=f'{lo:g}-{hi:g}';mask=valid&(depth>=lo)&(depth<hi);d=depth[mask].astype(np.float64);v=conf[mask].astype(np.float64)
   scale=np.clip(np.clip(d*d/fb,.1,3.)*(1+2*(1-v)),.1,3.)
   values[key].append(scale);confidence[key].append(v);tot[key]+=len(d)
 result={}
 for key in values:
  s=np.concatenate(values[key]);c=np.concatenate(confidence[key]);slope=gamma*.3/s
  result[key]={'pixels':len(s),'scale_quantiles_m':np.quantile(s,[.1,.5,.9]).tolist(),'max_gradient_slope_quantiles_relative_L1':np.quantile(slope,[.1,.5,.9]).tolist(),'confidence_quantiles':np.quantile(c,[.1,.5,.9]).tolist(),'floor_fraction':float((s<=.1000001).mean()),'ceiling_fraction':float((s>=2.999999).mean())}
 data={'status':'COMPLETE','bins':result,'scope':'all valid E6a input pixels in ten mapper-holdout DEV frames; theoretical Huber tail slopes, no rendering support, no sensor noise or causal proof','fb':fb,'gamma':gamma}
 out.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
if __name__=='__main__':main()
