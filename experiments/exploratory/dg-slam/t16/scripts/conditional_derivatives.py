"""CPU finite differences with initial native compositor branches fixed, audit only."""
from pathlib import Path
import json,hashlib,numpy as np
from compare_branches import composite,load,native_tile
F=Path(__file__).resolve().parents[1]
P=json.loads((F/'repair_protocol.json').read_text(encoding='utf-8'));manifest=json.loads((F/'trace_result.json').read_text())
def get(frame,axis,h):
 row=next(r for r in manifest['captures'] if r['frame']==frame and r['h']==h and (h==0 or r['axis']==axis))
 path=F/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];return load(path)
rows=[]
for frame in P['frames']:
 base=get(frame,3,0);prepared={};errors=[]
 for tile in P['diagnostic_tiles']:
  q=composite(base,tile);native=native_tile(base,tile);ys,xs=q['ys'],q['xs'];valid=base['support'][ys,xs]
  errors.append({'tile':tile,'RGB_max':float(np.abs(q['image']-native['image']).max()),'depth_max':float(np.abs(q['depth']-native['depth']).max())})
  prepared[tile]=(q,valid,np.sign(q['image'].astype(float)-base['observed_color'][ys,xs].astype(float)),np.sign(q['depth'].astype(float)-base['observed_depth'][ys,xs].astype(float)))
 def signed_value(a):
  v=0.
  for tile,(q0,valid,rgb_sign,depth_sign) in prepared.items():
   q=composite(a,tile,ids=q0['branch']['ids'],frozen=q0['branch'])
   delta=.9*(rgb_sign*(q['image'].astype(float)-q0['image'].astype(float))).sum(-1)+.1*depth_sign*(q['depth'].astype(float)-q0['depth'].astype(float))
   assert np.isfinite(delta).all();v+=float(delta[valid].sum())
  return v
 fd=[]
 for h in P['conditional_FD_h']:
  v=[]
  for axis in P['diagnostic_axes']:
   v.append((signed_value(get(frame,axis,h))-signed_value(get(frame,axis,-h)))/(2*h))
  fd.append(v)
 rows.append({'frame':frame,'axes':P['diagnostic_axes'],'h':P['conditional_FD_h'],'FD_gradients':fd,'CPU_reconstruction':errors,
  'regional_support_pixels':int(sum(int(v[1].sum()) for v in prepared.values()))})
print('CONDITIONAL_FD',frame,fd,flush=True)
out={'scope':'Diagnostic conditional function only, never candidate scoring','cases':rows,'protocol_sha256':hashlib.sha256((F/'repair_protocol.json').read_bytes()).hexdigest()}
(F/'receipts/conditional_derivatives.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('CONDITIONAL_DERIVATIVES_WRITTEN')
