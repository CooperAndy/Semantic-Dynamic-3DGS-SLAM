"""Supplement numerical audit with causal stereo source and gate coverage evidence."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import protect_parent;protect_parent.install()
from common import R,E,sha,dump
OLD=R.parent/'stage3_g20_20261029_p47149'
assert R.name=='stage3_g30_recovery_run_v2_20261101_p44181'
assert json.loads((R/'receipts/independent_audit.json').read_text())['status'].startswith('PASS_SIX_NATIVE_MAPPING_POSE_BIRTH_TARGET_AND_FLOAT_AUDIT')
for n in ['backend.py','factor_hooks.py','feature_pose.py','pose_chain.py','stable_birth.py','stable_pose.py','visibility.py']:
 assert sha(R/'scripts'/n)==sha(OLD/'scripts'/n),n
assert sha(R/'source/tracking_policy.py')==sha(OLD/'source/tracking_policy.py')
assert sha(R/'source/scripts/splatam_external_tm.py')==sha(OLD/'source/scripts/splatam_external_tm.py')
m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());K=np.asarray(m['K'],np.float32).astype(np.float64);points=0;proposals=0;ray_error=0.;summaries=[]
for v in ['B00_HARD_S0','B03_HARD_S0','B00_MASS_S0','B03_MASS_S0']:
 rows=[json.loads(s) for s in (R/'runs'/v/'backend_events.jsonl').read_text().splitlines()][1:]
 for e in rows:
  p=json.loads((R/'runs'/v/e['proposal_path']).read_text());ref=p['causal_stereo_reference'];i=e['frame'];a=e['anchor'];assert ref['frame']==a<i and ref['current_frame']==i
  r=m['rows'][a]
  for n in ['depth','confidence']:assert sha(r[n])==r['hashes'][n]==ref[n+'_sha256'] and r[n]==ref[n+'_path']
  if 'object_points' in p:
   xyz=np.asarray(p['object_points']).reshape(-1,3);xy=np.asarray(p['reference_pixels']).reshape(-1,2)
   if len(xyz):
    d=np.load(r['depth']);c=np.load(r['confidence']);valid=np.isfinite(d)&np.isfinite(c)&(d>=.5)&(d<=80)&(c>=.8);x=xy[:,0].astype(int);y=xy[:,1].astype(int);u=xy[:,0]-x;t=xy[:,1]-y;w=np.column_stack([(1-u)*(1-t),u*(1-t),(1-u)*t,u*t])
    z=np.sum(np.column_stack([d[y,x],d[y,x+1],d[y+1,x],d[y+1,x+1]])*w,axis=1);alpha=np.sum(np.column_stack([valid[y,x],valid[y,x+1],valid[y+1,x],valid[y+1,x+1]])*w,axis=1);assert (alpha>.99).all()
    expected=np.column_stack([(xy[:,0]-K[0,2])*z/K[0,0],(xy[:,1]-K[1,2])*z/K[1,1],z]);err=float(np.abs(expected-xyz).max());assert err<1e-9;ray_error=max(ray_error,err);points+=len(xyz)
  proposals+=1
 accepted=sum(e['accepted'] for e in rows)
 summaries.append(dict(variant=v,proposals=len(rows),geometry_ok=sum(e['geometry_ok'] for e in rows),support_sufficient=sum(e['support']['sufficient'] for e in rows),accepted=accepted,reasons=dict(collections.Counter(e['reason'] for e in rows))))
assert proposals==3996
dump(R/'receipts/stereo_input_independent_audit.json',dict(status='PASS_3996_NATIVE_STEREO_PROPOSAL_INPUTS_AND_UNCHANGED_METHOD_GATES',proposals=proposals,matched_3D_points=points,max_ray_coordinate_difference_m=ray_error,variants=summaries,actual_accepted_feedback=sum(x['accepted'] for x in summaries),feedback_propagation_exercised=any(x['accepted'] for x in summaries),new_window_GT_reads=0,quality_adoption=False))
print((R/'receipts/stereo_input_independent_audit.json').read_text())
