from common import *
import numpy as np,collections
locks(json.loads((R/'registration.json').read_text()));rows=[];paired=[];cameras=[];exposure=[]
for seed in range(3):
 check_donor(seed);init=[];births=[]
 with np.load(R/'runs'/f'B00C0_S{seed}'/'target_schedule.npz') as z:s={k:z[k].copy() for k in z.files}
 pools={x['frame']:x['targets'] for x in map(json.loads,(R/'runs'/f'B00C0_S{seed}'/'base_pools.jsonl').read_text().splitlines())}
 for t,pool in pools.items():
  for it in range(60):assert s['base'][t,it]==pool[np.random.default_rng(np.random.SeedSequence([seed,t,it,0])).integers(len(pool))]
 counts={};us={}
 for b in [0,3]:
  v=f'B{b:02d}C0_S{seed}';p=R/'runs'/v;w=json.loads((p/'smoke_receipt.json').read_text());assert w['status']=='PASS_ENGINEERING' and w['frames']==N and w['native_camera_updates']==w['GT_blocked_opens']==0
  u=[json.loads(x) for x in (p/'mapping_updates.jsonl').read_text().splitlines()];assert len(u)==57000;us[b]=u
  with np.load(p/'target_schedule.npz') as z:
   for k in z.files:assert np.array_equal(z[k],s[k])
  seen=set()
  for x in u:
   t,it=x['online'],x['iteration'];assert (t,it) not in seen;seen.add((t,it));assert t not in HOLD
   assert x['base_target']==int(s['base'][t,it]) and x['target']==int(s['history_03' if b else 'base'][t,it])
   assert x['history_slot']==bool(s['flag_03'][t,it] if b else False) and not x['color_box']
  assert sum(x['history_slot'] for x in u)==(2280 if b else 0)
  o=w['execution_observation'];assert o['completed_steps']==57000 and o['camera_backward_loss_calls']==o['box_projection_steps']==0
  a=[json.loads(x) for x in (p/'addition_history.jsonl').read_text().splitlines()];assert len(a)==949 and all(x['added']<=3000 for x in a)
  for x in a:
   f=p/'addition_masks'/f'{x["frame"]:06d}.npz';assert sha(f)==x['mask_sha256']
   with np.load(f) as z:assert len(z['selected_ids'])==x['added'] and len(np.unique(z['selected_ids']))==x['added']
  births.append([(x['frame'],x['added'],x['mask_sha256']) for x in a]);init.append(w['initial_gaussian_sha256'])
  with np.load(p/'estimated_trajectory.npz') as z:cameras.append(z['c2w'].copy())
  with np.load(p/'params.npz') as z:
   q=z['cam_unnorm_rots'].copy();t=z['cam_trans'].copy();assert not np.isin(z['timestep'],HOLD).any() and not np.isin(z['keyframe_time_indices'],HOLD).any()
   assert all(np.isfinite(z[k]).all() for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales'])
  with np.load(p/'params199.npz') as z:assert np.array_equal(z['cam_unnorm_rots'][...,:200],q[...,:200]) and np.array_equal(z['cam_trans'][...,:200],t[...,:200])
  counts[b]=collections.Counter(x['target'] for x in u)
  rows.append(dict(variant=v,steps=57000,history_slots=sum(x['history_slot'] for x in u),history_changed=sum(x['history_changed'] for x in u),births=sum(x['added'] for x in a)))
 assert init[0]==init[1] and births[0]==births[1]
 added=collections.Counter();removed=collections.Counter()
 for a,b in zip(us[0],us[3]):
  assert (a['online'],a['iteration'])==(b['online'],b['iteration'])
  if a['target']!=b['target']:removed[a['target']]+=1;added[b['target']]+=1
 e=[dict(target=k,H0=counts[0][k],H3=counts[3][k],added=added[k],removed=removed[k],net=counts[3][k]-counts[0][k]) for k in sorted(counts[0].keys()|counts[3].keys())]
 assert all(x['net']==x['added']-x['removed'] for x in e);exposure.append(dict(seed=seed,targets=e))
 paired.append(dict(seed=seed,same_initial_sha=init[0],same_actual_births=True))
assert all(np.array_equal(x,cameras[0]) for x in cameras)
files=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in (R/'runs').rglob('*') if p.is_file()]
dump(R/'receipts/raw_seal.json',dict(at=now(),files=files));dump(R/'receipts/target_exposure.json',dict(at=now(),pairs=exposure))
dump(R/'receipts/independent_audit.json',dict(status='PASS_SIX1000_FRAME_PAIRED_RUNS',at=now(),runs=rows,paired=paired,all_camera_arrays_identical=True,past200_cameras_unchanged=True,GT_reads=0,native_GPU_calls=0))
print('AUDIT_PASS',len(files),flush=True)
