from common import *
import sys,numpy as np
seed=int(sys.argv[1]);v=f'B00C0_S{seed}';p=R/'runs'/v;w=json.loads((p/'smoke_receipt.json').read_text());assert w['status']=='PASS_ENGINEERING' and w['frames']==N and w['native_camera_updates']==0
assert not donor_manifest(seed).exists()
with np.load(p/'target_schedule.npz') as z:
 base=z['base'];hist=z['history_03'];flag=z['flag_03'];assert base.shape==(N,60) and flag.sum()==2280
 for t in range(N):
  if t in HOLD:assert (base[t]==-1).all();continue
  keys=[0]+[q for q in range(1,t) if (q+1)%5==0 and q not in HOLD];old=[q for q in keys if q<=t-200]
  for it in range(60):
   assert base[t,it] in keys+[t]
   eligible=bool(old and it in [15,35,55]);assert flag[t,it]==eligible
   if eligible:assert hist[t,it]==old[np.random.default_rng(np.random.SeedSequence([seed,t,it,1])).integers(len(old))]
   else:assert hist[t,it]==base[t,it]
files=[p/'target_schedule.npz',p/'base_pools.jsonl',p/'addition_history.jsonl',p/'smoke_receipt.json']+sorted((p/'addition_masks').glob('*.npz'))
assert len(list((p/'addition_masks').glob('*.npz')))==949
dump(donor_manifest(seed),dict(status='SEALED_H0_PAIR_INPUTS',at=now(),seed=seed,files=[dict(path=f.relative_to(R).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files],actual_births=sum(x['added'] for x in w['execution_observation']['additions']),H3_started=False))
print('DONOR_SEALED',seed,flush=True)
