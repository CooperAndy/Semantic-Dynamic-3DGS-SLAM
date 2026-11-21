"""Replace only backend reference depth/mask; all proposal and acceptance rules stay native."""
from pathlib import Path
import hashlib,json
import numpy as np
def install(backend,manifest):
 original=backend.Matcher
 assert not getattr(original,'_causal_stereo_input',False)
 class StereoMatcher:
  _causal_stereo_input=True
  def __init__(self,target):self.original=original(target)
  def fit(self,rgb,unused_depth,unused_alpha,unused_mask,K,initial,seed):
   i=int(seed)-180028;assert 1<=i<1000
   anchor=i-1
   if anchor%20==19:anchor-=1
   assert 0<=anchor<i
   row=manifest['rows'][anchor];arrays={}
   for name in ['depth','confidence']:
    p=Path(row[name]);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['hashes'][name];arrays[name]=np.load(p,allow_pickle=False)
   d,c=arrays['depth'],arrays['confidence'];assert d.dtype==c.dtype==np.float32 and d.shape==c.shape==(manifest['H'],manifest['W'])
   valid=np.isfinite(d)&np.isfinite(c)&(d>=.5)&(d<=80)&(c>=.8);mask=valid.copy();mask[:4]=False;mask[-4:]=False;mask[:,:4]=False;mask[:,-4:]=False
   result=self.original.fit(rgb,d,valid.astype(np.float32),mask,K,initial,seed)
   result['causal_stereo_reference']=dict(frame=anchor,current_frame=i,depth_sha256=row['hashes']['depth'],confidence_sha256=row['hashes']['confidence'],depth_path=row['depth'],confidence_path=row['confidence'])
   return result
 backend.Matcher=StereoMatcher
 return original
def verify_donor(R,variant):
 old=R.parent/'stage3_g20_20261029_p47149';new=R/'runs'/variant;prior=old/'runs/B00_ORB_S0'
 a=json.loads((new/'receipt.json').read_text());b=json.loads((prior/'receipt.json').read_text());assert a['initial_gaussian_sha256']==b['initial_gaussian_sha256']
 with np.load(new/'target_schedule.npz') as x,np.load(prior/'target_schedule.npz') as y:
  for k in ['base','history_03','flag_03']:assert np.array_equal(x[k],y[k][:1000]),k
 for i in range(1,1000):
  if i%20==19:continue
  with np.load(new/'addition_masks'/f'{i:06}.npz') as x,np.load(prior/'addition_masks'/f'{i:06}.npz') as y:
   assert np.array_equal(x['selected_ids'],y['selected_ids'])
 (R/'receipts'/f'donor_prefix_match_{variant}.json').write_text(json.dumps(dict(status='PASS_G20_FULL1000_DONOR_IDENTITY',variant=variant,initial_gaussian_sha256=a['initial_gaussian_sha256'],birth_files=949,frames=1000,quality_adoption=False),indent=2))
