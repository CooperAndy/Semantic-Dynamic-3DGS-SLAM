from pathlib import Path
import json,types,numpy as np,hashlib
from stereo_proposal import install
from protect_parent import forbidden
assert forbidden('/root/autodl-tmp/stage3_g20_evidence_20261029_p47149/controller.json')
assert forbidden('/root/autodl-tmp/stage3_g20_20261029_p47149/runs')
assert forbidden('/root/autodl-tmp/stage3_g29_stereo_prefix_v4_20261029_p47149/runs')
assert not forbidden('/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/runs')
R=Path(__file__).resolve().parent.parent;m=json.loads((R/'inputs/runtime/input_manifest.json').read_text());seen=[]
class Original:
 def __init__(self,target):self.target=target
 def fit(self,rgb,d,a,mask,K,initial,seed):seen.append((d,a,mask,seed));return dict(ok=True,reason='CONTRACT_ONLY')
backend=types.SimpleNamespace(Matcher=Original);install(backend,m)
for i in [1,19,20,999]:
 result=backend.Matcher(None).fit(None,None,None,None,None,None,180028+i);a=i-1 if (i-1)%20!=19 else i-2;assert result['causal_stereo_reference']['frame']==a<i
 d,alpha,mask,seed=seen[-1];assert alpha.dtype==np.float32 and mask.dtype==bool;assert not mask[:4].any() and not mask[-4:].any() and not mask[:,:4].any() and not mask[:,-4:].any();assert (d[mask]>=.5).all() and (d[mask]<=80).all()
rejected=False
try:backend.Matcher(None).fit(None,None,None,None,None,None,180028)
except AssertionError:rejected=True
assert rejected
(R/'receipts/stereo_input_cpu_contract.json').write_text(json.dumps(dict(status='PASS_STEREO_REFERENCE_WRAPPER_ANCHOR_MASK_HASH_AND_BAD_SEED',cases=4,source_sha256=hashlib.sha256((Path(__file__).parent/'stereo_proposal.py').read_bytes()).hexdigest(),GPU_calls=0,GT_reads=0,quality_adoption=False),indent=2))
