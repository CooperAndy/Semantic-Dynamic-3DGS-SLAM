from common import *
import runpy
a=json.loads((R/'amendment_v2.json').read_text());assert sha(R/'registration.json')==a['original_registration_sha256']
for name,digest in a['source_files'].items():assert sha(R/name)==digest,name
assert a['real_cache_frames_at_amendment']==0
runpy.run_path(str(R/'audit.py'),run_name='__main__')
r=json.loads((R/'receipts/audit.json').read_text());r.update(contract_revision=2,amendment_sha256=sha(R/'amendment_v2.json'));dump(R/'receipts/audit.json',r)
