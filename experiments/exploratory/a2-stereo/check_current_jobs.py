import json,hashlib,time,os
from pathlib import Path
R=Path('/root/autodl-tmp/A2_union_20261029')
jobs=[('00',s,R/'runs'/f'C0_g06_240_6000_200_s{s}',p) for s,p in enumerate((158769,158770,158771))]+[('03',s,R/'seq03/runs'/f'C0_g06_240_6000_200_s{s}',p) for s,p in enumerate((160288,160289,160290))]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for _ in range(720):
 records=[]
 for seq,s,p,pid in jobs:
  receipt=p/'combo_receipt.json'
  if receipt.exists():
   rec=json.loads(receipt.read_text()); mp=p/'params.npz'; status=rec['status']; ok=status=='PASS_ENGINEERING' and mp.exists() and sha(mp)==rec.get('map_sha256'); records.append({'seq':seq,'seed':s,'pid':pid,'status':status,'map_hash_verified':ok,'receipt':str(receipt)})
  elif Path(f'/proc/{pid}').exists():records.append({'seq':seq,'seed':s,'pid':pid,'status':'RUNNING'})
  else:records.append({'seq':seq,'seed':s,'pid':pid,'status':'EXITED_WITHOUT_RECEIPT'})
 out={'policy':'NO_NEW_RUNS','records':records,'status':'RUNNING' if any(x['status']=='RUNNING' for x in records) else ('COMPLETE_HASH_VERIFIED' if all(x.get('map_hash_verified') for x in records) else 'REQUIRES_INSPECTION')}
 tmp=R/'paused_current_jobs.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(R/'paused_current_jobs.json')
 if out['status']!='RUNNING':print(json.dumps(out),flush=True);break
 time.sleep(30)
