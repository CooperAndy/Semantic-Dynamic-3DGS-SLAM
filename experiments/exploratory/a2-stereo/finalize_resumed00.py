from pathlib import Path
import hashlib,json,shutil
R=Path('/root/autodl-tmp/A2_union_20261029')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
seal=json.loads((R/'pre_score_seal.json').read_text());assert seal['status']=='MAPS_SEALED_BEFORE_SCORE' and len(seal['items'])==12
for x in seal['items']:assert sha(x['path'])==x['sha256']
audit=json.loads((R/'independent_lidar_score_audit.json').read_text());assert audit['status']=='PASS_SCORE_RECOMPUTE' and len(audit['checked'])==12
(R/'resume00_complete.json').write_text(json.dumps({'status':'COMPLETE_00_SCORED_INDEPENDENT_LIDAR_ARITHMETIC_AUDIT','quality_adoption':False,'seeds':[0,1,2],'audit_first_failed':'alpha dtype mismatch; preserved independent_lidar_audit_00.log','audit_correction':'float64 alpha before comparison, v2 passed'},indent=2))
dest=R/'package_seq00';dest.mkdir(exist_ok=False)
patterns=('evaluation_200_s*.json','eval_*_s*.json','lidar_*s*.json','independent_lidar_score_audit.json','pre_score_seal.json','pre_run_locks.json','resume00_complete.json','resume00_af.log','input_audit.json','independent_lidar_audit*00.log','audit_lidar_scores*.py')
for f in {p for pat in patterns for p in R.glob(pat)}:shutil.copy2(f,dest/f.name)
for f in (R/'runs').glob('*_200_s*/combo_receipt.json'):shutil.copy2(f,dest/(f.parent.name+'_receipt.json'))
with (R/'finalize_watch.log').open('a') as f:f.write('PACKAGE_READY 00\n')
print('COMPLETE 00 PACKAGE READY')
