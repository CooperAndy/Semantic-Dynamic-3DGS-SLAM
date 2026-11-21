import json,shutil,time
from pathlib import Path
R=Path('/root/autodl-tmp/A2_union_20261029')
for i in range(720):
 p=R/'resume00_complete.json'
 if p.exists():break
 log=(R/'resume00_af.log').read_text(errors='replace')
 if 'Traceback (most recent call last)' in log:raise RuntimeError('resume failed; inspect resume00_af.log')
 time.sleep(30)
else:raise TimeoutError('00 resume incomplete')
assert json.loads(p.read_text())['status']=='COMPLETE_00_SCORED_INDEPENDENT_LIDAR_ARITHMETIC_AUDIT'
assert json.loads((R/'independent_lidar_score_audit.json').read_text())['status']=='PASS_SCORE_RECOMPUTE'
dest=R/'package_seq00';dest.mkdir(exist_ok=False)
patterns=('evaluation_200_s*.json','eval_*_s*.json','lidar_*s*.json','independent_lidar_score_audit.json','pre_score_seal.json','pre_run_locks.json','resume00_complete.json','resume00_af.log','input_audit.json')
for f in {p for pat in patterns for p in R.glob(pat)}:shutil.copy2(f,dest/f.name)
for f in (R/'runs').glob('*_200_s*/combo_receipt.json'):shutil.copy2(f,dest/(f.parent.name+'_receipt.json'))
with (R/'finalize_watch.log').open('a') as f:f.write('PACKAGE_READY 00\n')
print('PACKAGE_READY 00',flush=True)
