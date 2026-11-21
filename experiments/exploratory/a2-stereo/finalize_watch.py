from pathlib import Path
import time,subprocess,json,sys,hashlib
P=Path('/root/autodl-tmp/A2_union_20261029');py='/root/miniconda3/bin/python';remaining={'00','03','08'}
for _ in range(720):
 for seq in list(remaining):
  R=P if seq=='00' else P/f'seq{seq}'
  target=R/'lidar_summary_s2_B_F.json'
  done_log=P/'finish_00.log' if seq=='00' else R/'lidar.log'
  token='COMPLETE_00_WITH_LIDAR' if seq=='00' else 'LIDAR_COMPLETE '+seq
  if not target.exists() or not done_log.exists() or token not in done_log.read_text(errors='replace'):continue
  if not (R/'independent_lidar_score_audit.json').exists():
   with (R/'independent_lidar_audit.log').open('xb') as f:subprocess.run([py,str(P/'audit_lidar_scores.py'),seq],stdout=f,stderr=subprocess.STDOUT,check=True)
  files=[]
  for g in ('evaluation_200_s*.json','eval_*_s*.json','lidar_*s*.json','independent_lidar_score_audit.json','pre_score_seal.json','pre_run_locks.json','run_formal.log'):
   files.extend(R.glob(g))
  dest=P/f'package_seq{seq}';dest.mkdir(exist_ok=False)
  for f in set(files):
   import shutil
   shutil.copy2(f,dest/f.name)
  for x in (R/'runs').glob('*_200_s*/combo_receipt.json'):
   shutil.copy2(x,dest/(x.parent.name+'_receipt.json'))
  print('PACKAGE_READY',seq,flush=True);remaining.remove(seq)
 if not remaining:
  (P/'stop_monitor').touch();(P/'ALL_SEQUENCES_SCORED').write_text(json.dumps({'status':'COMPLETE_DESCRIPTIVE_NOT_ADOPTED','seqs':['00','03','08']}));print('ALL_COMPLETE',flush=True);break
 time.sleep(30)
else:raise TimeoutError('6h watchdog timeout; remaining '+','.join(remaining))
