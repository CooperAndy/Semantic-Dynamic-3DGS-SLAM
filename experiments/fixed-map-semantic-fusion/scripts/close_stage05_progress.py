import json,hashlib,tarfile,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'05_条件轻量候选与历史诊断';p=R/'执行进度与下一步.md';mp=D/'completion_05_file_hashes.json'
old=D/'completion_05_file_hashes_original_v2.json';assert not old.exists();shutil.copy2(mp,old)
hashes=json.loads(mp.read_text(encoding='utf-8-sig'));hashes[p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();mp.write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
meta=D/'progress_correction.json';meta.write_text(json.dumps({'change':'Replace stale 04 CPU-only/05-not-started summary with latest completed GPU/05 status','scientific_results_changed':False,'original_completion_archive_preserved':True},indent=2)+'\n',encoding='utf-8')
A=D/'stage3_05_progress_sync_20261007_v1.tar.gz';assert not A.exists()
with tarfile.open(A,'w:gz') as t:
 for q in [p,mp,old,meta,Path(__file__)]:t.add(q,arcname=q.relative_to(R).as_posix())
result={'sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'bytes':A.stat().st_size,'files':5};(D/'progress_sync_archive_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
