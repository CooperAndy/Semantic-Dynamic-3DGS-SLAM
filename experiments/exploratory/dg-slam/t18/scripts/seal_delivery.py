"""Read-only historical integrity audit; seal and verify isolated T18 delivery."""
from pathlib import Path
import hashlib,json,zipfile,shutil
from datetime import datetime,timezone
F=Path(__file__).resolve().parents[1]
B=F.parent/'CHY_T17_NO_GS_vs_ID完整轨迹_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
scope=json.loads((F/'CPU_scope.json').read_text(encoding='utf-8'))
assert sha(B/'delivery_manifest.json')==scope['T17_manifest_sha256']
old=json.loads((B/'delivery_manifest.json').read_text(encoding='utf-8'))
for entry in old['files']:assert sha(B/entry['path'])==entry['sha256'],entry['path']
lock=json.loads((B/'execution_lock.json').read_text(encoding='utf-8'))
matches=[(p,h) for p,h in lock['runtime_immutable_files'].items() if p.endswith('/native/droid/src/droid_kernels.cu')]
assert len(matches)==1 and matches[0][1]==sha(F/'source_reference/droid_kernels.cu')
shutil.copyfile(B/'execution_lock.json',F/'source_reference/T17_execution_lock.json')
native_receipt=json.loads((F/'receipts/native_source_reference.json').read_text(encoding='utf-8'))
native_receipt.update(T17_execution_lock_entry=matches[0][0],T17_execution_lock_hash_match=True,T17_execution_lock_sha256=sha(B/'execution_lock.json'))
dump(F/'receipts/native_source_reference.json',native_receipt)
dump(F/'receipts/T17_final_integrity.json',{'status':'PASS','files_checked':len(old['files']),'manifest_sha256':sha(B/'delivery_manifest.json'),'T17_frozen_files_unchanged':True,'original_files_modified':0,'checked_at_UTC':datetime.now(timezone.utc).isoformat()})
excluded={'delivery_manifest.json','receipts/bundle_receipt.json'}
files=[]
for p in sorted(F.rglob('*')):
 if p.is_file() and p.relative_to(F).as_posix() not in excluded:
  files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(F/'delivery_manifest.json',{'task':'T18','files':files,'excluded':sorted(excluded),'scope':'CPU-only existing-result diagnosis, no GPU, no formal acceptance change'})
dest=F.parent/'T18_CHY_CPU离线补全诊断_20261012.zip'
members=files+[{'path':'delivery_manifest.json','bytes':(F/'delivery_manifest.json').stat().st_size,'sha256':sha(F/'delivery_manifest.json')}]
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for entry in members:z.write(F/entry['path'],F.name+'/'+entry['path'])
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(members)
 for entry in members:assert hashlib.sha256(z.read(F.name+'/'+entry['path'])).hexdigest()==entry['sha256']
receipt={'status':'PASS','zip_path':str(dest),'zip_bytes':dest.stat().st_size,'zip_sha256':sha(dest),'file_manifest_entries':len(files),'manifest_sha256':sha(F/'delivery_manifest.json'),'zip_members':len(members),'CRC_and_every_member_SHA256':'PASS','T17_281_files_unchanged':len(old['files'])==281,'GPU_or_SLAM_runs':0,'remote_connections':0,'receipt_excluded_from_own_manifest_and_zip':True}
dump(F/'receipts/bundle_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
