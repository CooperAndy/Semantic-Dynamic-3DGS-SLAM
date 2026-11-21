from pathlib import Path
import ast,datetime,hashlib,json,zipfile
F=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for block in iter(lambda:s.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
assert load(F/'receipts/registered_launch.json')['status']=='ALL_SIX_TERMINAL'
assert load(F/'receipts/independent_audit.json')['status']=='PASS'
assert load(F/'receipts/historical_integrity.json')['status']=='PASS'
assert load(F/'receipts/server_final_state.json')['status']=='COMPLETED_IDLE'
L=load(F/'execution_lock.json');R=load(F/'protocol.json')['remote_root']
assert sha(F/'protocol.json')==L['protocol_sha256']
repair=load(F/'receipts/CPU_eval_precision_repair.json')
assert sha(F/'scripts/evaluate_cpu64.py')==repair['wrapper_sha256']
assert sha(F/'scripts/evaluate.py')==repair['frozen_evaluator_sha256']==L['evaluator_sha256']
verified=0
for path,h in L['runtime_immutable_files'].items():
 if path.startswith(R+'/'):
  p=F/path[len(R)+1:];assert sha(p)==h,str(p);verified+=1
for rel,h in load(F/'receipts/raw_seal.json')['files'].items():assert sha(F/rel)==h,rel
for p in F.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
receipt={'status':'PASS','new_locked_runtime_files_verified_local':verified,'raw_seal_verified':True,'historical_files_unchanged':True,'report_figure_visually_checked':True,'six_registered_runs_once':True,'all_six_terminal_before_CPU_GT':True,'scope':'Exposed complete DEV0:200, no new formal acceptance','dependency_note':'Unchanged native sm120 environment, raw RGB/depth/masks and weights remain in the preserved T14/teammate/data archives; execution_lock binds exact hashes. New source tree and workers, inputs, complete trajectories, map snapshots, camera traces, logs and CPU evaluation references are in this package.'}
(F/'receipts/delivery_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
files=[]
for p in sorted(F.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.relative_to(F).as_posix() not in ['delivery_manifest.json','receipts/bundle_receipt.json']:
  files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'stage':'T17_COMPLETE_DEV_TRAJECTORY_COMPARISON','sealed_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'bytes':sum(r['bytes'] for r in files),'excluded':['delivery_manifest.json','receipts/bundle_receipt.json','__pycache__']}
(F/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=F.parent/'T17_CHY_NO_GS与ID完整轨迹对照_20261012.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for row in files:z.write(F/row['path'],arcname=F.name+'/'+row['path'])
 z.write(F/'delivery_manifest.json',arcname=F.name+'/delivery_manifest.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for row in files:assert hashlib.sha256(z.read(F.name+'/'+row['path'])).hexdigest()==row['sha256']
 assert z.read(F.name+'/delivery_manifest.json')==(F/'delivery_manifest.json').read_bytes()
receipt={'status':'ARCHIVE_CRC_AND_ALL_MEMBER_SHA256_PASS','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_sha256':sha(F/'delivery_manifest.json'),'manifest_files':len(files),'zip_members':len(files)+1,'excluded_self_receipt':True}
(F/'receipts/bundle_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
