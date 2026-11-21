"""Seal T16 analysis inputs, evidence and archive without modifying older stages."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,zipfile
F=Path(__file__).resolve().parents[1];B=F.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for block in iter(lambda:s.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
old=F/'receipts/bundle_receipt.json'
if old.exists() and not (F/'receipts/preparation_bundle_receipt.json').exists():
 shutil.copy2(old,F/'receipts/preparation_bundle_receipt.json')
T15=B/'CHY_T15_DEV同图回放_20261012'
copies=[]
sources=[(p,'inputs/'+p.name) for p in (T15/'inputs').glob('*') if p.is_file()]
sources += [(p,'dependencies/T15/scripts/'+p.name) for p in (T15/'scripts').glob('*.py')]
sources += [(T15/name,'dependencies/T15/'+name) for name in ['protocol.json','execution_lock.json','delivery_manifest.json']]
P=B/'DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012/阶段3/DGSLAM_方案核查_20261011/t02_receipts'
sources += [(P/'evaluation_reference.npz','evaluation_inputs/evaluation_reference.npz'),(P/'04_reproduction/T02_200_DIAG_20261011_195357/keyframe_pose_audit.jsonl','evaluation_inputs/keyframe_pose_audit.jsonl')]
for src,rel in sources:
 dst=F/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 assert sha(src)==sha(dst)
 copies.append({'source':str(src),'backup':rel,'sha256':sha(dst)})
write(F/'receipts/backup_input_sources.json',{'copied_after_raw_seal':True,'evaluation_inputs_local_CPU_only':True,'files':copies,'runtime_dependencies':'Original T14 native/environment and T15 Context remain external read-only dependencies; originals preserved. Archive contains raw buffers, original snapshot inputs, evaluation references, scripts and receipts. Deployment paths remain those registered; restore original directory layout for GPU replay.'})
for item in json.loads((F/'trace_result.json').read_text(encoding='utf-8'))['captures']:assert sha(F/item['path'])==item['sha256']
seal=json.loads((F/'receipts/replay_raw_seal.json').read_text(encoding='utf-8'))
for rel,h in seal['files'].items():assert sha(F/rel)==h
remote=json.loads((F/'receipts/server_final_state_T16.json').read_text(encoding='utf-8'))
for rel,h in remote['registered_files_sha256'].items():assert sha(F/rel)==h,rel
py=list(F.rglob('*.py'))
for p in py:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
write(F/'receipts/delivery_validation.json',{'status':'PASS','trace_captures_hash_verified':len(json.loads((F/'trace_result.json').read_text(encoding='utf-8'))['captures']),'raw_replay_files_hash_verified':len(seal['files']),'python_sources_AST_valid':len(py),'remote_registration_hashes_match_local':True,'independent_audit':json.loads((F/'receipts/completed_independent_audit.json').read_text(encoding='utf-8'))['status'],'input_backups_verified':len(copies),'figure_visually_checked':True,'complete_DEV_windows':0,'formal_runs':0})
report=F/'T16_分支验证修复与8次回放结果.md'
extra='\n## 最终交付\n\n[交付清单](delivery_manifest.json)、[封存校验](receipts/delivery_validation.json)、[服务器最终状态](receipts/server_final_state_T16.json)。快照输入、原T15脚本及CPU评价参考已复制进本包；GT参考仅保存在本地evaluation_inputs，未用于GPU优化。CUDA原生依赖仍使用已归档的T14环境，详见[输入与依赖说明](receipts/backup_input_sources.json)。\n\n原preparation_manifest.json及preparation_bundle_receipt.json只代表离线准备阶段；最终有效交付清单为delivery_manifest.json。ZIP及ZIP回执放在清单之外，避免自引用。\n'
if '## 最终交付' not in report.read_text(encoding='utf-8'):report.write_text(report.read_text(encoding='utf-8')+extra,encoding='utf-8')
files=[]
for p in sorted(F.rglob('*')):
 if p.is_file() and p.relative_to(F).as_posix() not in ['delivery_manifest.json','receipts/bundle_receipt.json'] and '__pycache__' not in p.parts:
  files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'stage':'T16_COMPLETE_DEV_NEGATIVE','sealed_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'excludes':['delivery_manifest.json','receipts/bundle_receipt.json','__pycache__'],'bytes':sum(x['bytes'] for x in files)}
write(F/'delivery_manifest.json',manifest)
archive=B/'T16_CHY_分支修复与8次回放_20261012.zip'
assert not archive.exists(),'Refuse to overwrite completed archive'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for row in files:z.write(F/row['path'],arcname=F.name+'/'+row['path'])
 z.write(F/'delivery_manifest.json',arcname=F.name+'/delivery_manifest.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for row in files:assert hashlib.sha256(z.read(F.name+'/'+row['path'])).hexdigest()==row['sha256']
 assert z.read(F.name+'/delivery_manifest.json')==(F/'delivery_manifest.json').read_bytes()
write(F/'receipts/bundle_receipt.json',{'status':'ARCHIVE_CRC_AND_EVERY_MEMBER_SHA256_PASS','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_sha256':sha(F/'delivery_manifest.json'),'manifest_file_count':len(files),'zip_members':len(files)+1,'original_preparation_zip_preserved':True,'receipt_excluded_from_manifest':True})
print((F/'receipts/bundle_receipt.json').read_text(encoding='utf-8'))
