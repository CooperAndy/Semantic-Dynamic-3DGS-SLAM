"""Seal reviewed display assets and export, retaining large-data dependency bundles."""
import hashlib,json,tarfile,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'04_系统整合与最终归档/final_delivery_20261007_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ['independent_delivery_QA.json','browser_QA.json']:assert json.loads((O/name).read_text(encoding='utf-8'))['status'].startswith('PASS_')
d=json.loads((O/'viewer_data.json').read_text(encoding='utf-8'))
for case in d['cases']:
 assert (O/case['GT']).is_file()
 if case['dataset'].startswith('2013'):assert 'RGB' not in case
 for group,paths in case['groups'].items():
  for k in ['semantic','score','error']:assert (O/paths[k]).is_file()
assert not list(O.rglob('*.part'))
dump(O/'completion_status.json',{'status':'COMPLETE_04_CPU_SYSTEM_INTEGRATION_AND_LOCAL_QA','prerequisites':'PASS_WITH_DECLARED_QUALITY_LIMITS','default_method':'S0','views':15,'method_view_comparisons':81,'full_KITTI08_export_Gaussians':907488,'large_dependency_archives':4,'new_GPU_or_tuning':False,'algorithmic_quality_PASS':False,'research_extension_status':'PAUSED_AT_USER_REQUEST','server_sync_receipt':'separate server_delivery_sync.json after receipt validation'})
shutil.copy2(Path(__file__),O/'repro_sources/seal_stage04_delivery.py')
files=[p for p in O.rglob('*') if p.is_file() and not any(k in p.parts for k in ['runtime','__pycache__']) and p.name not in ['delivery_file_hashes.json','archive_receipt.json','server_delivery_sync.json']]
for p in files:assert p.resolve().is_relative_to(R.resolve())
dump(O/'delivery_file_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(files)});files.append(O/'delivery_file_hashes.json')
for n in ['build_stage04_delivery.py','verify_stage04_delivery.py','seal_stage04_delivery.py']:files.append(R/'scripts'/n)
archive=O.parent/'stage3_04_final_delivery_20261007_v1.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
 for p in sorted(files):tar.add(p,arcname=p.relative_to(R).as_posix(),recursive=False)
receipt={'status':'SEALED_LOCAL_QA_COMPLETE','path':archive.relative_to(R).as_posix(),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files),'large_archives_referenced_not_duplicated':True};dump(O/'archive_receipt.json',receipt);print(json.dumps(receipt),flush=True)
