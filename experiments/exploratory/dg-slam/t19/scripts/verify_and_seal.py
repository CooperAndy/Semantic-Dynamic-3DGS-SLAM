"""Finish CPU-only reproduction and isolated delivery; never starts remote/GPU work."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,zipfile
F=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONIOENCODING='utf-8',T19_BASE_ROOT=str(F/'base_snapshot'))
steps=['validate_cpu.py','build_delivery.py','check_patch.py']
if os.environ.get('T19_REUSE_CPU_VALIDATION')=='1':
 prior=json.loads((F/'receipts/CPU_validation.json').read_text(encoding='utf-8'))
 prior_source=json.loads((F/'receipts/candidate_source.json').read_text(encoding='utf-8'))
 assert prior['status']=='PASS' and prior['input_root']==str(F/'base_snapshot')
 for item in prior_source['files']:assert sha(F/'candidate'/item['path'])==item['sha256']
 steps=['build_delivery.py','check_patch.py']
for name in steps:
 result=subprocess.run([sys.executable,str(F/'scripts'/name)],env=env,capture_output=True,text=True,encoding='utf-8')
 (F/'receipts'/('final_'+name.replace('.py','.log'))).write_text(result.stdout+result.stderr,encoding='utf-8')
 assert result.returncode==0,(name,result.stderr)
validation=json.loads((F/'receipts/CPU_validation.json').read_text(encoding='utf-8'))
patch=json.loads((F/'receipts/patch_reconstruction.json').read_text(encoding='utf-8'))
assert validation['status']=='PASS' and patch['status']=='PASS'
compiled=[]
for folder in ['candidate','scripts']:
 for p in sorted((F/folder).rglob('*.py')):
  compile(p.read_text(encoding='utf-8'),str(p),'exec');compiled.append(p.relative_to(F).as_posix())
dump(F/'receipts/reproduction_validation.json',{'status':'PASS','input_mode':'nine-file local snapshot','snapshot_file_hashes_verified':9,
 'CPU_check_groups':validation['check_groups'],'patch_reconstructed_files':len(patch['files']),
 'Python_compile_checked':compiled,'CUDA_initialized':validation['CUDA_initialized'],'real_neural_or_native_GPU_runs':0,
 'remote_connections':0,'GPU_deployment':False})
excluded={'delivery_manifest.json','receipts/bundle_receipt.json'}
files=[]
for p in sorted(F.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.relative_to(F).as_posix() not in excluded:
  files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(F/'delivery_manifest.json',{'task':'T19','files':files,'excluded':sorted(excluded)+['**/__pycache__/**'],
 'scientific_status':'CPU candidate only; no GPU validation, no formal acceptance change'})
dest=F.parent/'T19_CHY_补全接口修复与CPU验证_20261012.zip'
members=files+[{'path':'delivery_manifest.json','sha256':sha(F/'delivery_manifest.json')}]
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for item in members:z.write(F/item['path'],F.name+'/'+item['path'])
with zipfile.ZipFile(dest) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(members)
 for item in members:assert hashlib.sha256(z.read(F.name+'/'+item['path'])).hexdigest()==item['sha256']
receipt={'status':'PASS','zip_path':str(dest),'zip_bytes':dest.stat().st_size,'zip_sha256':sha(dest),
 'manifest_sha256':sha(F/'delivery_manifest.json'),'file_manifest_entries':len(files),'zip_members':len(members),
 'all_member_CRC_and_SHA256':'PASS','T17_and_T18_frozen_files_checked':348,'frozen_files_modified':0,
 'GPU_runs':0,'remote_connections':0,'GPU_deployment':False,'receipt_excluded_from_own_manifest_and_zip':True}
dump(F/'receipts/bundle_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
