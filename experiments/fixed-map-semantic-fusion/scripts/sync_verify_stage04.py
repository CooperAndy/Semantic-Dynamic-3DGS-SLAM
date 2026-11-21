import hashlib,json,tarfile,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/stage3_chy_20261006')
O=R/'04_系统整合与最终归档/final_delivery_20261007_v1'
A=R/'04_系统整合与最终归档/stage3_04_final_delivery_20261007_v1.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(A)=='c5397d6a68f1184506e44610b539a81f9a230bc14d757ab6e1c1309378ec702e'
with tarfile.open(A) as t:
 for m in t.getmembers():
  assert m.isfile() and (R/m.name).resolve().is_relative_to(R.resolve()),m.name
 t.extractall(R)
manifest=json.loads((O/'delivery_file_hashes.json').read_text())
for p,h in manifest.items():assert sha(R/p)==h,p
deps=json.loads((O/'bundle_dependencies.json').read_text())
for d in deps:assert sha(R/d['path'])==d['sha256'],d['path']
qa_path=O/'independent_delivery_QA.json'
sealed_qa=qa_path.read_bytes()
subprocess.run(['/root/miniconda3/bin/python',str(R/'scripts/verify_stage04_delivery.py')],check=True)
assert json.loads(qa_path.read_bytes())==json.loads(sealed_qa)
qa_path.write_bytes(sealed_qa) # Preserve sealed Windows line endings after Linux reproduction.
for p,h in manifest.items():assert sha(R/p)==h,p
result={'status':'PASS_REMOTE_04_DELIVERY','archive_sha256':sha(A),'sealed_files':len(manifest),'all_payload_hashes_match':True,'dependencies_verified':len(deps),'CPU_independent_QA':'PASS_INDEPENDENT_04_ALIGNMENT_AND_EXPORT','GPU_used':False}
(O/'server_delivery_sync.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result),flush=True)
