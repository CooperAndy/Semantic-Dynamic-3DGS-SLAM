"""Constrained extraction: only regular files in Stage3, protect differing files."""
import hashlib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];archive=R/'02_基础因子DEV与Oracle诊断/semantickitti08_sparse_GT_complete_20261007_v1.tar.gz';expected='21d3e01460428496bfaa2070937eb8084cdfe0d044f5b6f6dfc40ce3a2dbf5e6'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert sha(archive)==expected
with tarfile.open(archive,'r:gz') as tar:
 for i,member in enumerate(tar):
  assert member.isfile() and member.size<64*1024*1024,member.name;dst=(R/member.name).resolve();assert dst.is_relative_to(R.resolve());stream=tar.extractfile(member);data=stream.read();assert len(data)==member.size
  if dst.exists():assert sha(dst)==hashlib.sha256(data).hexdigest(),'preserve differing existing file: '+str(dst)
  else:dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
print('PASS_ARCHIVE_SHA256_AND_CONSTRAINED_EXTRACTION',i+1,flush=True)
