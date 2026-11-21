from common import *
import shutil,tarfile,time,hashlib
start=time.monotonic();assert json.loads((E/'controller_v2.json').read_text())['status']=='COMPLETE_AUDITED_PAST_ONLY_INPUT_REPLAY';audit=json.loads((R/'receipts/audit.json').read_text());assert audit['status']=='PASS_1000_PAST_ONLY_CONFIDENCE_INPUT_REPLAY'
reg=json.loads((R/'registration.json').read_text());am=json.loads((R/'amendment_v2.json').read_text())
for locks in [reg['source_files'],am['source_files']]:
 for name,digest in locks.items():assert sha(R/name)==digest
op=R/'operators_final';op.mkdir()
for p in E.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.log']:shutil.copy2(p,op/p.name)
parents=[]
for row in reg['rows']:
 for k,x in row['inputs'].items():
  p=Path(x['path']);assert sha(p)==x['sha256']
  if k!='pose':parents.append(dict(path='data_pipeline/'+p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=x['sha256']))
fp=[]
for rel in ['registration.json','receipts/audit.json','output/external_causal_P2_c2w.txt']:
 p=F/rel;fp.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
files=[p for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts];manifest=dict(root='G19_PAST',files=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],parents=[dict(root='G14',files=parents),dict(root='G19_CAUSAL',files=fp)],scope='Past-only confidence with causal returned poses;cached input replay,not joint3DGS/quality/FPS adoption',quality_adoption=False);mp=E/'MANIFEST.json';dump(mp,manifest);target=E/'G19_past_only_confidence_science.tar.gz';assert not target.exists()
with tarfile.open(target,'w:gz',compresslevel=1,dereference=True) as t:
 for p in files:t.add(p,arcname='G19_PAST/'+p.relative_to(R).as_posix(),recursive=False)
 t.add(mp,arcname='MANIFEST.json',recursive=False)
remaining={'G19_PAST/'+x['path']:x for x in manifest['files']};count=0
with tarfile.open(target,'r:gz') as t:
 for m in t:
  assert m.isfile();h=hashlib.sha256();n=0;f=t.extractfile(m)
  for block in iter(lambda:f.read(2**20),b''):h.update(block);n+=len(block)
  if m.name=='MANIFEST.json':assert h.hexdigest()==sha(mp)
  else:
   x=remaining.pop(m.name);assert n==x['bytes'] and h.hexdigest()==x['sha256']
  count+=1
assert not remaining;dump(E/'archive_receipt.json',dict(status='PASS_SERVER_WHOLE_EVERY_MEMBER',path=str(target),bytes=target.stat().st_size,sha256=sha(target),members=count,parent_files=sum(len(x['files']) for x in manifest['parents']),seconds=time.monotonic()-start));print((E/'archive_receipt.json').read_text())
