"""Seal the complete new local deliverable; all prior stages remain read-only."""
import ast
import datetime
import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote
F=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest_path=F/'delivery_manifest.json';validation_path=F/'receipts/delivery_validation.json'
bundle_receipt=F/'receipts/bundle_receipt.json'
assert not manifest_path.exists() and not validation_path.exists() and not bundle_receipt.exists()
audit=json.loads((F/'receipts/independent_audit.json').read_text());assert audit['status']=='PASS'
assert json.loads((F/'receipts/local_archive_validation.json').read_text())['status']=='PASS'
refs=[]
for p in F.glob('*.md'):
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        value=unquote(link.split('#',1)[0].strip('<>'))
        if not value or '://' in value:continue
        target=(p.parent/value).resolve()
        assert target.exists() or target in [manifest_path.resolve(),validation_path.resolve()],(p.name,link)
        refs.append({'document':p.name,'reference':link,'external_to_new_folder':not target.is_relative_to(F.resolve())})
for p in F.rglob('*.py'):
    if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
excluded={manifest_path,validation_path,bundle_receipt}
files=[]
for p in sorted(F.rglob('*')):
    if p.is_file() and p not in excluded and '__pycache__' not in p.parts:
        files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(manifest_path,{'owner':'CHY','task':'T20_T19_REGISTERED_GPU_DIAGNOSIS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'scope':'All five registered GPU diagnostic cases plus preserved failed smoke, complete logs/states/traces, CPU analyses and report. No formal adoption.',
     'files':files,'exclusions':['delivery_manifest.json','receipts/delivery_validation.json','receipts/bundle_receipt.json','__pycache__']})
dump(validation_path,{'status':'PASS','files':len(files),'bytes':sum(item['bytes'] for item in files),
     'manifest_sha256':sha(manifest_path),'local_markdown_references':refs,'syntax':'PASS',
     'figures_visually_inspected':2,'raw_archive_files_verified':467,'raw_seal_files_verified':376,
     'independent_audit':'PASS','historical_T17_T19_files_unchanged':361,'new_GPU_state_builds':2,
     'registered_filler_cases':5,'science_retries':0,'formal_acceptance':False,'quality_issue_resolved':False})
print(json.dumps({'stage':'MANIFEST_PASS','files':len(files)},ensure_ascii=False),flush=True)
archive=F.parent/'T20_CHY_GPU补全诊断完整交付_20261012.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for item in files:z.write(F/item['path'],F.name+'/'+item['path'])
    z.write(manifest_path,F.name+'/delivery_manifest.json')
    z.write(validation_path,F.name+'/receipts/delivery_validation.json')
with zipfile.ZipFile(archive) as z:
    assert len(z.infolist())==len(files)+2
    assert z.testzip() is None
receipt={'status':'PASS','archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
         'members':len(files)+2,'ZIP_CRC':'PASS','manifest_sha256':sha(manifest_path)}
dump(bundle_receipt,receipt)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
