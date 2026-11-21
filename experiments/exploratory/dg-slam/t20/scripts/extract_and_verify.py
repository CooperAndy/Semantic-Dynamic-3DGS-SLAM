"""Verify downloaded ZIP and safely extract without overwriting local history."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path,PurePosixPath

F=Path(__file__).resolve().parents[1]
archive=F.parent/'T20_CHY_GPU原始诊断_20261012_parallel.zip'
receipt=json.loads((F/'receipts/remote_bundle_receipt.json').read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
with zipfile.ZipFile(archive) as z:
    assert len(z.infolist())==receipt['ZIP_members']
    names=set()
    for info in z.infolist():
        pure=PurePosixPath(info.filename)
        assert not pure.is_absolute() and '..' not in pure.parts and ':' not in info.filename
        assert info.filename not in names
        names.add(info.filename)
        target=(F/Path(*pure.parts)).resolve();assert target.is_relative_to(F.resolve())
        if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
        if target.exists():
            with z.open(info) as stream:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(2**20),b''):h.update(block)
            assert target.stat().st_size==info.file_size and sha(target)==h.hexdigest(),str(target)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination,2**20)
manifest=json.loads((F/'remote_manifest.json').read_text(encoding='utf-8'))
assert sha(F/'remote_manifest.json')==receipt['manifest_sha256']
for item in manifest['files']:
    p=F/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
seal=json.loads((F/'receipts/raw_seal.json').read_text(encoding='utf-8'))
for rel,h in seal['files'].items():assert sha(F/rel)==h,rel
result={'status':'PASS','archive_sha256':receipt['sha256'],'archive_bytes':receipt['bytes'],
        'remote_manifest_verified_files':len(manifest['files']),'raw_seal_verified_files':len(seal['files']),
        'historical_or_local_files_overwritten':0,'GT_scoring_started_before_validation':False}
(F/'receipts/local_archive_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
