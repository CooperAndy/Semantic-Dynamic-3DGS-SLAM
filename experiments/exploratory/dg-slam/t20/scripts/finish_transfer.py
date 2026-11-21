"""Remove only this task's incomplete downloads/byte-range intermediates after SHA PASS."""
import hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parents[1];BASE=F.parent.resolve()
r=json.loads((F/'receipts/parallel_transfer.json').read_text())
archive=BASE/r['archive']
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(2**20),b''):h.update(block)
assert archive.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256']
removed=[]
for filename in ['T20_CHY_GPU原始诊断_20261012.zip','T20_CHY_GPU原始诊断_20261012_stream.zip']:
    p=BASE/filename
    if p.exists():
        assert p.resolve().parent==BASE and not p.is_symlink() and p.stat().st_size<r['bytes']
        removed.append({'kind':'cancelled_partial_download','name':p.name,'bytes':p.stat().st_size})
        p.unlink()
parts=BASE/'.T20_raw_transfer_parts_20261012'
assert parts.resolve().parent==BASE and not parts.is_symlink()
assert sorted(p.name for p in parts.iterdir())==[f'part{i:02d}.bin' for i in range(8)]
for item in r['parts']:
    p=parts/f"part{item['index']:02d}.bin"
    assert p.resolve().parent==parts.resolve() and not p.is_symlink()
    assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    removed.append({'kind':'verified_byte_range_intermediate','name':p.name,'bytes':p.stat().st_size})
    p.unlink()
parts.rmdir()
receipt={'status':'PASS','complete_archive_retained':archive.name,'complete_archive_sha256':r['sha256'],
         'removed_only_own_transfer_intermediates':removed,'historical_scientific_files_removed':0}
(F/'receipts/transfer_cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'PASS','own_intermediates_removed':len(removed)}))
