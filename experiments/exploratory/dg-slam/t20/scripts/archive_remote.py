"""Read/verify frozen outputs, then create a complete additive raw archive."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import zipfile

F=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert json.loads((F/'receipts/launch_v2.json').read_text())['status']=='ALL_FIVE_DIAGNOSTICS_COMPLETE'
for path,h in json.loads((F/'execution_lock_v2.json').read_text())['runtime_immutable_files'].items():
    assert sha(Path(path))==h,path
seal=json.loads((F/'receipts/raw_seal.json').read_text())
for rel,h in seal['files'].items():assert sha(F/rel)==h,rel
workers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
assert not workers,workers
dump(F/'receipts/server_after_diagnostics.json',{'status':'ALL_FIVE_COMPLETE_GPU_IDLE',
     'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
     'GPU_workers':workers,'disk_free_bytes':shutil.disk_usage(F).free,
     'runtime_files_reverified':2094,'formal_runs':0,'server_shutdown':False})
excluded={'remote_manifest.json','remote_bundle_receipt.json','T20_GPU_raw_20261012.zip'}
files=[]
for p in sorted(F.rglob('*')):
    if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts:
        files.append({'path':p.relative_to(F).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest=F/'remote_manifest.json'
dump(manifest,{'status':'SEALED_RAW_ARCHIVE','files':files,'raw_seal_sha256':sha(F/'receipts/raw_seal.json')})
archive=F/'T20_GPU_raw_20261012.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for item in files:z.write(F/item['path'],item['path'])
    z.write(manifest,'remote_manifest.json')
receipt={'status':'PASS','archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
         'listed_files':len(files),'ZIP_members':len(files)+1,'manifest_sha256':sha(manifest)}
dump(F/'remote_bundle_receipt.json',receipt)
print(json.dumps(receipt))
