"""Seal only local T16 preparation; never claim the blocked GPU repair is complete."""
from pathlib import Path
import ast,hashlib,json,re,zipfile
F=Path(__file__).resolve().parents[1];B=F.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def verify(root,name):
 manifest=json.loads((root/name).read_text(encoding='utf-8'))
 for r in manifest['files']:
  p=root/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
 return len(manifest['files'])
for p in (F/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
lock=json.loads((F/'trace_lock.json').read_text(encoding='utf-8'))
assert sha(F/'trace_protocol.json')==lock['protocol_sha256']
assert sha(F/'scripts/trace_runtime.py')==lock['script_sha256']
assert json.loads((F/'receipts/compositor_selfcheck.json').read_text(encoding='utf-8'))['status']=='PASS'
assert not list((F/'traces').iterdir()) and not (F/'trace_result.json').exists()
links=re.findall(r'\]\(([^)]+)\)',(F/'README_修复进度.md').read_text(encoding='utf-8'))
for link in links:assert (F/link).is_file(),link
native=B/'CHY_正式Tracking采用验收_20261012/archive/native/raster'
source_hashes={str(p.relative_to(native)).replace('\\','/'):sha(p) for p in [native/'cuda_rasterizer/rasterizer_impl.cu',native/'cuda_rasterizer/rasterizer_impl.h',native/'cuda_rasterizer/forward.cu',native/'diff_gaussian_rasterization_depth_pose/__init__.py']}
dump(F/'receipts/comparison_lock.json',{'CPU_script_sha256':sha(F/'scripts/compare_branches.py'),'native_layout_source_hashes':source_hashes,
 'reconstruction_gates':{'RGB_max_abs':1e-4,'depth_max_abs_m':1e-3},'selected_tiles':8,'scope':'Trace diagnosis only, no optimizer acceptance or formal-quality change.'})
unchanged={}
for folder,manifest in [('CHY_T15_DEV同图回放_20261012','delivery_manifest.json'),('CHY_正式Tracking采用验收_20261012','delivery_manifest.json'),('DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012','文件清单_SHA256.json')]:unchanged[folder]=verify(B/folder,manifest)
dump(F/'receipts/preparation_validation.json',{'status':'LOCAL_PREPARATION_PASS_GPU_REPAIR_PENDING','old_files_unchanged':unchanged,
 'python_syntax':'PASS','closed_form_compositor':'PASS','markdown_links':len(links),'new_GPU_runs':0,'optimization_replays':0,
 'blocker':'SSH endpoint refused; user confirmed server powered off. No native buffer captures or real-data repair validation yet.'})
files=[{'path':str(p.relative_to(F)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(F.rglob('*')) if p.is_file() and p.name not in ['preparation_manifest.json','bundle_receipt.json']]
dump(F/'preparation_manifest.json',{'owner':'CHY','scope':'T16 local preparation only; blocked remote work remains incomplete','files':files})
out=B/'T16_CHY_分支修复准备_20261012.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for row in files:z.write(F/row['path'],row['path'])
 z.write(F/'preparation_manifest.json','preparation_manifest.json')
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for row in files:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
dump(F/'receipts/bundle_receipt.json',{'status':'PASS','zip':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'members':len(files)+1})
print(json.dumps({'local_preparation':'PASS','GPU_repair':'PENDING_SERVER_OFF','old_files_unchanged':unchanged,'zip_bytes':out.stat().st_size},ensure_ascii=False))
