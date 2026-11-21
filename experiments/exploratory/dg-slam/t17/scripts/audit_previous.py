from pathlib import Path
import datetime,hashlib,json
F=Path(__file__).resolve().parents[1];B=F.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for block in iter(lambda:s.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
out={}
for name,manifest in [('CHY_T16_渲染分支诊断与修复_20261012','delivery_manifest.json'),('CHY_T15_DEV同图回放_20261012','delivery_manifest.json'),('CHY_正式Tracking采用验收_20261012','delivery_manifest.json'),('DGSLAM_队友交接_截至T10_20261012/DGSLAM_队友交接_截至T10_20261012','文件清单_SHA256.json')]:
 root=B/name;M=json.loads((root/manifest).read_text(encoding='utf-8'))
 for row in M['files']:
  p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
 out[name]={'files_unchanged':len(M['files']),'manifest_sha256':sha(root/manifest)}
receipt={'status':'PASS','checked_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'historical_artifacts':out}
(F/'receipts/historical_integrity.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
