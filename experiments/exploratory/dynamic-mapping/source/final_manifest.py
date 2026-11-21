import hashlib,json
from pathlib import Path
p=Path('/root/autodl-tmp/stage2_round17_delivery_v1');rows=[]
for f in sorted(p.iterdir()):
 if not f.is_file() or f.name in ('download_SHA256.txt','download_manifest.json'):continue
 h=hashlib.sha256()
 with f.open('rb') as file:
  for b in iter(lambda:file.read(8*1024*1024),b''):h.update(b)
 rows.append({'name':f.name,'bytes':f.stat().st_size,'sha256':h.hexdigest()})
(p/'download_SHA256.txt').write_text(''.join(f"{x['sha256']}  {x['name']}\n" for x in rows));(p/'download_manifest.json').write_text(json.dumps({'files':rows,'total_bytes':sum(x['bytes'] for x in rows),'note':'top-level files only; restore_check is temporary and excluded'},indent=2));print('files',len(rows),'total bytes',sum(x['bytes'] for x in rows))
