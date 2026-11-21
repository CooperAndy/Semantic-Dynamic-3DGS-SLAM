"""Package Stage 3 CPU artifacts with per-file hashes, excluding base and cache."""
import hashlib,json,tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix not in ['.tgz','.pyc'] and p.name not in ['sync_manifest.json','server_sync_validation.json','SHA256SUMS.txt']
        and not (p.name=='params.npz' and p.parent.name=='base_map')]
    records=[{'relative_path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    manifest={'files':records,'base_separately_uploaded':{'relative_path':'inputs/base_map/params.npz','sha256':sha(root/'inputs/base_map/params.npz')}}
    target=root/'04_系统整合与最终归档/sync_manifest.json';target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=root/'04_系统整合与最终归档/stage3_cpu_20261006_v1.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[target]:tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    lines=[f"{r['sha256']}  {r['relative_path']}" for r in records]
    lines.append(f"{manifest['base_separately_uploaded']['sha256']}  inputs/base_map/params.npz")
    (root/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(files),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)}))

if __name__=='__main__':main()
