from pathlib import Path
import tarfile,hashlib,json,time
out=Path('/root/autodl-tmp/stage2_round16_backup_v1');out.mkdir(exist_ok=False);items=[('scene02_run',Path('/root/autodl-tmp/stage2_round16_scene02_v1')),('scene06_audit',Path('/root/autodl-tmp/stage2_round16_creation_audit_v1')),('gaussian_export',Path('/root/autodl-tmp/stage2_round16_gaussian_export_v1'))];rows=[]
for name,src in items:
 dst=out/(name+'.tar.gz');t=time.perf_counter()
 with tarfile.open(dst,'w:gz',compresslevel=1,dereference=False) as tar:tar.add(src,arcname=src.name,recursive=True)
 sha=hashlib.sha256();
 with dst.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
 rows.append({'path':str(dst),'bytes':dst.stat().st_size,'sha256':sha.hexdigest(),'seconds':time.perf_counter()-t,'note':'Scene02 bridge uses symlinks to historical RGB/depth; retain original referenced files or dereference when moving' if name=='scene02_run' else 'archive'})
(out/'backup_manifest.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
