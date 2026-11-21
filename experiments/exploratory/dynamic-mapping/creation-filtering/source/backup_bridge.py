from pathlib import Path
import tarfile,hashlib,json
src=Path('/root/autodl-tmp/stage2_round16_scene02_v1/bridge');out=Path('/root/autodl-tmp/stage2_round16_backup_v1/scene02_bridge_dereferenced.tar.gz')
with tarfile.open(out,'w:gz',compresslevel=1,dereference=True) as t:t.add(src,arcname='bridge')
h=hashlib.sha256(out.read_bytes()).hexdigest();print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':h,'note':'bridge links dereferenced; can restore independently'}))
