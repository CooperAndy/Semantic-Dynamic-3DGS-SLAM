from pathlib import Path
import tarfile,hashlib,json
out=Path('/root/autodl-tmp/stage2_round17_delivery_v1');paths=[Path('/root/autodl-tmp/stage2_round14_scene06_20261031_v2/bridge'),Path('/root/autodl-tmp/stage2_round12_20261031_v1/p5_window')];dst=out/'p5_scene06_and_40frame_bridges_dereferenced.tar.gz'
with tarfile.open(dst,'w:gz',compresslevel=1,dereference=True) as t:
 for p in paths:t.add(p,arcname=str(p.relative_to('/root/autodl-tmp')))
count=0
with tarfile.open(dst,'r:gz') as t:
 for m in t:
  if m.isfile():
   f=t.extractfile(m)
   while f.read(4*1024*1024):pass
   count+=1
h=hashlib.sha256(dst.read_bytes()).hexdigest();d={'path':str(dst),'bytes':dst.stat().st_size,'sha256':h,'regular_members':count,'full_read_ok':True};(out/'bridge_supplement_manifest.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
