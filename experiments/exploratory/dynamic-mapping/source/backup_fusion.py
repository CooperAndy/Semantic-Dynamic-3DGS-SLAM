from pathlib import Path
import tarfile,hashlib,json
b=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('06_*'))/'vehicle_only_v2';out=Path('/root/autodl-tmp/stage2_round17_delivery_v1');paths=[b/'dev_prob_fusion_baselines_20261005_v1',b/'dev_selected_features_20261005_v1'];dst=out/'probability_fusion_DEV_evidence.tar.gz'
with tarfile.open(dst,'w:gz',compresslevel=1,dereference=False) as t:
 for p in paths:t.add(p,arcname=p.name)
n=0
with tarfile.open(dst,'r:gz') as t:
 for m in t:
  if m.isfile():
   f=t.extractfile(m)
   while f.read(2*1024*1024):pass
   n+=1
h=hashlib.sha256(dst.read_bytes()).hexdigest();d={'path':str(dst),'bytes':dst.stat().st_size,'sha256':h,'regular_members':n,'full_read_ok':True};(out/'fusion_manifest.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
