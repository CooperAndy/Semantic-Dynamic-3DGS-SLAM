from pathlib import Path
import tarfile,json,hashlib
out=Path('/root/autodl-tmp/stage2_round17_delivery_v1');b=Path('/root/autodl-tmp/SplaTAM_e6b');stage=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('07_*'));paths=[b/x for x in ['scripts','utils','datasets','configs']]+[stage/'dev_coverage_20261001_v2/base_mapping_config.py',stage/'dev_coverage_20261001_v2/base_train_config.py'];assert all(p.exists() for p in paths);target=out/'stage1_and_p5_config_source.tar.gz'
with tarfile.open(target,'w:gz',compresslevel=1,dereference=False) as t:
 for p in paths:t.add(p,arcname=str(p.relative_to('/root/autodl-tmp')))
n=0
with tarfile.open(target,'r:gz') as t:
 for m in t:
  if m.isfile():
   f=t.extractfile(m)
   while f.read(2*1024*1024):pass
   n+=1
h=hashlib.sha256(target.read_bytes()).hexdigest();d={'path':str(target),'bytes':target.stat().st_size,'sha256':h,'regular_members':n,'full_read_ok':True,'caveat':'source snapshot current state, not independently verified all files are exact historical experiment code; configs import original absolute paths'};(out/'source_supplement_manifest.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
