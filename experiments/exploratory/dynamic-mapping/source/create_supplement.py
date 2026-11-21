from pathlib import Path
import json,tarfile,hashlib,os
out=Path('/root/autodl-tmp/stage2_round17_delivery_v1');base=Path('/root/autodl-tmp');P=base/'SplaTAM_e6b/experiments/KITTI';stage1=['seq00_e3_sgbm_holdout_200_0','seq00_e4_raft_holdout_200_0','seq07_e3_sgbm_holdout_200_0','seq07_e6a_confhard_holdout_200_0'];groups={'stage1_heldout_supplement.tar.gz':[P/n for n in stage1], 'stage2_lifecycle_evidence.tar.gz':[base/'stage2_round14_scene06_20261031_v2',base/'stage2_round15_online_eval_20261031_v1',base/'stage2_round12_p5_integrated_v1',base/'stage2_round13_metadata_injection_v1'], 'p5_40frame_and_source.tar.gz':[base/'stage2_round12_20261031_v1/p5_snapshot',base/'stage2_round12_20261031_v1/maps',base/'stage2_round12_20261031_v1/input_freeze.json',base/'stage2_round12_20261031_v1/source',base/'stage2_round12_20261031_v1/source_sha256.txt']};result=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,paths in groups.items():
 for p in paths:
  if not p.exists():raise FileNotFoundError(p)
 dst=out/name
 with tarfile.open(dst,'w:gz',compresslevel=1,dereference=False) as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(base)),recursive=True)
 count=0;total=0
 with tarfile.open(dst,'r:gz') as t:
  for m in t:
   count+=1
   if m.isfile():
    f=t.extractfile(m)
    while True:
     b=f.read(4*1024*1024)
     if not b:break
     total+=len(b)
 result.append({'path':str(dst),'bytes':dst.stat().st_size,'sha256':sha(dst),'members':count,'uncompressed_regular_bytes':total,'archive_full_read_ok':True,'source_paths':[str(p) for p in paths]})
(out/'supplement_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
