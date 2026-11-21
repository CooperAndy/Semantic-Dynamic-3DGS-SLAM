from pathlib import Path
import json,hashlib,subprocess,sys
P=Path('/root/autodl-tmp/A2_union_20261029'); seq=sys.argv[1];assert seq in ('03','08'); R=P/f'seq{seq}'
O=Path('/root/autodl-tmp/A2_stereo_robust_geometry/recheck08_v1')
old=Path('/root/autodl-tmp/A2_stereo_robust_geometry/seq03_recheck_v1/seed_0/lidar_score_config_200.json') if seq=='03' else O/'lidar_score_config_200.json'
ref=json.loads(old.read_text());py='/root/miniconda3/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
seal=json.loads((R/'pre_score_seal.json').read_text());assert seal['status']=='MAPS_SEALED_BEFORE_SCORE'
for x in seal['items']:assert sha(x['path'])==x['sha256']
for seed in (0,1,2):
 for left,right in (('B','A'),('F','AF'),('B','AF'),('B','F')):
  arms=[left,right];frames=[]
  for j,fr in enumerate(ref['frames']):
   f=19+20*j; src=fr['reference'];assert sha(src['path'])==src['sha256']
   renders={}
   for a in arms:
    p=R/f'render_s{seed}'/f'{f:03d}_{a}.npz';renders[a]={'path':str(p),'sha256':sha(p)}
   frames.append({'source_frame':fr['source_frame'],'reference':src,'renders':renders})
  cfg=R/f'lidar_config_s{seed}_{left}_{right}.json';cfg.write_text(json.dumps({'arms':arms,'frames':frames,'scope':'post-map sparse LiDAR paired common support, exposed DEV'},indent=2))
  score=R/f'lidar_score_s{seed}_{left}_{right}.json';summary=R/f'lidar_summary_s{seed}_{left}_{right}.json'
  subprocess.run([py,str(O/'source/score_lidar_renders.py'),'--config',str(cfg),'--out',str(score)],check=True)
  subprocess.run([py,str(O/'aggregate_lidar_score_v1.py'),'--infile',str(score),'--out',str(summary)],check=True)
print('LIDAR_COMPLETE',seq)
