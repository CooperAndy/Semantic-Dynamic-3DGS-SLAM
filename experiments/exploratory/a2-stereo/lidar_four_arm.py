import json,hashlib,subprocess,sys
from pathlib import Path
R=Path('/root/autodl-tmp/A2_union_20261029'); O=Path('/root/autodl-tmp/A2_stereo_robust_geometry/recheck08_v1'); arms=['B','A','F','AF'];ref=json.loads((R/'lidar00_projected/manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for s in (0,1,2):
 frames=[]
 for row in ref['frames']:
  f=row['source_frame'];p=R/'lidar00_projected'/row['filename'];assert sha(p)==row['sha256'];renders={}
  for a in arms:
   p2=R/f'render_s{s}'/f'{f:03d}_{a}.npz';renders[a]={'path':str(p2),'sha256':sha(p2)}
  frames.append({'source_frame':f,'reference':{'path':str(p),'sha256':sha(p)},'renders':renders})
 cfg=R/f'lidar_config_s{s}_all4.json';score=R/f'lidar_score_s{s}_all4.json';summary=R/f'lidar_summary_s{s}_all4.json'
 if cfg.exists() or score.exists() or summary.exists():raise FileExistsError('preserve existing results')
 cfg.write_text(json.dumps({'arms':arms,'frames':frames,'scope':'four map common support, descriptive DEV'},indent=2))
 subprocess.run([sys.executable,str(O/'source/score_lidar_renders.py'),'--config',str(cfg),'--out',str(score)],check=True,stdout=subprocess.DEVNULL)
 subprocess.run([sys.executable,str(O/'aggregate_lidar_score_v1.py'),'--infile',str(score),'--out',str(summary)],check=True,stdout=subprocess.DEVNULL)
 r=json.loads(summary.read_text())['buckets']['all'];print(s,r['common_points'],r['arms'],flush=True)
