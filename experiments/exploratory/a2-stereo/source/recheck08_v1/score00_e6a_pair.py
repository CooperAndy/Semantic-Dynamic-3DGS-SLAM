"""C200 E6a proxy recomputation using only C0/C2 paired support; preserves old six-map score."""
import pathlib,json,hashlib,numpy as np
R=pathlib.Path('/root/autodl-tmp/A2_stereo_robust_geometry/recheck08_v1');P=pathlib.Path('/root/autodl-tmp/stage3_g07_g08_20261025_p54957/C200/inputs/runtime/input_manifest.json');cfg=json.loads((R/'lidar00_pair_config.json').read_text());m=json.loads(P.read_text());rows={x['source_frame_id']:x for x in m['rows']};frames=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rec in cfg['frames']:
 row=rows[rec['source_frame']];dp=pathlib.Path(row['depth']);cp=pathlib.Path(row['confidence']);assert sha(dp)==row['hashes']['depth'] and sha(cp)==row['hashes']['confidence'];d=np.load(dp);c=np.load(cp);mask=np.isfinite(d)&(d>.5)&(d<80)&np.isfinite(c)&(c>=.8);pred={}
 for arm in ('C0','C2'):
  item=rec['renders'][arm];p=pathlib.Path(item['path']);assert sha(p)==item['sha256']
  with np.load(p) as z:depth=z['depth'];alpha=z['alpha'];mask &= (alpha>.99)&np.isfinite(depth)&(depth>.5)&(depth<80);pred[arm]=depth
 cell={'source_frame':rec['source_frame'],'common_pixels':int(mask.sum()),'mae_m':{arm:float(np.abs(pred[arm][mask]-d[mask]).mean()) if mask.any() else None for arm in ('C0','C2')}};frames.append(cell)
result={'status':'COMPLETE_POSTHOC_PAIR_MASK','scope':'KITTI00 C200 previously exposed DEV, E6a proxy not independent geometry','frames':frames,'means':{arm:float(np.mean([x['mae_m'][arm] for x in frames if x['mae_m'][arm] is not None])) for arm in ('C0','C2')},'quality_adoption':False};p=R/'e6a00_pair_score.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
