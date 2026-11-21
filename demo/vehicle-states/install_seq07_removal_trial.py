"""Build reversible F1/F200 displays from the seq07 trial's dynamic region."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np

ROOT=Path(__file__).resolve().parent.parent.parent
DEMO=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本'
BASE=DEMO/'scene-data/KITTI07_C60S42'; OUT=BASE/'VehicleRemovalTrial'
BACK=DEMO/'修改记录/20261117_seq07_vehicle_removal_trial'
CFG=DEMO/'源码/scene-config.kitti07-v009-track20-trial.json'
KEY='4ae010a3d6f08ab50e5ca544021de40bdce2f5c19a75fb80de51b2a5929ed6d8'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_ply(p):
 with p.open('rb') as f:
  h=b''
  while not h.endswith(b'end_header\n'):
   line=f.readline();assert line;h+=line
 n=int(re.search(rb'element vertex (\d+)',h).group(1));assert len(re.findall(rb'property float ',h))==17 and p.stat().st_size==len(h)+n*68
 return h,n
def main():
 assert not OUT.exists() and not BACK.exists()
 cfg=json.loads(CFG.read_text(encoding='utf8'));snapshot=BASE/'KITTI07_V009_Track20Trial/snapshot.json';snap=json.loads(snapshot.read_text(encoding='utf8'));source=json.loads((BASE/'KITTI07_ObjectStates/manifest.json').read_text(encoding='utf8'))
 assert cfg['sceneId']==snap['scene_id']=='kitti07_c60s42_v009_track20_trial' and source['source_map_sha256']==KEY and cfg.get('dynamicVehicleRemoval') is None
 dyn=[o for o in snap['objects'] if o['final_state']=='dynamic'];assert len(dyn)==1 and dyn[0]['object_id']=='V009'
 with np.load(BASE/'KITTI07_ObjectStates/evidence/gaussian_region_ids.npz',allow_pickle=False) as z:region=z['region_id']
 mask=np.isin(region,[o['gaussian_region_index'] for o in dyn]);count=int(mask.sum());assert len(mask)==3168264 and count==878==dyn[0]['region_gaussian_count']
 f1=BASE/'KITTI07_F1/params.npz';f200=BASE/'KITTI07_F200/params.npz';assert sha(f1)==KEY
 with np.load(f1,allow_pickle=False) as x,np.load(f200,allow_pickle=False) as y:
  for k in ['means3D','log_scales','unnorm_rotations','cam_trans','cam_unnorm_rots','timestep']:assert np.array_equal(x[k],y[k]),k
 originals={v:BASE/f'KITTI07_{v}/KITTI07_{v}_RGB.ply' for v in ('F1','F200')}
 before={v:sha(p) for v,p in originals.items()};OUT.mkdir();np.savez_compressed(OUT/'selected_mask.npz',selected=mask)
 result={}
 for v,p in originals.items():
  h,n=read_ply(p);assert n==len(mask);source_rows=np.memmap(p,dtype='<f4',mode='r',offset=len(h),shape=(n,17));newh=re.sub(rb'element vertex \d+',f'element vertex {n-count}'.encode(),h)
  removed=OUT/f'{v}_RGB_removed.ply';preview=OUT/f'{v}_RGB_highlight.ply'
  with removed.open('xb') as out,preview.open('xb') as highlight:
   out.write(newh);highlight.write(h)
   for i in range(0,n,200000):
    rows=np.array(source_rows[i:i+200000],copy=True);sel=mask[i:i+len(rows)];rows[~sel].tofile(out);rows[sel,6:9]=(np.array([1.,0.,0.])-.5)/0.28209479177387814;rows.tofile(highlight)
  assert sha(p)==before[v]; result[v]=dict(source_rgb_sha256=before[v],removed_sha256=sha(removed),preview_sha256=sha(preview))
 manifest=dict(sequence='07',scene=cfg['sceneId'],source_map_sha256=KEY,source_snapshot_sha256=sha(snapshot),selection='All Gaussian rows assigned to region V009, currently displayed as dynamic in the Track 20 trial',selected_objects=['V009'],region_index=8,removed_gaussians=count,retained_gaussians=len(mask)-count,mask_sha256=sha(OUT/'selected_mask.npz'),versions=result,limitations='Track 20 dynamic observations overlap this region, but different vehicles share it and the Gaussian-to-moving-vehicle identity is not verified. This may hide static or incomplete geometry; the original map remains unchanged.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 BACK.mkdir();shutil.copy2(CFG,BACK/CFG.name);prefix='scene-data/KITTI07_C60S42/VehicleRemovalTrial/'
 cfg['dynamicVehicleRemoval']=dict(version=2,removedCount=count,gaussianCount=len(mask)-count,maskSha256=manifest['mask_sha256'],manifest=prefix+'manifest.json',highlights={v:prefix+f'{v}_RGB_highlight.ply' for v in ('F1','F200')},variants={v:{'rgb':prefix+f'{v}_RGB_removed.ply','rgbWithTrajectory':None} for v in ('F1','F200')},warning='Trial: V009 may contain points from other vehicles; Track 20 is not verified as the owner of all region Gaussians. Preview before hiding.')
 CFG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
