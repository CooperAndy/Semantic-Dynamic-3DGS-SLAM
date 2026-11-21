"""Build reversible display copies using the existing seq03 predicted object regions."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent.parent.parent;DEMO=ROOT/'阶段2/动态语义Demo_设计/T07_动态语义独立副本';BASE=DEMO/'scene-data/KITTI03_M3F_801';OUT=BASE/'VehicleRemovalTrial';BACK=DEMO/'修改记录/20261116_seq03_vehicle_removal_trial';CFG=DEMO/'源码/scene-config.kitti03-m3f-801.json';KEY='2185763e939269d6b4276fcf509cf1821f1156d714447cbe82fcdad2fffdd58b'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not OUT.exists() and not BACK.exists()
 cfg=json.loads(CFG.read_text(encoding='utf8'));snap=json.loads((BASE/'ObjectStates/snapshot.json').read_text(encoding='utf8'));source=json.loads((BASE/'manifest.json').read_text(encoding='utf8'));assert cfg['sceneId']==snap['scene_id']=='kitti03_m3f_000000_000800' and source['params_sha256']==KEY and cfg.get('dynamicVehicleRemoval') is None
 objects=[o for o in snap['objects'] if o['final_state']=='dynamic'];assert len(objects)==1 and objects[0]['object_id']=='V005'
 with np.load(BASE/'ObjectStates/evidence/gaussian_region_ids.npz',allow_pickle=False) as z: regions=z['region_id']
 mask=np.isin(regions,[o['gaussian_region_index'] for o in objects]);count=int(mask.sum());assert len(mask)==2058667 and count==8603 and count==objects[0]['region_gaussian_count']
 rgb=BASE/'RGB.ply';before=sha(rgb)
 with rgb.open('rb') as f:
  header=b''
  while not header.endswith(b'end_header\n'):
   line=f.readline();assert line;header+=line
 assert int(re.search(rb'element vertex (\d+)',header).group(1))==len(mask) and len(re.findall(rb'property float ',header))==17 and rgb.stat().st_size==len(header)+len(mask)*68
 OUT.mkdir(); np.savez_compressed(OUT/'selected_mask.npz',selected=mask)
 original=np.memmap(rgb,dtype='<f4',mode='r',offset=len(header),shape=(len(mask),17));newheader=re.sub(rb'element vertex \d+',f'element vertex {len(mask)-count}'.encode(),header)
 with (OUT/'RGB_removed.ply').open('xb') as removed,(OUT/'RGB_highlight.ply').open('xb') as highlight:
  removed.write(newheader);highlight.write(header)
  for i in range(0,len(mask),200000):
   rows=np.array(original[i:i+200000],copy=True);sel=mask[i:i+len(rows)];rows[~sel].tofile(removed);rows[sel,6:9]=(np.array([1.,0.,0.])-.5)/0.28209479177387814;rows.tofile(highlight)
 assert sha(rgb)==before
 manifest=dict(sequence='03',source_map_sha256=KEY,source_rgb_sha256=before,selection='Existing predicted dynamic region V005, region index 4; not independently confirmed to be moving',selected_object_ids=['V005'],removed_gaussians=count,retained_gaussians=len(mask)-count,mask_sha256=sha(OUT/'selected_mask.npz'),removed_rgb_sha256=sha(OUT/'RGB_removed.ply'),highlight_rgb_sha256=sha(OUT/'RGB_highlight.ply'),limitations='V005 visually appears parked; its dynamic prediction is likely wrong. Selection is only a predicted semantic region, not a verified complete physical vehicle. Original RGB map is unchanged.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 BACK.mkdir();shutil.copy2(CFG,BACK/CFG.name);prefix='scene-data/KITTI03_M3F_801/VehicleRemovalTrial/'
 cfg['dynamicVehicleRemoval']=dict(version=2,removedCount=count,gaussianCount=len(mask)-count,maskSha256=manifest['mask_sha256'],manifest=prefix+'manifest.json',highlights={'F1':prefix+'RGB_highlight.ply'},variants={'F1':{'rgb':prefix+'RGB_removed.ply','rgbWithTrajectory':None}},warning='Trial: V005 may be a parked vehicle. Preview selected Gaussians before hiding; motion and object boundaries are not verified.')
 CFG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
