"""Adapt seq13 local-index predictions to source frames and map poses."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
import numpy as np
from prepare_seq10_tmd_motion import anchor,sha,save
R=Path('/root/autodl-tmp/seq03_000000_000800_O3F_M3F_20261116');B=R/'motion_demo_20261116';A=B/'annotations_20261107';OUT=B/'prepared';MAP=R/'cell_m3only_full/runs/M3F/params.npz';KEY='2185763e939269d6b4276fcf509cf1821f1156d714447cbe82fcdad2fffdd58b'
def main():
 assert sha(MAP)==KEY and json.loads((B/'status.json').read_text())['stage']=='complete'
 poses=json.loads((B/'camera_poses.json').read_text());assert len(poses)==801
 with np.load(MAP) as z:k=z['intrinsics'];assert int(z['org_width'])==1242 and int(z['org_height'])==375
 OUT.mkdir();(OUT/'frames').mkdir(); records=json.loads((R/'input_manifest.json').read_text());lookup={(r['local'],r['camera']):r['sha256'] for r in records}
 for i,pose in enumerate(poses):
  n=i;assert pose['source_frame']==n and sha(R/f'sequences/03/image_2/{i:06d}.png')==lookup[i,2]
  raw=json.loads((A/f'motion/{i:06d}.json').read_text());meta=json.loads((A/f'instances/{i:06d}.json').read_text());assert raw['frame']==i
  with np.load(A/f'instances/{i:06d}_masks.npz') as z:masks=z['masks']
  depth=np.load(R/f'raft_left/{i:06d}_depth.npy'); assert depth.shape==masks.shape[1:]==(375,1242)
  for obj in raw['objects']:
   mask=masks[obj['mask_index']].astype(bool);assert int(mask.sum())==obj['mask_area'];obj.update(anchor(mask,depth,pose,k));obj['state']={'static':'静态','dynamic':'动态','unknown':'未知'}[obj['raw_state']]
  snap=dict(dataset='KITTI odometry',sequence='03',scene_id='kitti03_m3f_000000_000800',camera='image_2',reference_frame=n,image_width=1242,image_height=375,source_map_sha256=KEY,motion_model='Pi3MOS',threshold=.5,window=[x for x in raw['window']],objects=raw['objects'],history_note='Causal current and previous seven frames; offline predictions.',depth='Metric RAFT stereo depth; MASS selected estimated poses.')
  save(OUT/f'frames/{n:06d}.json',snap)
 subprocess.run(['/root/miniconda3/bin/python',str(B/'prepare_inputs.py'),'--instances',str(A/'instances'),'--depth',str(R/'raft_left'),'--motion',str(A/'motion'),'--params',str(MAP),'--sequence','03','--start','0','--end','800','--local-start','0','--output',str(OUT/'object_inputs')],check=True)
 manifest=json.loads((OUT/'object_inputs/manifest.json').read_text())
 for i,row in enumerate(manifest['frames']):row['rgb_sha256']=lookup[i,2]
 save(OUT/'object_inputs/manifest.json',manifest)
 with tarfile.open(B/'prepared.tar.gz','w:gz') as tar:tar.add(OUT,arcname='prepared')
 print(json.dumps(dict(archive=str(B/'prepared.tar.gz'),sha256=sha(B/'prepared.tar.gz'),frames=801)),flush=True)
if __name__=='__main__':main()
