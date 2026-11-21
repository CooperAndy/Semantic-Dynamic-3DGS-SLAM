"""Adapt seq13 local-index predictions to source frames and map poses."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
import numpy as np
from prepare_seq10_tmd_motion import anchor,sha,save
R=Path('/root/autodl-tmp/seq13_000780_001039_O3F_260_20261115');B=R/'motion_demo_20261116';A=B/'annotations_20261107';OUT=B/'prepared';MAP=R/'cell_O3_retry1/runs/O3F/params.npz';KEY='6cebfe6efc70888706c17c443e5774e3894bd74e0d0baa6f495b51a33bb97c20'
def main():
 assert sha(MAP)==KEY and json.loads((B/'status.json').read_text())['stage']=='complete'
 poses=json.loads((B/'camera_poses.json').read_text());assert len(poses)==260
 with np.load(MAP) as z:k=z['intrinsics'];assert int(z['org_width'])==1241 and int(z['org_height'])==376
 OUT.mkdir();(OUT/'frames').mkdir(); records=json.loads((R/'input_manifest.json').read_text());lookup={(r['local'],r['camera']):r['sha256'] for r in records}
 for i,pose in enumerate(poses):
  n=i+780;assert pose['source_frame']==n and sha(R/f'sequences/13/image_2/{i:06d}.png')==lookup[i,2]
  raw=json.loads((A/f'motion/{i:06d}.json').read_text());meta=json.loads((A/f'instances/{i:06d}.json').read_text());assert raw['frame']==i
  with np.load(A/f'instances/{i:06d}_masks.npz') as z:masks=z['masks']
  depth=np.load(R/f'raft_left/{i:06d}_depth.npy'); assert depth.shape==masks.shape[1:]==(376,1241)
  for obj in raw['objects']:
   mask=masks[obj['mask_index']].astype(bool);assert int(mask.sum())==obj['mask_area'];obj.update(anchor(mask,depth,pose,k));obj['state']={'static':'静态','dynamic':'动态','unknown':'未知'}[obj['raw_state']]
  snap=dict(dataset='KITTI odometry',sequence='13',scene_id='kitti13_o3f_780_1039',camera='image_2',reference_frame=n,image_width=1241,image_height=376,source_map_sha256=KEY,motion_model='Pi3MOS',threshold=.5,window=[x+780 for x in raw['window']],objects=raw['objects'],history_note='Causal current and previous seven frames; offline predictions.',depth='Metric RAFT stereo depth; O3F estimated poses.')
  save(OUT/f'frames/{n:06d}.json',snap)
 subprocess.run(['/root/miniconda3/bin/python',str(B/'prepare_inputs.py'),'--instances',str(A/'instances'),'--depth',str(R/'raft_left'),'--motion',str(A/'motion'),'--params',str(MAP),'--sequence','13','--start','780','--end','1039','--local-start','0','--output',str(OUT/'object_inputs')],check=True)
 manifest=json.loads((OUT/'object_inputs/manifest.json').read_text())
 for i,row in enumerate(manifest['frames']):row['rgb_sha256']=lookup[i,2]
 save(OUT/'object_inputs/manifest.json',manifest)
 with tarfile.open(B/'prepared.tar.gz','w:gz') as tar:tar.add(OUT,arcname='prepared')
 print(json.dumps(dict(archive=str(B/'prepared.tar.gz'),sha256=sha(B/'prepared.tar.gz'),frames=260)),flush=True)
if __name__=='__main__':main()
