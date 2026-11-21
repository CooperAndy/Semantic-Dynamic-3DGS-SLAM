from pathlib import Path
import numpy as np, csv, json, hashlib
r=Path('/root/autodl-tmp/seq21_chy_20260926'); p=r/'protocol'; p.mkdir(exist_ok=True)
a=np.loadtxt(r/'tracking/CameraTrajectory.txt').reshape(200,3,4)
T=np.tile(np.eye(4),(200,1,1)); T[:,:3,:4]=a
reb=np.linalg.inv(T[0])@T
np.savetxt(r/'tracking/poses_orb_p2_c2w_rebased.txt',reb[:,:3,:4].reshape(200,12),fmt='%.12e')
assert np.max(np.abs(reb[0]-np.eye(4)))<1e-6
valid=r/'static_valid_masks'; valid.mkdir(exist_ok=True)
for i in range(200):
 label=np.load(r/f'semantic/labels/{i:06d}_semantic.npy',allow_pickle=False)
 assert label.shape==(376,1241)
 np.save(valid/f'{i:06d}_valid.npy',(~np.isin(label,range(10,19))).astype(bool),allow_pickle=False)
rows=[{'local_index':i,'original_frame_id':1200+i,'split':'Dev' if i%10==4 else 'Test' if i%10==9 else 'Train'} for i in range(200)]
with (p/'split.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
train=[i for i in range(200) if rows[i]['split']=='Train']
for mode,ids in [('train160',train),('smoke10',train[:10])]:
 dst=p/'sequences'/f'21_{mode}'
 for c in (2,3):
  (dst/f'image_{c}').mkdir(parents=True,exist_ok=True)
  for i in ids: (dst/f'image_{c}/{i:06d}.png').symlink_to(r/f'sequences/21_1200_1399/image_{c}/{i:06d}.png')
 (dst/'calib.txt').symlink_to(r/'sequences/21_1200_1399/calib.txt')
 with (p/f'mapping_index_{mode}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['source_local_index','original_frame_id']);w.writeheader();w.writerows([{'source_local_index':i,'original_frame_id':1200+i} for i in ids])
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
(p/'protocol_manifest.json').write_text(json.dumps({'sequence':'KITTI21','window':[1200,1399],'count':{'Train':160,'Dev':20,'Test':20},'split_sha256':h(p/'split.csv'),'pose_sha256':h(r/'tracking/poses_orb_p2_c2w_rebased.txt'),'seed_threshold':0.60,'loss_threshold':0.20,'tracking_used_all_200_images':True,'status':'FROZEN_BEFORE_EVAL'},indent=2))
print('PROTOCOL_PASS',len(train),h(p/'split.csv'))
