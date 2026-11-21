from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
r=Path('/root/autodl-tmp/stage2_round16_gaussian_export_v1');src={'E6a seq00':Path('/root/autodl-tmp/SplaTAM_e6b/experiments/KITTI/seq00_e6a_confhard_holdout_200_0/params.npz'),'Scene02 D3':Path('/root/autodl-tmp/stage2_round16_scene02_v1/maps/D3/params.npz')};fig,ax=plt.subplots(1,2,figsize=(12,5))
for a,(name,p) in zip(ax,src.items()):
 z=np.load(p);pts=z['means3D'];ix=np.linspace(0,len(pts)-1,min(len(pts),100000),dtype=int);rgb=np.clip(z['rgb_colors'][ix],0,1);a.scatter(pts[ix,0],pts[ix,2],s=.05,c=rgb,rasterized=True);q=z['cam_unnorm_rots'][0].T;t=z['cam_trans'][0].T;q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12);R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();center=-np.einsum('nji,nj->ni',R,t);a.plot(center[:,0],center[:,2],color='red',lw=1,label='Estimated camera');a.set_title(f'{name}: centers sample {len(ix):,}/{len(pts):,}');a.set_xlabel('map x (m)');a.set_ylabel('map z (m)');a.legend();a.axis('equal')
fig.suptitle('Gaussian center spatial diagnostic (not splat rendering)');fig.tight_layout();fig.savefig(r/'gaussian_centers_and_trajectory.png',dpi=150);print('done')
