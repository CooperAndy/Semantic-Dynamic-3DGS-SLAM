from pathlib import Path
from importlib.machinery import SourceFileLoader
from copy import deepcopy
r=Path('/root/autodl-tmp/seq21_chy_20260926'); p=r/'protocol'
base=SourceFileLoader('s3base','/root/autodl-tmp/s3_seq08_work/configs/splatam_seq08_gt_raft_confhard_200.py').load_module().config
for mode,count,iters in [('smoke10',10,5),('train160',160,60),('all200',200,60)]:
 yaml=r/f'kitti21_{mode}.yaml'
 depth=p/'confidence_full/depth_confhard' if mode!='all200' else r/'confidence_all200/depth_confhard'
 yaml.write_text('dataset_name: "kitti"\npose_path: "'+str(r/'tracking/poses_orb_p2_c2w_rebased.txt')+'"\ndepth_folder: "'+str(depth)+'"\ncamera_params:\n  image_height: 376\n  image_width: 1241\n  fx: 718.856\n  fy: 718.856\n  cx: 607.1928\n  cy: 185.2157\n  png_depth_scale: 1.0\n  crop_edge: 0\n')
 cfg=deepcopy(base)
 cfg['workdir']=str(r/'runs')
 cfg['run_name']=f'{mode}_orb_raft_t060'
 cfg['data'].update(basedir=str(p/'sequences') if mode!='all200' else str(r/'sequences'),sequence=f'21_{mode}' if mode!='all200' else '21_1200_1399',gradslam_data_cfg=str(yaml),desired_image_height=376,desired_image_width=1241,num_frames=count,start=0,end=-1,stride=1)
 cfg['mapping']['num_iters']=iters
 cfg['use_wandb']=False
 if mode=='smoke10': cfg['report_global_progress_every']=1000
 (r/f'splatam_{mode}_config.py').write_text('# Fixed external ORB-SLAM3 P2 c2w poses; SplaTAM use_gt_poses is a legacy branch name.\nconfig = '+repr(cfg)+'\n')
 print(mode,yaml)
