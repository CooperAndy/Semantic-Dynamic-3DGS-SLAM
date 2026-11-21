from pathlib import Path
from copy import deepcopy
from importlib.machinery import SourceFileLoader
import os
os.environ['CHY07_POLICY']='D3'
r=Path('/root/autodl-tmp/stage2_round12_20261031_v1');b=next(Path('/root/autodl-tmp/slam_project/work_chy_20260924/stage2').glob('07_*'));base=SourceFileLoader('base',str(b/'dev_coverage_20261001_v2/base_train_config.py')).load_module();config=deepcopy(base.config);w=r/'p5_window';policy=os.environ['P5_POLICY'];config['workdir']=str(r/'maps');config['run_name']=policy;config['data']['basedir']=str(w/'replica_layout');config['data']['gradslam_data_cfg']=str(w/'camera_config.yaml');config['data']['num_frames']=40;config['tsdd_states']['asset_root']=str(w/'sidecars/D3');config['tsdd_states']['sidecar_dir']=str(w/'sidecars/D3');config['use_tsdd_states']=policy=='D3';config['tsdd_states']['behavior_enabled']=policy=='D3';config['save_checkpoints']=True;config['checkpoint_interval']=1;config['eval_every']=1;config['report_global_progress_every']=1
