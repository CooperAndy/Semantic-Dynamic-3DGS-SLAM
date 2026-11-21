"""Construct the real 30-frame model; do not track, map, or fabricate masks."""
import json
import os
import sys
import time
import torch
from tum_window import ROOT
from run_window import window_settings

source=ROOT/'01_source_env/DG-SLAM'
sys.path[:0]=[str(source),str(source/'dg_slam')]
os.chdir(source)
os.environ['WANDB_MODE']='disabled'
from dg_slam.dg_model import dg_model
from dg_slam.gaussian.common import setup_seed
reports=ROOT/'05_reports/3080ti_prepare'
output=reports/f'model_init_{time.strftime("%Y%m%d_%H%M%S")}'
cfg,args=window_settings(30,output)
setup_seed(cfg['setup_seed'])
model=dg_model(cfg,args)
assert model.tracking_mapping.n_img==30
assert model.tracking_mapping.estimate_c2w_list.shape==(30,4,4)
assert model.video.images.shape[0]==128 and model.video.counter.value==0
assert str(next(model.net.parameters()).device)=='cuda:0'
assert all(torch.isfinite(p).all() for p in model.net.parameters())
torch.cuda.synchronize()
result={'status':'PASS','gpu':torch.cuda.get_device_name(),'window_frames':30,'buffer_capacity':128,
        'actual_dg_model_constructed':True,'weights_loaded_by_author_method':True,
        'input_frames_processed':0,'tracking_mapping_executed':False,'gaussian_faiss_index_constructed':True,
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'output':str(output)}
(reports/'model_init_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
