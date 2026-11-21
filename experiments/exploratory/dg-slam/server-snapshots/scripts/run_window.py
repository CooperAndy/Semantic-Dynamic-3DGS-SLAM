"""GPU-only R0 reference window. Uses author's defaults and records actual GS work."""
import argparse
import functools
import importlib
import json
import os
import sys
import time
from pathlib import Path
from tum_window import ROOT, DATA, rows_for, check_masks, load_window, sha256

class CoverageIncomplete(RuntimeError):
    pass

def window_settings(frames,output):
    from dg_slam.config import load_config
    config='configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz.yaml'
    cfg=load_config(config,'configs/dg_slam.yaml')
    cfg['data']['n_img']=frames
    cfg['data']['output']=str(output)
    cfg['data']['exp_name']='R0_author_GT_assisted'
    cfg['wandb']=False
    args=argparse.Namespace(weights=str(ROOT/'01_source_env/weights/droid.pth'),buffer=max(128,frames+32),image_size=[384,512],stereo=False,disable_vis=True,plot_curve=False,beta=.6,filter_thresh=1.75,warmup=12,keyframe_thresh=2.25,frontend_thresh=12.,frontend_window=25,frontend_radius=2,frontend_nms=1,backend_thresh=15.,backend_radius=2,backend_nms=3,config=config)
    return cfg,args

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, choices=[30,200], required=True)
    p.add_argument('--run-name', required=True)
    a = p.parse_args()
    assert Path(a.run_name).name == a.run_name and a.run_name not in ('.','..')
    rows, index = rows_for(a.frames)
    check_masks(rows, required=True)
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'GPU required; preparation never runs SLAM'
    assert torch.cuda.get_device_capability() == (8,6), 'This build targets sm_86; rebuild for other GPU architectures'
    os.environ['WANDB_MODE'] = 'disabled'
    source = ROOT/'01_source_env/DG-SLAM'
    os.chdir(source)
    sys.path[:0] = [str(source), str(source/'dg_slam')]
    from dg_slam.gaussian.common import setup_seed
    from dg_slam.dg_model import dg_model
    from lietorch import SE3
    import lietorch
    from evaluation.tartanair_evaluator import TartanAirEvaluator
    from run_tum import get_tensor_from_camera
    output = ROOT/('03_gpu_smoke' if a.frames==30 else '04_reproduction')/a.run_name
    output.mkdir(exist_ok=False)
    cfg,args=window_settings(a.frames,output)  # Length set BEFORE model construction.
    setup_seed(cfg['setup_seed'])
    meta = {'status':'RUNNING','variant':'R0_author_GT_assisted','frames':a.frames,'index_sha256':sha256(index),'mask_manifest_sha256':sha256(DATA/'seg_mask/manifest.json'),'adapter_sha256':sha256(Path(__file__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'args':vars(args),'cfg':cfg,'notes':['Author GT initialization / early poses retained','Author evaluator scale=True retained; not metric KITTI or GT-free evidence','Reduced buffer capacity only; training/optimization defaults retained','Short window input count does not establish GS tracking was exercised','Final Gaussian NPZ is an adapter snapshot, not an author final-color-refinement checkpoint']}
    (output/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    counts = {key:0 for key in ['input_frames','input_frames_submitted','gs_tracking_calls','gs_mapping_calls','gs_pose_optimization_calls','gs_render_calls','mapping_optimize_calls']}
    started = time.time()
    try:
        model = dg_model(cfg,args)
        mapper = model.tracking_mapping
        def count_method(name,key):
            original = getattr(mapper,name)
            @functools.wraps(original)
            def wrapped(*args,**kwargs):
                counts[key] += 1
                result = original(*args,**kwargs)
                if name=='mapping':
                    print(json.dumps({'event':'mapping_completed','keyframe':int(args[0]),'gaussians':int(mapper.gaussians.get_xyz().shape[0])}),flush=True)
                return result
            setattr(mapper,name,wrapped)
        for name,key in [('tracking','gs_tracking_calls'),('mapping','gs_mapping_calls'),('optimize_cam_in_batch','gs_pose_optimization_calls'),('optimize_map','mapping_optimize_calls')]: count_method(name,key)
        module = importlib.import_module(type(mapper).__module__)
        original_render = module.render
        def counted_render(*args,**kwargs):
            counts['gs_render_calls'] += 1
            return original_render(*args,**kwargs)
        module.render = counted_render
        data = load_window(a.frames)
        for t,image,depth,intrinsics,pose,mask in data:
            counts['input_frames_submitted']+=1
            try:
                model.track(t,image,depth,pose,intrinsics,mask)
            except RuntimeError as error:
                if t==a.frames-1 and not model.frontend.is_initialized and model.frontend.graph.ii.numel()==0 and 'min()' in str(error):
                    meta['author_end_window_error']=repr(error)
                    raise CoverageIncomplete(f'Short window ended with {model.video.counter.value}/{args.warmup} warmup keyframes and no graph edges; author final-frame flush cannot run') from error
                raise
            counts['input_frames'] += 1
            if counts['input_frames']%10 == 0:
                print(json.dumps(counts),flush=True)
                meta['counters']=dict(counts)
                (output/'run_manifest.json').write_text(json.dumps(meta,indent=2))
        n = model.video.counter.value
        if n==0 or model.mapping_counter==0:
            raise CoverageIncomplete('DROID warmup or GS mapping was not reached')
        gaussian_arrays = {key:getattr(mapper.gaussians,method)().detach().cpu().numpy() for key,method in [('xyz','get_xyz'),('features_dc','get_features_dc'),('features_rest','get_features_rest'),('scaling','get_scaling'),('rotation','get_rotation'),('opacity','get_opacity')]}
        assert gaussian_arrays['xyz'].shape[0]>0 and all(np.isfinite(x).all() for x in gaussian_arrays.values()), 'Empty or non-finite final Gaussian state'
        np.savez(output/'gaussian_snapshot.npz',**gaussian_arrays,estimated_keyframe_c2w=mapper.estimate_c2w_list[:n].cpu().numpy(),reference_keyframe_c2w=mapper.gt_c2w_list[:n].cpu().numpy(),keyframe_input_indices=model.video.tstamp[:n].cpu().numpy())
        meta['gaussian_snapshot']={'file':'gaussian_snapshot.npz','sha256':sha256(output/'gaussian_snapshot.npz'),'count':int(gaussian_arrays['xyz'].shape[0]),'all_finite':True}
        estimated = torch.stack([get_tensor_from_camera(x,Tquad=True).cpu() for x in mapper.estimate_c2w_list[:n]])
        model.video.poses[:n] = lietorch.cat([SE3(estimated)],0).inv().data
        trajectory = model.terminate_woBA(data)
        reference = np.array([x[4].numpy() for x in data])
        np.savez(output/'trajectories.npz', estimated=trajectory, reference=reference, rgb_timestamps=[float(x['rgb_timestamp']) for x in rows])
        result = TartanAirEvaluator().evaluate_one_trajectory(reference,trajectory,scale=True,title='walking_xyz_'+a.run_name,save_path=str(output))
        meta.update(status='COMPLETED', evaluation=result, droid_keyframes=n, mapped_keyframes=model.mapping_counter, gaussian_count=int(mapper.gaussians.get_xyz().shape[0]), gs_tracking_exercised=counts['gs_pose_optimization_calls']>0)
        if not all(counts[key]>0 for key in ['gs_pose_optimization_calls','gs_render_calls','gs_mapping_calls']):
            raise CoverageIncomplete('Input window ended before GS tracking/mapping/rendering were all exercised; do not sign smoke PASS')
    except Exception as error:
        meta.update(status='COVERAGE_INSUFFICIENT' if isinstance(error,CoverageIncomplete) else 'FAILED', error=repr(error))
        raise
    finally:
        if 'model' in locals():
            meta.update(droid_keyframes=int(model.video.counter.value),mapped_keyframes=int(model.mapping_counter),frontend_initialized=bool(model.frontend.is_initialized),graph_edges=int(model.frontend.graph.ii.numel()),warmup_required=int(args.warmup))
        meta.update(counters=counts,wall_seconds=time.time()-started,max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        (output/'run_manifest.json').write_text(json.dumps(meta,indent=2,default=str))

if __name__=='__main__': main()
