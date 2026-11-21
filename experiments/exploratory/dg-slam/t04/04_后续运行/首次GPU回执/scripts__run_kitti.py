"""R1 GPU worker: no GT poses loaded, accepted R0 optimization budget retained."""
import argparse
import functools
import importlib
import json
import os
import sys
import time
from pathlib import Path
from kitti_inputs import ROOT,sha256,load_r1_window,install_gt_guard
from r1_pose_io import matrix_to_droid_pose

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);a=p.parse_args()
    assert Path(a.run_name).name==a.run_name and a.run_name.startswith('KITTI00_200_')
    output=ROOT/'04_reproduction'/a.run_name;output.mkdir(exist_ok=False)
    source=ROOT/'source'
    os.chdir(source);sys.path[:0]=[str(source),str(source/'dg_slam')]
    audit=install_gt_guard()
    import numpy as np
    import torch
    import lietorch
    from lietorch import SE3
    from dg_slam.config import load_config
    from dg_slam.dg_model import dg_model
    from dg_slam.gaussian.common import setup_seed
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
    frozen=json.loads((ROOT/'frozen_config.json').read_text());cfg=frozen['cfg'];args=argparse.Namespace(**frozen['args'])
    cfg['data'].update(n_img=200,output=str(output),exp_name='KITTI00_NO_GT_NO_SEMANTIC');cfg['wandb']=False
    setup_seed(cfg['setup_seed'])
    counts={k:0 for k in ['input_frames','input_frames_submitted','gs_tracking_calls','gs_mapping_calls','gs_pose_optimization_calls','gs_render_calls','mapping_optimize_calls']}
    meta={'status':'RUNNING','variant':'R11_no_runtime_GT_timestamps','frames':200,'cfg':cfg,'args':vars(args),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'runtime_source':str(source),'worker_sha256':sha256(Path(__file__)),'inputs_code_sha256':sha256(Path(__file__).with_name('kitti_inputs.py')),'frozen_config_sha256':sha256(ROOT/'frozen_config.json'),'notes':['GT poses absent from runtime input and video state','First GS pose Identity; second uses DROID; idx>=2 keeps author iterations','Measured RGB-D retained','Post-run CPU evaluation is a separate process','Adapter snapshot; author final color refinement not triggered by this short window']}
    meta['variant']='KITTI_STEREO_DERIVED_RGBD_NO_SEMANTIC_ENGINEERING'
    meta['notes']=['SGBM pseudo-depth, no GT or ORB pose input','No semantic exclusion; not original OneFormer method','640x192 engineering adaptation; not formal quality acceptance']
    started=time.time()
    def receipt():
        meta.update(counters=dict(counts),gt_access_audit=dict(audit),wall_seconds=time.time()-started,max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        (output/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    receipt()
    try:
        data,rows,input_meta=load_r1_window();meta.update(input_meta)

        model=dg_model(cfg,args);mapper=model.tracking_mapping
        pixel_hash_to_id={__import__('hashlib').sha256(x[1][0].numpy().tobytes()).hexdigest():x[0] for x in data}
        assert len(pixel_hash_to_id)==200
        frame_audit=[]
        original_dispatch=mapper.run
        def audited_dispatch(idx,*args,**kwargs):
            key=int(idx)
            pixels=model.video.images[key].cpu().numpy()
            pixel_hash=__import__('hashlib').sha256(pixels.tobytes()).hexdigest()
            actual_id=pixel_hash_to_id[pixel_hash]
            recorded_id=int(model.video.tstamp[key].item())
            assert actual_id==recorded_id, (key,actual_id,recorded_id)
            frame_audit.append({'keyframe':key,'actual_rgb_input_frame':actual_id,'recorded_input_frame':recorded_id,'resized_rgb_sha256':pixel_hash})
            return original_dispatch(idx,*args,**kwargs)
        mapper.run=audited_dispatch
        assert not hasattr(model.video,'poses_gt') and not hasattr(mapper,'gt_c2w_list')
        for module in [sys.modules[type(model).__module__],sys.modules[type(mapper).__module__]]:
            assert Path(module.__file__).resolve().is_relative_to(source.resolve())
        def count_method(name,key):
            original=getattr(mapper,name)
            @functools.wraps(original)
            def wrapped(*args,**kwargs):
                counts[key]+=1;result=original(*args,**kwargs)
                if name=='mapping':print(json.dumps({'event':'mapping_completed','keyframe':int(args[0]),'gaussians':int(mapper.gaussians.get_xyz().shape[0])}),flush=True)
                return result
            setattr(mapper,name,wrapped)
        for name,key in [('tracking','gs_tracking_calls'),('mapping','gs_mapping_calls'),('optimize_cam_in_batch','gs_pose_optimization_calls'),('optimize_map','mapping_optimize_calls')]:count_method(name,key)
        module=importlib.import_module(type(mapper).__module__);original_render=module.render
        def counted_render(*args,**kwargs):
            counts['gs_render_calls']+=1;return original_render(*args,**kwargs)
        module.render=counted_render
        for t,image,depth,intrinsics,none,mask in data:
            assert none is None;counts['input_frames_submitted']+=1
            model.track(t,image,depth,intrinsics,mask);counts['input_frames']+=1
            if counts['input_frames']%10==0:receipt();print(json.dumps(counts),flush=True)
        n=int(model.video.counter.value)
        assert len(frame_audit)==n
        (output/'mapped_frame_identity.json').write_text(json.dumps(frame_audit,indent=2))
        meta['mapping_frame_identity_audit']='PASS'
        meta['mapping_frame_identity_sha256']=sha256(output/'mapped_frame_identity.json')
        assert n==model.mapping_counter and n>2 and model.frontend.is_initialized
        matrices=mapper.estimate_c2w_list[:n].cpu().numpy()
        assert np.allclose(matrices[0],np.eye(4),atol=1e-6)
        arrays={key:getattr(mapper.gaussians,method)().detach().cpu().numpy() for key,method in [('xyz','get_xyz'),('features_dc','get_features_dc'),('features_rest','get_features_rest'),('scaling','get_scaling'),('rotation','get_rotation'),('opacity','get_opacity')]}
        assert arrays['xyz'].shape[0]>0 and all(np.isfinite(x).all() for x in arrays.values())
        np.savez(output/'gaussian_snapshot.npz',**arrays,estimated_keyframe_c2w=matrices,keyframe_input_indices=model.video.tstamp[:n].cpu().numpy())
        estimated=torch.stack([matrix_to_droid_pose(x) for x in mapper.estimate_c2w_list[:n]])
        assert np.allclose(SE3(estimated).matrix().numpy(),matrices,atol=1e-5), 'DROID pose conversion must preserve GS matrices'
        np.save(output/'droid_pose_seeds.npy',estimated.numpy())
        meta['droid_pose_seed_roundtrip']='PASS'
        model.video.poses[:n]=lietorch.cat([SE3(estimated)],0).inv().data
        trajectory=model.terminate_woBA(data)
        assert trajectory.shape==(200,7) and np.isfinite(trajectory).all()
        np.savez(output/'trajectories.npz',estimated=trajectory,rgb_timestamps=[float(x['rgb_timestamp']) for x in rows])
        assert all(counts[k]>0 for k in ['gs_pose_optimization_calls','gs_mapping_calls','gs_render_calls'])
        assert audit['blocked_gt_accesses']==0
        meta.update(status='COMPLETED',gaussian_count=int(arrays['xyz'].shape[0]),gaussian_snapshot_sha256=sha256(output/'gaussian_snapshot.npz'),trajectory_sha256=sha256(output/'trajectories.npz'),first_keyframe_identity=True,video_gt_pose_buffer_present=False)
    except Exception as error:
        meta.update(status='FAILED',error=repr(error));raise
    finally:
        if 'model' in locals():meta.update(droid_keyframes=int(model.video.counter.value),mapped_keyframes=int(model.mapping_counter),frontend_initialized=bool(model.frontend.is_initialized),graph_edges=int(model.frontend.graph.ii.numel()))
        receipt()

if __name__=='__main__':main()
