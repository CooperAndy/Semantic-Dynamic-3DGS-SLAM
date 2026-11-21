"""Read-only acceptance of completed R0 window artifacts; no CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from tum_window import ROOT,rows_for,sha256

p=argparse.ArgumentParser()
p.add_argument('run_name')
a=p.parse_args()
assert Path(a.run_name).name==a.run_name
run=ROOT/'04_reproduction'/a.run_name
meta=json.loads((run/'run_manifest.json').read_text())
assert meta['status']=='COMPLETED',meta
assert meta['frames']==200 and meta['frontend_initialized']
assert meta['counters']['input_frames']==meta['counters']['input_frames_submitted']==200
for key in ['gs_tracking_calls','gs_mapping_calls','gs_pose_optimization_calls','gs_render_calls','mapping_optimize_calls']:
    assert meta['counters'][key]>0,key
with np.load(run/'trajectories.npz') as f:
    estimated=f['estimated']; reference=f['reference']; timestamps=f['rgb_timestamps']
assert estimated.shape==reference.shape==(200,7)
assert np.isfinite(estimated).all() and np.isfinite(reference).all()
assert np.allclose(np.linalg.norm(estimated[:,3:],axis=1),1,atol=1e-4)
rows,index=rows_for(200)
assert np.array_equal(timestamps,np.array([float(x['rgb_timestamp']) for x in rows]))
with np.load(run/'gaussian_snapshot.npz') as f:
    gaussian_shapes={k:list(f[k].shape) for k in ['xyz','features_dc','features_rest','scaling','rotation','opacity']}
    count=f['xyz'].shape[0]
    assert count==meta['gaussian_count'] and count>0
    assert all(np.isfinite(f[k]).all() and f[k].shape[0]==count for k in gaussian_shapes)
    assert f['estimated_keyframe_c2w'].shape==(meta['droid_keyframes'],4,4)
    assert np.isfinite(f['estimated_keyframe_c2w']).all()
    assert meta['mapped_keyframes']==meta['droid_keyframes'], 'Some DROID keyframes were not mapped'
    assert np.allclose(f['estimated_keyframe_c2w'][:,3,:],np.array([0,0,0,1]))
    keyframe_indices=f['keyframe_input_indices']
    assert (np.diff(keyframe_indices)>0).all() and keyframe_indices.min()>=0 and keyframe_indices.max()<200
    rotation=f['estimated_keyframe_c2w'][:,:3,:3]
    assert np.allclose(rotation@rotation.transpose(0,2,1),np.eye(3),atol=1e-4)
    assert np.allclose(np.linalg.det(rotation),1,atol=1e-4)
assert np.isfinite(meta['evaluation']['ate_score'])
assert np.isfinite(meta['evaluation']['rpe_score']).all()
sys.path.insert(0,str(ROOT/'01_source_env/DG-SLAM'))
from evaluation.evaluator_base import ATEEvaluator,quats2SEs
from evaluation.evaluate_ate_scale import align
from evaluation.evaluate_rpe import evaluate_trajectory
ate_no_scale,gt_aligned,est_aligned,_=ATEEvaluator().evaluate(reference,estimated,False)
_,_,_,author_scale=align(np.matrix(reference[:,:3].T),np.matrix(estimated[:,:3].T),True)
gt_matrices,est_matrices=quats2SEs(np.asarray(gt_aligned),np.asarray(est_aligned))
rpe_fixed={}
for delta in [1,10]:
    errors=np.array(evaluate_trajectory(gt_matrices,est_matrices,param_max_pairs=0,param_fixed_delta=True,param_delta=delta))
    rpe_fixed[str(delta)]={'pairs':int(len(errors)),'translation_rmse_m':float(np.sqrt(np.mean(errors[:,2]**2))),'rotation_rmse_rad':float(np.sqrt(np.mean(errors[:,3]**2)))}
path_length=float(np.linalg.norm(np.diff(reference[:,:3],axis=0),axis=1).sum())
evaluation=dict(meta['evaluation'])
if not np.isfinite(evaluation['kitti_score']).all():
    assert path_length<5, 'KITTI scores missing for another reason; inspect evaluator'
    evaluation['kitti_score']=None
    evaluation['kitti_score_status']='NOT_APPLICABLE: path shorter than author minimum 5m segment; raw manifest retains NaN'
supplementary={'ate_se3_no_scale_m':float(ate_no_scale),'author_est_to_gt_scale_factor':float(author_scale),'gt_path_length_m':path_length,'rpe_fixed_frame_delta_no_scale':rpe_fixed}
artifacts={k:{'bytes':(run/k).stat().st_size,'sha256':sha256(run/k)} for k in ['run_manifest.json','trajectories.npz','gaussian_snapshot.npz']}
report={'status':'PASS','scope':'200-frame GT-assisted R0 pipeline acceptance; not research improvement evidence',
        'run':str(run),'frames':200,'window_seconds':float(timestamps[-1]-timestamps[0]),
        'counters':meta['counters'],'droid_keyframes':meta['droid_keyframes'],'mapped_keyframes':meta['mapped_keyframes'],
        'gaussian_count':count,'gaussian_shapes':gaussian_shapes,'all_artifacts_finite':True,
        'trajectory_quaternions_and_keyframe_rigid_transforms':'PASS','timestamps_match_index':True,
        'evaluation':evaluation,'supplementary_cpu_metrics':supplementary,'wall_seconds':meta['wall_seconds'],
        'peak_torch_allocated_bytes':meta['max_gpu_memory_bytes'],'artifacts':artifacts,
        'limitations':['GT-assisted initialization/early poses','Author evaluator scale=True','Chosen OneFormer checkpoint is not an author-specified exact checkpoint','No final color refinement: author tests keyframe index against input frame count','No Gaussian-off comparison; no tracking/mapping improvement claim']}
out=ROOT/'05_reports/dev200_20261011'
(out/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2,allow_nan=False))
