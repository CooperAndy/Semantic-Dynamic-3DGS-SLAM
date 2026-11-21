"""CPU-only matched-protocol scoring; never supply reference poses to a worker."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_kitti_tracking import score,load_kitti,rigid
R=Path(__file__).resolve().parents[1];O=R/'05_报告与归档/适配与公平性排查_20261012';O.mkdir(exist_ok=True)
gt=load_kitti(R/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
delta=np.array(json.loads((R/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])
def p0(a):
    b=a.astype(np.float64).copy();b[:,:3,3]+=b[:,:3,:3]@delta;rigid(b);return b
def poses(a):
    assert a.shape==(200,7) and np.isfinite(a).all() and np.allclose(np.linalg.norm(a[:,3:],axis=1),1,atol=1e-5)
    b=np.tile(np.eye(4),(len(a),1,1));b[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix();b[:,:3,3]=a[:,:3];return p0(b)
result={};stages={};plot={}
old=json.loads((R/'05_报告与归档/首次tracking评测_20261012/metrics.json').read_text())['full_trajectory']
result.update(old)
for mode in ['off','on']:
    d=R/'04_后续运行'/('matched_v2_'+mode)
    if not (d/'run_manifest.json').exists():continue
    m=json.loads((d/'run_manifest.json').read_text());assert m['status']=='COMPLETED' and (d/'tracking.exit').read_text().strip()=='0'
    assert m['counters']['input_frames']==200 and m['gt_access_audit']['blocked_gt_accesses']==0
    assert hashlib.sha256((d/'trajectories.npz').read_bytes()).hexdigest()==m['trajectory_sha256']
    a=np.load(d/'trajectories.npz');b=poses(a['estimated']);name='Gaussian_V2_'+mode
    result[name],plot[name],_=score(b,gt)
    z=np.load(d/'pose_stages.npz');ids=z['frame_ids'].astype(int);stages[mode]=z
    assert len(ids)==m['mapped_keyframes'] and np.all(np.diff(ids)>0)
    for stage in ['coarse','refined']:
        result[name+'_'+stage+'_keyframes']=score(p0(z[stage]),gt[ids])[0]
        result[name+'_'+stage+'_keyframes']['rpe_warning']='delta counts keyframes; not directly comparable to full input-frame RPE'
    change=np.linalg.norm(z['refined'][:,:3,3]-z['coarse'][:,:3,3],axis=1)
    result[name]['gs_step_position_m']={'median':float(np.median(change)),'max':float(max(change))}
orbdir=R/'04_后续运行/orb_matched_v2'
if (orbdir/'CameraTrajectory.txt').exists():
    assert (orbdir/'tracking.exit').read_text().strip()=='0'
    b=p0(load_kitti(orbdir/'CameraTrajectory.txt'))
    result['ORB_matched_stereo'],plot['ORB_matched_stereo'],_=score(b,gt)
if len(stages)==2:
    a,b=stages['off'],stages['on'];same=np.array_equal(a['frame_ids'],b['frame_ids'])
    result['off_on_coarse_correspondence']={'same_keyframe_ids':bool(same)}
    if same:result['off_on_coarse_correspondence']['max_matrix_difference']=float(np.max(np.abs(a['coarse']-b['coarse'])))
(O/'matched_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# KITTI00 0–199：适配排查与统一条件对比','','本页由独立CPU评价生成。GT仅在本地评价读取；全轨迹P2→P0、SE(3)对齐、无尺度校正。','', '| 200帧全轨迹 | ATE RMSE/m | 1帧平移RPE/m | 10帧平移RPE/m |','|---|---:|---:|---:|']
for k,v in result.items():
    if 'ate_translation_rmse_m' in v and not k.endswith('_keyframes'):
        lines.append(f"| {k} | {v['ate_translation_rmse_m']:.6f} | {v['rpe_delta1']['translation_rmse_m']:.6f} | {v['rpe_delta10']['translation_rmse_m']:.6f} |")
lines+=['','修正版统一640×192、0–199帧、P2/P3图像、像素中心内参、基线、原时间戳、无语义掩码。Gaussian OFF/ON另统一同一SGBM深度、种子与建图预算。','', '历史ORB与本次恢复库的版本来源不同，单列而不替换历史结果。ORB Stereo与Gaussian伪RGB-D的观测处理仍不同；单次DEV结果不能代表方法总体表现。OFF/ON若粗位姿或关键帧不一致，不作为严格配对因果实验。','', 'coarse/refined关键帧评价在JSON中单列；其RPE间隔按关键帧计算，不能与全200帧RPE直接混比。']
(O/'修正版对比结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if plot:
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(gt[:,0,3],gt[:,2,3],color='black',label='GT')
    for k,v in plot.items():ax.plot(v[:,0,3],v[:,2,3],label=k)
    ax.axis('equal');ax.grid(alpha=.3);ax.legend();ax.set(xlabel='X / m',ylabel='Z / m');fig.tight_layout();fig.savefig(O/'matched_trajectory.png',dpi=160);plt.close(fig)
print(json.dumps(result,ensure_ascii=False,indent=2))
