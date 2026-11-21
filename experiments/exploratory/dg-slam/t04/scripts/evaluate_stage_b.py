"""Full CPU evaluation of B, including support/fallback and paired diagnostics."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_kitti_tracking import score,load_kitti,rigid

R=Path(__file__).resolve().parents[1]
D=R/'04_后续运行/stage_b_depthselection/full200'
O=R/'05_报告与归档/阶段B观测一致筛选_20261012';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'run_manifest.json').read_text())
assert m['status']=='COMPLETED' and (D/'tracking.exit').read_text().strip()=='0'
assert m['counters']['input_frames']==m['counters']['input_frames_submitted']==200
assert m['gt_access_audit']['blocked_gt_accesses']==0
for name,key in [('trajectories.npz','trajectory_sha256'),('gaussian_snapshot.npz','gaussian_snapshot_sha256'),('mapped_frame_identity.json','mapping_frame_identity_sha256')]:assert sha(D/name)==m[key]
prep=R/'02_Gaussian适配审计/GS_SLAM借鉴_20261012/stage_b_patch/preparation.json'
assert sha(prep)==m['selection_protocol']['preparation_sha256']
assert sha(R/'scripts/depth_consistent_selector.py')==m['selection_protocol']['selector_sha256']
assert sha(R/'scripts/run_stage_b.py')==m['worker_sha256']
A=R/'04_后续运行/stage_a_depthguard/full200'
am=json.loads((A/'run_manifest.json').read_text())
assert m['runtime_protocol_sha256']==am['runtime_protocol_sha256']
assert m['frozen_config_sha256']==am['frozen_config_sha256']
assert m['gpu']==am['gpu'] and m['torch']==am['torch']
gt=load_kitti(R/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
delta=np.array(json.loads((R/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])
def p0(x):
    y=x.astype(np.float64).copy();y[:,:3,3]+=y[:,:3,:3]@delta;rigid(y);return y
def matrices(a):
    assert a.shape==(200,7) and np.isfinite(a).all()
    assert np.allclose(np.linalg.norm(a[:,3:],axis=1),1,atol=1e-5)
    x=np.tile(np.eye(4),(200,1,1));x[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix();x[:,:3,3]=a[:,:3]
    return p0(x)
t=np.load(D/'trajectories.npz',allow_pickle=False);at=np.load(A/'trajectories.npz',allow_pickle=False)
assert np.array_equal(t['rgb_timestamps'],at['rgb_timestamps']) and np.all(np.diff(t['rgb_timestamps'])>0)
metrics,aligned,errors=score(matrices(t['estimated']),gt)
a_metrics,a_aligned,a_errors=score(matrices(at['estimated']),gt)
g=np.load(D/'gaussian_snapshot.npz',allow_pickle=False);assert all(np.isfinite(g[k]).all() for k in g.files)
ids=g['keyframe_input_indices'].astype(int);identity=json.loads((D/'mapped_frame_identity.json').read_text())
assert len(ids)==len(identity)==m['mapped_keyframes']==m['droid_keyframes']
assert np.all(np.diff(ids)>0) and ids[0]>=0 and ids[-1]<200
z=np.load(D/'pose_stages.npz',allow_pickle=False);az=np.load(A/'pose_stages.npz',allow_pickle=False)
assert np.array_equal(ids,z['frame_ids'])
hist=[json.loads(x) for x in (D/'selection_history.jsonl').read_text().splitlines()]
assert len(hist)==len(ids)
for key,(i,row,h) in enumerate(zip(ids,identity,hist)):
    assert row['actual_rgb_input_frame']==row['recorded_input_frame']==h['source_frame']==int(i)
    assert h['keyframe']==key
eligible=[h for h in hist if h['keyframe']>1]
fallback=[h for h in eligible if h['fallback']]
success=[h for h in eligible if not h['fallback']]
assert all(h['iterations_completed']==50 for h in success)
assert sum(h['iterations_completed'] for h in hist)==m['counters']['gs_pose_optimization_calls']
fb=np.array([h['keyframe'] for h in fallback],dtype=int)
assert np.allclose(z['coarse'][fb],z['refined'][fb],atol=1e-7)
def quantiles(values):return dict(zip(['min','p25','p50','p75','max'],np.quantile(values,[0,.25,.5,.75,1]).tolist())) if len(values) else {}
support={'eligible_keyframes':len(eligible),'complete_50_iteration_refinements':len(success),
    'fallback_keyframes':len(fallback),'fallback_rate_eligible':len(fallback)/len(eligible),
    'fallback_rate_all_keyframes':len(fallback)/len(hist),'fallback_source_frames':[h['source_frame'] for h in fallback],
    'fallback_reasons':{reason:sum(h['reason']==reason for h in fallback) for reason in set(h['reason'] for h in fallback)},
    'selected_gaussian_count':quantiles([h['selected'] for h in eligible]),
    'consistent_fraction_of_valid_observation_candidates':quantiles([h['selected']/h['valid_observation'] for h in eligible if h['valid_observation']]),
    'successful_pixel_support_fraction_min_per_keyframe':quantiles([h['pixel_support_fraction_min'] for h in success]),
    'actual_pose_optimizer_calls':m['counters']['gs_pose_optimization_calls'],
    'baseline_A_pose_optimizer_calls':am['counters']['gs_pose_optimization_calls']}
same=np.array_equal(z['frame_ids'],az['frame_ids'])
correspondence={'same_keyframe_source_ids':bool(same)}
if same:correspondence.update(max_coarse_matrix_difference=float(np.max(np.abs(z['coarse']-az['coarse']))),max_coarse_position_difference_m=float(np.linalg.norm(z['coarse'][:,:3,3]-az['coarse'][:,:3,3],axis=1).max()))
sparse={k:score(p0(z[k]),gt[ids])[0] for k in ['coarse','refined']}
# Compare exactly the same adjacent keyframe edges where both B endpoints refined.
ok=np.array([not h['fallback'] and h['keyframe']>1 for h in hist])
edges=np.flatnonzero(ok[:-1]&ok[1:])
def edge_rpe(pred):
    truth=np.linalg.inv(gt[ids[edges]])@gt[ids[edges+1]]
    relative=np.linalg.inv(pred[edges])@pred[edges+1]
    e=np.linalg.inv(truth)@relative
    return {'pairs':len(edges),'translation_rmse_m':float(np.sqrt(np.mean(np.linalg.norm(e[:,:3,3],axis=1)**2))),
        'rotation_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(Rotation.from_matrix(e[:,:3,:3]).magnitude()**2))))}
successful_edges={k:edge_rpe(p0(z[k])) for k in ['coarse','refined']}
successful_edges['warning']='Adjacent original keyframe edges with two successful B endpoints only; not all input-frame RPE; post-selection diagnostic'
xyz=g['xyz'];assert len(xyz)==m['gaussian_count'];centers=g['estimated_keyframe_c2w'][:,:3,3]
near=np.concatenate([np.linalg.norm(xyz[i:i+2048,None,:]-centers[None,:,:],axis=-1).min(axis=1) for i in range(0,len(xyz),2048)])
scale=np.exp(g['scaling'].astype(np.float64)).max(axis=1);assert np.isfinite(scale).all()
geometry={'points':len(xyz),'more_than_80m_from_every_keyframe':int((near>80).sum()),'more_than_1000m_from_every_keyframe':int((near>1000).sum()),'max_nearest_keyframe_distance_m':float(near.max()),'max_activated_scale_m':float(scale.max()),'limits':'sanity diagnostics, not GT geometric reconstruction accuracy'}
prior=json.loads((R/'05_报告与归档/阶段A增点修复评测_20261012/metrics.json').read_text())['full_trajectory']
methods={**prior,'Gaussian_B_on':metrics}
changes={}
for baseline in ['Gaussian_A_on','Gaussian_V2_off','ORB_matched_stereo']:
    base=methods[baseline]
    changes[baseline]={'ate_percent':100*(metrics['ate_translation_rmse_m']/base['ate_translation_rmse_m']-1)}
    for d in [1,10]:
        for metric in ['translation_rmse_m','rotation_rmse_deg']:
            changes[baseline][f'rpe{d}_{metric}_percent']=100*(metrics[f'rpe_delta{d}'][metric]/base[f'rpe_delta{d}'][metric]-1)
report={'status':'CPU_EVALUATED_INTEGRITY_PASS','date_client':'2026-10-12','full_trajectory':methods,'relative_change':changes,
    'selection_and_fallback':support,'map_sanity':geometry,'coarse_refined_keyframes':sparse,'successful_adjacent_keyframe_edges':successful_edges,
    'A_B_correspondence':correspondence,'runtime':{'wall_seconds':m['wall_seconds'],'A_wall_seconds':am['wall_seconds'],'gpu':m['gpu'],'torch_peak_allocated_bytes':m['max_gpu_memory_bytes']},
    'hashes':{n:sha(D/n) for n in ['run_manifest.json','trajectories.npz','gaussian_snapshot.npz','mapped_frame_identity.json','selection_history.jsonl']},
    'evaluation_code_sha256':sha(Path(__file__)),
    'limits':['single exposed DEV window/seed','B combines pre-render selection and fallback; not a pure effect of Gaussian constraints',
        'A/B same runtime but independent online runs; inspect coarse pose differences','historical V2 OFF and ORB are not current-GPU causal controls',
        'filled final trajectories are not strictly online outputs','coarse/refined sparse RPE counts keyframe edges','no BA/20m or parameter search performed']}
(O/'完整评测.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'逐帧误差与回退.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['frame','rgb_timestamp','A_ate_m','B_ate_m','keyframe','fallback','selected_gaussians'])
    by_frame={h['source_frame']:h for h in hist}
    for i in range(200):
        h=by_frame.get(i,{});w.writerow([i,t['rgb_timestamps'][i],a_errors[i],errors[i],h.get('keyframe',''),h.get('fallback',''),h.get('selected','')])
lines=['# T04阶段B完整结果：观测深度一致的高斯筛选','', '200帧完整，退出0；轨迹、地图、帧身份、代码与冻结参数哈希通过；与A输入/时间戳/基础配置一致，GT只用于本地CPU评价。',
    '', '评价：P2→P0，SE(3)无尺度对齐；全轨迹RPE按输入帧间隔。', '',
    '| 方法 | ATE/m ↓ | RPE1平移/m ↓ | RPE1旋转/° ↓ | RPE10平移/m ↓ | RPE10旋转/° ↓ |','|---|---:|---:|---:|---:|---:|']
for k in ['ORB_matched_stereo','Gaussian_V2_off','Gaussian_V2_on','Gaussian_A_on','Gaussian_B_on']:
    v=methods[k];lines.append(f"| {k} | {v['ate_translation_rmse_m']:.6f} | {v['rpe_delta1']['translation_rmse_m']:.6f} | {v['rpe_delta1']['rotation_rmse_deg']:.6f} | {v['rpe_delta10']['translation_rmse_m']:.6f} | {v['rpe_delta10']['rotation_rmse_deg']:.6f} |")
c=changes['Gaussian_A_on']
lines+=['', f"相对同5090环境A：ATE {c['ate_percent']:+.2f}%；单帧平移RPE {c['rpe1_translation_rmse_m_percent']:+.2f}%；10帧平移RPE {c['rpe10_translation_rmse_m_percent']:+.2f}%。单帧/10帧旋转RPE变化分别{c['rpe1_rotation_rmse_deg_percent']:+.2f}%/{c['rpe10_rotation_rmse_deg_percent']:+.2f}%。",
    '', '**判断：B候选相对A减少局部误差，但ATE基本不变，仍未超过ORB或历史关闭精修基线。不能宣布高斯精修整体成功。**',
    '', f"回退：{len(eligible)}个可精修关键帧中，{len(success)}个完成50迭代，{len(fallback)}个回退（{100*len(fallback)/len(eligible):.2f}%），均因一致高斯不足256点；不是loss像素不足。回退帧的refined矩阵与coarse一致，已验证。实际优化调用{m['counters']['gs_pose_optimization_calls']}次，A为{am['counters']['gs_pose_optimization_calls']}次。改善可能同时来自选择更一致的高斯和跳过不可靠更新，尚未拆分两者贡献。",
    '', f"仅观察相邻且两端均成功精修的{len(edges)}条关键帧边，粗/精旋转RPE为{successful_edges['coarse']['rotation_rmse_deg']:.6f}°/{successful_edges['refined']['rotation_rmse_deg']:.6f}°，平移为{successful_edges['coarse']['translation_rmse_m']:.6f}/{successful_edges['refined']['translation_rmse_m']:.6f}米。此为选择后诊断，不能混作全200帧RPE。",
    '',f"地图{len(xyz):,}高斯，距所有关键帧相机超过80/1000米分别{geometry['more_than_80m_from_every_keyframe']}/{geometry['more_than_1000m_from_every_keyframe']}点；最大最近相机距离{near.max():.3f}米，最大激活尺度{scale.max():.3f}米。范围正常不代表地图精度验收。",
    '',f"worker耗时{m['wall_seconds']:.2f}秒，A为{am['wall_seconds']:.2f}秒；实际精修调用减少，不能把耗时下降单独归因于筛选加速。A/B粗位姿对应检查见JSON；独立在线运行，不保证逐位一致。",
    '', '后续建议：保留B为诊断候选。若继续定位收益来源，先用同环境A-OFF对照，再做只使用B的回退判定但成功帧不筛高斯的控制组；控制组会改变地图演化，需报告对应关系。上述是本项目诊断设计，不是论文要求。先拆分回退/筛选贡献，再决定20/40米或粗到精方案；本次未启动新实验。',
    '', '完整数值、筛选/覆盖分布、回退帧号、粗/精诊断及哈希见完整评测.json；逐帧表见逐帧误差与回退.csv。',
    '', '![轨迹、误差及筛选支持](完整结果.png)']
(O/'完整结果与判断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
orb=load_kitti(R/'04_后续运行/orb_matched_v2/CameraTrajectory.txt');_,orb_aligned,orb_errors=score(p0(orb),gt)
ax[0].plot(gt[:,0,3],gt[:,2,3],color='black',label='GT')
for label,x,e in [('ORB',orb_aligned,orb_errors),('A',a_aligned,a_errors),('A+B',aligned,errors)]:
    ax[0].plot(x[:,0,3],x[:,2,3],label=label);ax[1].plot(range(200),e,label=label,alpha=.85)
ax[0].axis('equal');ax[0].set(xlabel='X / m',ylabel='Z / m',title='SE(3), no scale correction')
ax[1].set(xlabel='Input frame',ylabel='Position error / m',title='Full 200-frame trajectory')
ax[2].plot([h['source_frame'] for h in eligible],[h['selected'] for h in eligible],label='Selected Gaussians')
ax[2].axhline(256,color='gray',linestyle='--',label='Minimum 256')
ax[2].scatter([h['source_frame'] for h in fallback],[h['selected'] for h in fallback],color='red',label='Fallback',zorder=4)
ax[2].set(xlabel='Input frame at keyframe',ylabel='Count',title=f'Fallback {len(fallback)}/{len(eligible)}')
for a in ax:a.legend(fontsize=8);a.grid(alpha=.25)
fig.tight_layout();fig.savefig(O/'完整结果.png',dpi=150);plt.close(fig)
print(json.dumps({'metrics':metrics,'changes_vs_A':c,'fallback':support,'map':geometry,'successful_edges':successful_edges,'correspondence':correspondence},ensure_ascii=False,indent=2))
