"""CPU evaluation of completed Stage A; preserve historical results."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_kitti_tracking import score, load_kitti, rigid

R=Path(__file__).resolve().parents[1]
D=R/'04_后续运行/stage_a_depthguard/full200'
O=R/'05_报告与归档/阶段A增点修复评测_20261012'
O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'run_manifest.json').read_text())
assert m['status']=='COMPLETED' and (D/'tracking.exit').read_text().strip()=='0'
assert m['counters']['input_frames']==m['counters']['input_frames_submitted']==200
assert m['gt_access_audit']['blocked_gt_accesses']==0
assert sha(D/'trajectories.npz')==m['trajectory_sha256']
mapfile=D/'gaussian_snapshot.npz'
assert sha(mapfile)==m['gaussian_snapshot_sha256']
assert sha(D/'mapped_frame_identity.json')==m['mapping_frame_identity_sha256']
traj=np.load(D/'trajectories.npz',allow_pickle=False);a=traj['estimated']
assert a.shape==(200,7) and np.isfinite(a).all()
assert np.allclose(np.linalg.norm(a[:,3:],axis=1),1,atol=1e-5)
v2=np.load(R/'04_后续运行/matched_v2_on/trajectories.npz',allow_pickle=False)
assert np.array_equal(traj['rgb_timestamps'],v2['rgb_timestamps'])
v2meta=json.loads((R/'04_后续运行/matched_v2_on/run_manifest.json').read_text())
assert m['runtime_protocol_sha256']==v2meta['runtime_protocol_sha256']
assert m['frozen_config_sha256']==v2meta['frozen_config_sha256']
assert np.all(np.diff(traj['rgb_timestamps'])>0)
gt=load_kitti(R/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
delta=np.array(json.loads((R/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])
def p0(x):
    x=x.astype(np.float64).copy();x[:,:3,3]+=x[:,:3,:3]@delta;rigid(x);return x
b=np.tile(np.eye(4),(200,1,1));b[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix();b[:,:3,3]=a[:,:3]
metrics,aligned,errors=score(p0(b),gt)
g=np.load(mapfile,allow_pickle=False)
assert all(np.isfinite(g[k]).all() for k in g.files)
ids=g['keyframe_input_indices'].astype(int);identity=json.loads((D/'mapped_frame_identity.json').read_text())
assert len(ids)==len(identity)==m['mapped_keyframes']==m['droid_keyframes']==109
assert np.all(np.diff(ids)>0) and ids[0]>=0 and ids[-1]<200
for row,i in zip(identity,ids):assert row['actual_rgb_input_frame']==row['recorded_input_frame']==int(i)
stages=np.load(D/'pose_stages.npz',allow_pickle=False);assert np.array_equal(ids,stages['frame_ids'])
stage_metrics={k:score(p0(stages[k]),gt[ids])[0] for k in ['coarse','refined']}
stage_metrics['warning']='RPE here counts keyframes, not consecutive input frames'
xyz=g['xyz'];centers=g['estimated_keyframe_c2w'][:,:3,3]
assert len(xyz)==m['gaussian_count']
nearest=np.concatenate([np.linalg.norm(xyz[i:i+2048,None,:]-centers[None,:,:],axis=-1).min(axis=1) for i in range(0,len(xyz),2048)])
scale=np.exp(g['scaling'].astype(np.float64)).max(axis=1)
assert np.isfinite(scale).all()
hist=[json.loads(x) for x in (D/'map_range_history.jsonl').read_text().splitlines()]
assert len(hist)==len(ids)
geometry={'points':len(xyz),'more_than_80m_from_every_keyframe':int((nearest>80).sum()),
    'more_than_1000m_from_every_keyframe':int((nearest>1000).sum()),
    'nearest_keyframe_distance_quantiles_m':dict(zip(['p50','p90','p99','p999','max'],np.quantile(nearest,[.5,.9,.99,.999,1]).tolist())),
    'max_activated_gaussian_scale_m':float(scale.max()),
    'largest_distance_current_camera_over_mapping_m':max(x['max_distance_current_camera_m'] for x in hist),
    'scope':'geometry sanity diagnostic, not GT map accuracy; raw log scales activated with exp'}
old=json.loads((R/'05_报告与归档/适配与公平性排查_20261012/matched_metrics.json').read_text())
methods={k:old[k] for k in ['ORB_matched_stereo','Gaussian_V2_off','Gaussian_V2_on']};methods['Gaussian_A_on']=metrics
base=old['Gaussian_V2_on']
changes={'ate_percent':100*(metrics['ate_translation_rmse_m']/base['ate_translation_rmse_m']-1)}
for d in [1,10]:changes[f'rpe{d}_translation_percent']=100*(metrics[f'rpe_delta{d}']['translation_rmse_m']/base[f'rpe_delta{d}']['translation_rmse_m']-1)
report={'status':'CPU_EVALUATED_INTEGRITY_PASS','run_name':'KITTI00_200_A_on_20261012_01','full_trajectory':methods,
    'change_relative_to_v2_on':changes,'map_sanity':geometry,'stage_diagnostic':stage_metrics,
    'runtime':{'wall_seconds':m['wall_seconds'],'gpu':m['gpu'],'torch_peak_allocated_bytes':m['max_gpu_memory_bytes'],'keyframes':len(ids)},
    'hashes':{'trajectory':sha(D/'trajectories.npz'),'map':sha(mapfile),'manifest':sha(D/'run_manifest.json'),'scorer':sha(Path(__file__))},
    'limits':['single DEV window and seed; no formal adoption', 'Stage A changes admission and expands historical 8m sample paths to 40m',
              'RTX5090 and sm120 compatibility binaries differ from historical V2 RTX3080Ti; not a pure same-runtime causal experiment',
              'final filled trajectory includes postprocessing; sparse stages not interchangeable with full trajectory',
              'map sanity does not establish geometric reconstruction accuracy; activated scales still need quality review',
              'GS-SLAM selection and 20m cutoff not enabled']}
(O/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'per_frame_ate.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['frame','rgb_timestamp','A_on_error_m']);w.writerows(zip(range(200),traj['rgb_timestamps'],errors))
lines=['# T04 阶段A：异常增点修复评测','', '日期：2026-10-12。200帧完整，退出码0，轨迹/地图/帧身份哈希、时间戳、有限性通过。GT只在本地CPU评价读取。',
    '', '评价：P2→P0，SE(3)对齐，无尺度校正；RPE按输入帧间隔计算。', '',
    '| 方法 | ATE/m ↓ | RPE1平移/m ↓ | RPE1旋转/° ↓ | RPE10平移/m ↓ | RPE10旋转/° ↓ |','|---|---:|---:|---:|---:|---:|']
for k,v in methods.items():lines.append(f"| {k} | {v['ate_translation_rmse_m']:.6f} | {v['rpe_delta1']['translation_rmse_m']:.6f} | {v['rpe_delta1']['rotation_rmse_deg']:.6f} | {v['rpe_delta10']['translation_rmse_m']:.6f} | {v['rpe_delta10']['rotation_rmse_deg']:.6f} |")
lines+=['',f"相对V2 ON：ATE {changes['ate_percent']:+.2f}%，RPE1平移 {changes['rpe1_translation_percent']:+.2f}%，RPE10平移 {changes['rpe10_translation_percent']:+.2f}%。正数表示误差增加。",
    '', '**判断：工程修复与tracking改进分别验收；轨迹指标混合，不能采用为tracking整体改善。ATE接近旧版，短间隔平移误差下降而10帧误差上升；ORB在所列全轨迹指标仍更好。**',
    '',f"地图：{len(xyz):,}高斯；距所有关键帧相机超过80米/1000米分别{geometry['more_than_80m_from_every_keyframe']}/{geometry['more_than_1000m_from_every_keyframe']}点；最近相机最大距离{nearest.max():.3f}米；最大激活尺度{scale.max():.3f}米。旧V2 ON有1个超过1000米的点、最大最近相机距离约539万米。本轮不再出现该极端远点，但尺度/渲染质量仍须检查，不能视为建图质量通过。",
    '', f"运行耗时{m['wall_seconds']:.2f}秒，{len(ids)}关键帧；5090与历史3080Ti不同，不计算速度倍率。新增点明显增多，也与8米路径扩至40米有关。",
    '', f"同一次A运行的关键帧诊断：粗位姿旋转RPE {stage_metrics['coarse']['rpe_delta1']['rotation_rmse_deg']:.6f}°，精修后 {stage_metrics['refined']['rpe_delta1']['rotation_rmse_deg']:.6f}°；平移RPE分别 {stage_metrics['coarse']['rpe_delta1']['translation_rmse_m']:.6f}/{stage_metrics['refined']['rpe_delta1']['translation_rmse_m']:.6f}米。这里按相邻关键帧计数，不能与上表连续输入帧RPE混比。清理极端增点后仍存在精修抖动；异常远点不是该现象的充分解释。",
    '', '后续：保留A作为候选修复，先在同一5090兼容环境补V2对照以控制环境变化，再按既定方案接入独立B观测深度一致高斯选择。B前冻结阈值、有效支持和回退规则；20/40米留到B验证后。此处只提出下一步，未启动任何新实验。',
    '', '完整数值、关键帧粗/精诊断、地图检查、哈希及限制见metrics.json。']
(O/'阶段A结果与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'metrics':metrics,'change':changes,'map':geometry,'stage_diagnostic':stage_metrics},ensure_ascii=False,indent=2))
