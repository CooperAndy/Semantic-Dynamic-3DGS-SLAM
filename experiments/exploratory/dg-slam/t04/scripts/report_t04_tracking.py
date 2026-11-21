"""Local independent validation and scoped T04 readout, no new experiments."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_kitti_tracking import load_kitti,score
R=Path(__file__).resolve().parents[1];O=R/'05_报告与归档/适配与公平性排查_20261012'
m=json.loads((O/'matched_metrics.json').read_text());gt=load_kitti(R/'01_ORB基线/历史评价/seq00_gt_kitti_200.txt')
delta=np.array(json.loads((R/'01_ORB基线/独立CPU核验.json').read_text())['p2_to_p0_translation'])
audit=json.loads((R/'04_后续运行/orb_matched_v2/completed_audit.json').read_text());assert audit['status']=='PASS'
p=json.loads((R/'03_数据索引/matched_v2/pair_manifest.json').read_text())
assert all(audit['right_rgb_hashes'][str(x['frame'])]==x['right_sha256'] for x in p['frames'])
poses={};diagnostics={};maps={};timing={}
for name in ['ORB_matched_stereo','Gaussian_V2_off','Gaussian_V2_on']:
    if name.startswith('ORB'):
        path=R/'04_后续运行/orb_matched_v2/CameraTrajectory.txt';a=load_kitti(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['trajectory_sha256']
    else:
        mode=name.rsplit('_',1)[1];d=R/'04_后续运行'/('matched_v2_'+mode);run=json.loads((d/'run_manifest.json').read_text())
        v=np.load(d/'trajectories.npz')['estimated'];a=np.tile(np.eye(4),(200,1,1));a[:,:3,:3]=Rotation.from_quat(v[:,3:]).as_matrix();a[:,:3,3]=v[:,:3]
        s=d/'gaussian_snapshot.npz';assert hashlib.sha256(s.read_bytes()).hexdigest()==run['gaussian_snapshot_sha256'];g=np.load(s)
        assert all(np.isfinite(g[x]).all() for x in g.files)
        maps[name]={'gaussians':len(g['xyz']),'keyframes':len(g['keyframe_input_indices']),'map_quality_evaluated':False,'xyz_extent_m':(g['xyz'].max(0)-g['xyz'].min(0)).tolist()}
        timing[name]={'whole_worker_seconds':run['wall_seconds'],'peak_torch_allocated_bytes':run['max_gpu_memory_bytes'],'counters':run['counters'],'scope':'SLAM worker, excludes SGBM preparation and launch; historical 3080Ti run'}
    a[:,:3,3]+=a[:,:3,:3]@delta;poses[name]=a;x=a[:,:3,3];y=gt[:,:3,3]
    rot,_=Rotation.align_vectors(y-y.mean(0),x-x.mean(0));aligned=rot.apply(x-x.mean(0))+y.mean(0);ate=float(np.sqrt(np.mean(np.sum((aligned-y)**2,axis=1))))
    assert abs(ate-m[name]['ate_translation_rmse_m'])<1e-7
    diagnostics[name]={'independent_ate_m':ate,'exported_valid_frames':200,'raw_exact_repeated_adjacent_matrices':int(np.sum(np.all(a[:-1]==a[1:],axis=(1,2))))}
    for step in [1,10]:
        pr=Rotation.from_matrix(a[:-step,:3,:3]);gr=Rotation.from_matrix(gt[:-step,:3,:3]);pt=pr.inv().apply(x[step:]-x[:-step]);t=gr.inv().apply(y[step:]-y[:-step]);err=pt-t
        v=float(np.sqrt(np.mean(np.sum(err**2,axis=1))));rr=(gr.inv()*Rotation.from_matrix(gt[step:,:3,:3])).inv()*(pr.inv()*Rotation.from_matrix(a[step:,:3,:3]));angle=float(np.rad2deg(np.sqrt(np.mean(rr.magnitude()**2))))
        # Rotation projects rounded matrices to SO(3); inverse-matrix evaluation
        # retains their sub-micro orthogonality error, accumulated at delta=10.
        assert abs(v-m[name][f'rpe_delta{step}']['translation_rmse_m'])<1e-6
        assert abs(angle-m[name][f'rpe_delta{step}']['rotation_rmse_deg'])<1e-5
        diagnostics[name][f'independent_rpe{step}_m']=v
        diagnostics[name][f'rpe{step}_difference_m']=abs(v-m[name][f'rpe_delta{step}']['translation_rmse_m'])
with (O/'matched_per_frame_ate.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['input_frame','timestamp_s',*poses.keys()]);es={k:score(v,gt)[2] for k,v in poses.items()}
    for i,row in enumerate(p['frames']):w.writerow([i,row['timestamp'],*[es[k][i] for k in poses]])
report={'status':'PASS_WITH_COMPARISON_LIMITS','independent_math':diagnostics,'input_right_hashes':'200/200 PASS','maps':maps,'costs':timing,'ORB_cost':{'mean_TrackStereo_seconds':.0397334,'median_TrackStereo_seconds':.0396445,'whole_worker_seconds':None,'peak_RSS_or_GPU_memory':None,'scope':'from successful native tracking.log; not comparable to Gaussian whole-worker timing'},'missing_for_full_T04':['ORB sparse map export and qualitative map comparison','mapping quality with appropriate metrics and explicit seen/held-out roles','hardware-matched end-to-end cost including depth/mask preprocessing','independent online failure/state logs; final full trajectories alone are insufficient']}
report['map_range_audit']=json.loads((R/'02_Gaussian适配审计/差异排查_20261012/map_extent_audit.json').read_text())
report['mapping_status']='RANGE_ANOMALY_FLAGGED; finite arrays do not establish valid geometry'
(O/'tracking_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# T04进展：KITTI00前200帧统一条件tracking对比','','日期：2026-10-12。恢复后只回收既有结果、核验和CPU评分，未重跑SLAM或启动GPU实验。','', '**判断：本轮tracking对照可以带限制汇报；整个T04尚未完成，mapping与系统成本仍需补齐。**','', '| 方法 | ATE/m | 1帧平移RPE/m | 1帧旋转RPE/° | 10帧平移RPE/m |','|---|---:|---:|---:|---:|']
for k in poses:
    v=m[k];lines.append(f"| {k} | {v['ate_translation_rmse_m']:.6f} | {v['rpe_delta1']['translation_rmse_m']:.6f} | {v['rpe_delta1']['rotation_rmse_deg']:.6f} | {v['rpe_delta10']['translation_rmse_m']:.6f} |")
ref=m['ORB_matched_stereo'];on=m['Gaussian_V2_on'];off=m['Gaussian_V2_off']
lines+=['',f"开启GS精修相对同分辨率ORB：ATE增加{100*(on['ate_translation_rmse_m']/ref['ate_translation_rmse_m']-1):.2f}%，1帧平移RPE为{on['rpe_delta1']['translation_rmse_m']/ref['rpe_delta1']['translation_rmse_m']:.2f}倍。相对OFF，ON的1帧平移RPE为{on['rpe_delta1']['translation_rmse_m']/off['rpe_delta1']['translation_rmse_m']:.2f}倍。",'', '全200帧、KITTI P2→P0、SE(3)对齐、禁止尺度校正，1帧199对／10帧190对。独立SciPy对齐与向量形式RPE复核通过；逐帧ATE见matched_per_frame_ate.csv。','', '## 能确认与不能确认的解释','', '- 已确认旧适配的像素中心投影错误；V2实际GPU投影测试通过。修复后当前实现仍明显抖动，因此不能仅把差距归因于分辨率或半像素错误。','- 同一ON运行的109关键帧：精修前相邻关键帧旋转RPE 0.091928°，精修后1.471463°；这一变化发生在轨迹补全之前。OFF/ON粗位姿存在毫米级数值差，故不能宣称逐位配对因果实验。','- 补全还会改动已有关键帧；完整轨迹指标包括补全过程，不能直接称严格在线tracking性能。','- 残差求和与随位姿变化的opacity mask、深度尺度及早期初始化均为待验证机制，不把猜测写成唯一根因。队友的TUM固定支持／预条件实验是另一分支，本次没有复跑。','', '## 比较条件与限制','', '- 相同KITTI00 0–199、640×192、P2/P3像素、像素中心K、基线与时间戳、无语义排除；600份左右RGB／深度哈希核对。','- ORB用双目特征匹配，Gaussian是DROID前端＋同帧SGBM派生RGB-D＋GS精修变体。OFF是该系统的关闭精修参考，不是“独立Gaussian算法”；观测处理仍不同。','- ORB采用恢复的4452a3c4标签库并在隔离目录重建DBoW2以适配CPU；与历史WSL基线身份不同，历史结果单列。失败的FPS配置与非法指令日志均保留。','- 单窗口、单次DEV结果，不代表方法普遍优劣或正式采用通过。轨迹完整200/200不是独立测得的在线成功率。','', '## 成本与地图交付边界','', f"Gaussian OFF／ON worker分别耗时{timing['Gaussian_V2_off']['whole_worker_seconds']:.2f}／{timing['Gaussian_V2_on']['whole_worker_seconds']:.2f}秒，均不含伪深度准备。ORB日志单次TrackStereo均值0.0397334秒，不含初始化与等待等，缺少同步whole-worker与峰值内存；**不可用这两个口径计算系统加速倍数**。",'', '两个Gaussian地图快照已回收，文件哈希与有限性检查通过；高斯数和空间范围只证明地图产物存在，不证明几何质量。ORB本次只导出了轨迹，没有稀疏地图。','', '接下来优先补T04 mapping：保留当前tracking基线，另建导出实验以获取ORB稀疏地图和Gaussian地图；分别说明稀疏／稠密表示、可渲染性和实际适用的质量指标。若另做统一外部位姿的mapping对照，另立协议，不能代替当前系统自主tracking对比。当前地图使用过所有200输入帧，不能把其中视角称held-out。','','![统一条件轨迹](matched_trajectory.png)']
lines+=['', '## 地图范围异常：下一轮优先审计','', 'OFF/ON各有1个点距所有相机超过1000m，最大距离分别约19,380,838m与5,391,339.5m；两张图99.9%点分别在相机轨迹34.30m／22.54m范围内。异常点未删除，原快照未修改。','', '快照scaling与opacity为原始log尺度／logit，须分别exp／sigmoid后解释；异常点opacity约0.1，激活尺度也极大。详见map_extent_audit.json及audit_map_extent.py。', '', '已确认采样深度资格不一致：common.get_samples仍有8m上限；get_samples_with_pixel_grad和get_samples_point_add未传depth_limit时无上限；add_neural_points只检查depth>0。mapper的loss支持却限40m。模型送入mapper的是DROID优化后disps_up的倒数，不是输入SGBM深度的直接副本，因此输入缓存0.5–80m筛选不保证实际增点深度有界。', '', '这构成明确的范围检查缺口，但尚未证明特定异常点的源帧或它对tracking误差的因果贡献。下一轮在独立副本记录每关键帧实际mapper深度的范围／无效比例和首次异常源帧，统一所有增点入口的深度资格，再比较地图异常点和轨迹；不先改预条件或复制队友机制实验。']
(O/'T04当前进展与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'metrics':{k:m[k] for k in poses},'maps':maps},ensure_ascii=False,indent=2))
