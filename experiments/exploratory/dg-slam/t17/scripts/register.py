from pathlib import Path
import datetime,json,hashlib,shutil
F=Path(__file__).resolve().parents[1];B=F.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P={'task':'T17_NO_GS_REFINE_VS_ID_COMPLETE_DEV','owner':'CHY','registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Exposed DEV full 200-frame window, not entire TUM sequence; full pipeline and author offline trajectory filler, not strictly causal dense online tracking. No new formal acceptance or unseen Test claim.',
 'remote_root':'/root/autodl-tmp/dg_slam_chy_20261011/10_T17_NO_GS_vs_ID_20261012',
 'T14_root':'/root/autodl-tmp/dg_slam_chy_20261011/07_formal_tracking_T14_20261012',
 'windows':[{'id':'DEV200','selected_start':0,'selected_stop_exclusive':200,'frames':200}],
 'selection_rule':'Continue the entire old DEV0:200 recommended in T16; no GT/metric based window search; frames47/95 already exposed. W1/W2 not rerun.',
 'seeds':[1729,2718,31415],'seed_rule':'Three distinct fresh Python/NumPy/Torch CPU/CUDA initializations, paired across arms; previously used seed labels do not restore data independence. No significance claims from one window.',
 'arms':['NO_GS_REFINE','LOCAL_ID'],'run_order':'Per seed index k rotate two arms by k%2. Six fresh processes, sequential; no result-based reorder, retries, seed changes or tuning.',
 'pipeline':'Original T14 R1.1 source/native/defaults. Fresh Identity initialization and DROID second pose. Gaussian mapping retained (500 first/100 subsequent iterations, BA false). NO_GS uses original gs_pose_refine=False tracking bypass, retains DROID coarse poses, performs zero camera backward; ID uses unchanged T14 local I preconditioner, normalized radius.05,50 iterations,Armijo1e-4,six backtracks. No FULL, no new P/step/coverage tuning. Maps may diverge as treatment consequences.',
 'pose_filler':'Same terminate_woBA offline filler, with each arm own mapped keyframe seeds. Dense trajectory primary; keyframe coarse/tracked/mapped and filler pose change separately reported. NO_GS is not removal of Gaussian mapping or a standalone pure-DROID benchmark.',
 'evaluation':{'GT':'Blocked in GPU runtime. CPU scoring only after all six terminal and raw files sealed. Same original association incl GT-timestamp provenance.','ATE':'SE3 no-scale SVD translation RMSE on all200 expected frames; independent Horn crosscheck.','RPE':'All expected pairs delta1 (199 pairs) and10 (190), translation RMSE and SO3 rotation RMSE.','aggregation':'Arithmetic mean across three seeds; paired per-seed values mandatory. No dropped failures.','sparse':'Same-run coarse/tracked/mapped on actual keyframe IDs; not fixed-map causal effect.'},
 'gates':{'all_six_complete':True,'ID_benefit':{'mean_ATE_ID_over_NO_max':.90,'mean_RPE_t_ID_over_NO_max_both_deltas':.95,'mean_RPE_rotation_ID_over_NO_max_both_deltas':1.05,'each_seed_ATE_RPE_t_RPE_r_ID_over_NO_max':1.05},
 'NO_GS_retention':{'when':'Only if ID benefit gate fails and all six engineering runs complete','mean_and_each_seed_ATE_RPE_t_RPE_r_NO_over_ID_max':1.05,'mean_seconds_NO_over_ID_max':1.05,'mean_peak_memory_NO_over_ID_max':1.20,'meaning':'Observed DEV guardrail only, not statistical equivalence/noninferiority or formal model acceptance'},
 'decision_order':'ID_BENEFIT_SUPPORTED if all ID gates; else PREFER_NO_GS_WITHIN_DEV if all NO retention gates; else INCONCLUSIVE_NEITHER_QUALIFIED. Engineering failure => ENGINEERING_INCOMPLETE, no quality PASS. No automatic new experiments.',
 'resources':'Report seconds, peak allocated memory, render/backward counts and Gaussian counts. If ID >1.2xNO on time/memory/points, ID result only accuracy/resource tradeoff, no equal-resource claim.'},
 'budgets':{'registered_runs':6,'per_worker_timeout_seconds':900,'minimum_free_bytes':5*1024**3,'output_max_bytes':512*1024**2,'smoke':'Before execution lock: first60 old DEV frames, seed1219, both arms; no GT scores. Must exercise mapping and ID camera update, NO camera backward=0. Failures preserve receipts; engineering repair versioned before six workers.','one_shot':'Six runs once, preserve crash/OOM/timeout; no metric-driven retries.'},
 'registration_layers':'This design immutable before smoke/GPU workers. Separate execution lock binds scripts, inputs, common T14 source/native and evaluator after smoke, before six registered runs.',
 'excluded':'New Test claims, independent generalization, Gaussian map quality acceptance, Stage2 reopening, FULL enhancement, hyperparameter search.'}
path=F/'protocol.json';assert not path.exists();path.write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
for role in ['runtime','evaluation']:
 src=B/f'CHY_正式Tracking采用验收_20261012/archive/inputs/DEV_SMOKE_{role}.csv'
 shutil.copy2(src,F/f'inputs/DEV200_{role}.csv')
note=f'''# T17 完整DEV轨迹对照预登记

登记时间：{P['registered_at']}。协议SHA256：`{sha(path)}`。

实验窗口为已曝光DEV的0–199帧，共200帧；不是整条TUM序列。种子1729、2718、31415，每个种子配对NO_GS_REFINE和ID，共6次全新进程。先完成前60帧、seed1219的无GT工程冒烟，再冻结代码与输入执行锁，再运行6次登记实验。

两组均保留高斯建图、相同DROID前端及作者离线轨迹补全。NO_GS_REFINE使用原有关闭相机精修分支；ID保持T14原参数。地图各自从零构建，可随处理方式而变化。不能把NO_GS称作彻底移除高斯建图或纯DROID。

主指标为200帧dense post-filler轨迹的无尺度SE3 ATE和delta1/10的平移、旋转RPE，稀疏关键帧单独报告。所有原始结果封存后才打开GT评价。

ID获益条件：三种子均值ATE至少改善10%、两种平移RPE至少改善5%，两种旋转RPE均值不退化超过5%；每个种子的ATE及两种平移/旋转RPE也不得退化超过5%。

若ID不满足上述条件，则NO_GS仅在全部精度指标的均值与每个种子都不劣于ID超过5%、平均运行时间不超过ID的1.05倍、峰值显存均值不超过1.20倍时，判为本DEV范围优先保留。否则判定不确定，不强行二选一。必须6次全部完整且审计通过；失败不丢弃、不为验收重跑。

这一结果是已曝光DEV上的研究决策，不是独立正式采用验收。完整协议见[protocol.json](protocol.json)。
'''
(F/'T17_完整轨迹对照预登记.md').write_text(note,encoding='utf-8');print('DESIGN_FROZEN',sha(path))
