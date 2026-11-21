from pathlib import Path
import csv,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parents[1]
A=json.loads((F/'receipts/acceptance.json').read_text(encoding='utf-8'));rows=json.loads((F/'receipts/trial_scores.json').read_text(encoding='utf-8'));P=json.loads((F/'protocol.json').read_text(encoding='utf-8'))
flat=[]
for row in rows:
 item={k:row[k] for k in ['window','seed','arm','status','exit_code']}
 if row['status']=='COMPLETED':item.update(row['metrics']);item.update({k:row[k] for k in ['seconds','peak_memory_bytes','gaussian_count']});item.update(row['counts'])
 flat.append(item)
keys=list(dict.fromkeys(k for r in flat for k in r))
with (F/'T17_六次完整轨迹结果.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
arms=P['arms'];colors=['#4677b1','#d18138'];fig,axes=plt.subplots(1,4,figsize=(14,3.6),layout='constrained')
for ax,key,label,mul in zip(axes,['ATE','RPE_t_1','RPE_t_10','seconds'],['ATE (cm)','RPE translation delta1 (cm)','RPE translation delta10 (cm)','Wall time (seconds)'],[100,100,100,1]):
 values=[[r['metrics'][key]*mul if key!='seconds' else r['seconds'] for r in rows if r['arm']==arm and r['status']=='COMPLETED'] for arm in arms]
 for j,v in enumerate(values):
  if v:ax.bar(j,np.mean(v),color=colors[j],alpha=.75);ax.scatter(j+np.linspace(-.12,.12,len(v)),v,c='black',s=17,zorder=3)
 ax.set_xticks([0,1],['NO GS','ID']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
fig.suptitle('T17: exposed DEV 0:200, three paired seeds; dots are seeds, no confidence intervals')
(F/'figures').mkdir(exist_ok=True);fig.savefig(F/'figures/T17_complete_trajectory.png',dpi=160);plt.close(fig)
lines=['# T17：不做高斯精修与ID的完整DEV轨迹对照','',f"结论：`{A['decision']}`。6次完整轨迹对照是否全部完成：{A['all_six_complete']}。这一结果不改变T14/T15/T16历史结论，不是独立正式验收。",'','## 登记与执行','',f"窗口：已曝光DEV的0–199帧，200帧；种子：{P['seeds']}；两组各3次全新进程，共6次。协议SHA256：`{A['protocol_sha256']}`。",'', 'NO_GS_REFINE保留高斯建图，关闭相机精修并采用DROID粗位姿；ID使用T14原有局部单位矩阵方向与步长/回溯规则。两组共用原T14源码与原生依赖，各自从零构建地图。主轨迹包括作者离线轨迹补全，不能称为严格在线全帧tracking，也不是整条TUM序列。','', '首次60帧工程冒烟在DROID尚未达到12关键帧初始化时强制收尾而失败。尝试空图保护后仍因未初始化地图失败；该补丁已撤回。两次失败及代码均留存。工程冒烟扩展为完整200帧、seed1219，无GT评分，科学窗口/种子/门槛没有改变。两组完整冒烟通过后才冻结执行锁并启动6次登记运行。','', '原始结果全部封存后才进行CPU GT评分；ATE使用无尺度SE3对齐并通过独立Horn/SVD交叉检查，RPE使用完整delta1/10的199/190个期望帧对；不删除失败或不利样本。','', '## 结果','', '![完整轨迹对照](figures/T17_complete_trajectory.png)','', '|方法|ATE/cm|RPE1平移/cm|RPE10平移/cm|平均耗时/s|峰值显存均值/GiB|','|---|---:|---:|---:|---:|---:|']
for arm in arms:
 if arm in A['mean_metrics']:
  m=A['mean_metrics'][arm];lines.append(f"|{arm}|{100*m['ATE']:.4f}|{100*m['RPE_t_1']:.4f}|{100*m['RPE_t_10']:.4f}|{m['seconds']:.2f}|{m['peak_memory_bytes']/1024**3:.3f}|")
lines+=['','|种子|方法|ATE/cm|RPE1平移/cm|RPE10平移/cm|','|---:|---|---:|---:|---:|']
for row in sorted(rows,key=lambda r:(r['seed'],r['arm'])):
 if row['status']=='COMPLETED':m=row['metrics'];lines.append(f"|{row['seed']}|{row['arm']}|{100*m['ATE']:.4f}|{100*m['RPE_t_1']:.4f}|{100*m['RPE_t_10']:.4f}|")
 else:lines.append(f"|{row['seed']}|{row['arm']}|FAILED|FAILED|FAILED|")
lines+=['','## 按预登记门槛作出的决定','', 'ID需要ATE均值至少改善10%、两种平移RPE均值至少改善5%，旋转均值及各种子所有精度项不退化超过5%。若ID不通过，NO_GS需要所有精度项的均值与各种子均不劣于ID超过5%，平均耗时不超过ID的1.05倍、峰值显存均值不超过1.20倍，才能在本DEV范围优先保留。否则结论不确定；不能看完结果重定门槛。','',f"ID失败条件：{', '.join(A['failed_ID_gates']) or '无'}。",'',f"NO_GS保留失败条件：{', '.join(A['failed_NO_gates']) or '无'}。",'']
decision=A['decision']
if decision=='PREFER_NO_GS_WITHIN_DEV':lines+=['在本窗口和三个种子的登记条件下，ID没有提供足够的完整轨迹收益，NO_GS满足保留守卫。建议本DEV配置优先关闭高斯相机精修，保留建图功能；这不是统计等价证明，也没有验收地图质量。']
elif decision=='ID_BENEFIT_SUPPORTED':lines+=['ID通过完整DEV轨迹收益门槛，可保留为后续候选；是否能正式采用仍需未曝光数据和另行冻结的独立验收。']
elif decision=='INCONCLUSIVE_NEITHER_QUALIFIED':lines+=['两者均未满足各自登记条件，本轮不作自动采用决定。保留配对差异和负结果，不追加种子/调参来救回门槛。']
else:lines+=['存在工程不完整，不能给出质量通过结论；失败仍包含在总表中。']
lines+=['','## 边界与证据','', '仅一个已曝光DEV窗口，三个种子反映随机初始化变化，不能据此推断跨序列泛化或统计显著性。两组地图随处理变化，因此完整轨迹结果包含反馈与离线补全的总体效果；与T16固定地图快照的局部因果对照不同。稀疏coarse/tracked/mapped与补全关键帧变化见评分回执。','', '[预登记](T17_完整轨迹对照预登记.md) · [协议](protocol.json) · [执行锁](execution_lock.json) · [工程附录](receipts/engineering_appendix_v2.json) · [冒烟审计](receipts/smoke_audit.json) · [原始结果封存](receipts/raw_seal.json) · [六次CSV](T17_六次完整轨迹结果.csv) · [逐次dense/sparse评分](receipts/trial_scores.json) · [门槛决定](receipts/acceptance.json)']
if A['all_six_complete']:
 no=A['mean_metrics']['NO_GS_REFINE'];ident=A['mean_metrics']['LOCAL_ID']
 lines+=['','## 精度、补全与耗时的具体解释','',f"ID相对NO_GS的ATE均值高{(ident['ATE']/no['ATE']-1)*100:.2f}%，RPE1平移高{(ident['RPE_t_1']/no['RPE_t_1']-1)*100:.2f}%，RPE10平移高{(ident['RPE_t_10']/no['RPE_t_10']-1)*100:.2f}%。精度结果支持关闭精修；NO_GS所有均值/逐种子精度守卫均通过，但平均总耗时为ID的{no['seconds']/ident['seconds']:.6f}倍，高{(no['seconds']/ident['seconds']-1)*100:.2f}%，唯一失败条件是登记的时间守卫。不能将综合未通过解释成ID精度更好，也不能把NO_GS改判为综合通过。",'', '各次耗时（秒）：'+ '；'.join(f"seed{r['seed']} {r['arm']} {r['seconds']:.2f}" for r in rows)+ '。计时覆盖数据加载/模型构造、tracking/mapping及离线补全，源代码导入/bootstrap在计时之前。当前记录没有逐阶段计时；无法确认总耗时差来自建图工作量、GPU状态、顺序还是其他因素，也不能断言精修本身加快了流水线。', '', 'ID在seed1729/31415的稀疏tracked ATE低于各自coarse ATE，说明不能概括为每次局部精修都变差；但这两次的dense post-filler ATE仍高于NO_GS。seed2718的ID连稀疏tracked ATE也变差。补全对既有关键帧位置的最大改变，NO_GS约1.282cm，ID三个种子约1.134/6.923/3.910cm。因此完整轨迹差异包含精修和后续补全的共同效果，不能把全部dense退化都归因于某一个步骤。']
 audit_path=F/'receipts/independent_audit.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text(encoding='utf-8'))
  lines+=['',f"独立CPU复算6条轨迹全部通过，与冻结评价结果的最大数值差为{audit['metric_max_abs_difference']:.3e}。三个种子各有不同轨迹数组哈希，但NO_GS相对首种子的同帧最大位置差仅{audit['seed_variation']['NO_GS_REFINE']['max_same_frame_translation_difference_vs_first_seed_m']*1000:.4f}mm；ID约{audit['seed_variation']['LOCAL_ID']['max_same_frame_translation_difference_vs_first_seed_m']*100:.3f}cm。它们是新进程随机状态重复，不是三个独立场景/独立数据样本。",'','[独立CPU审计](receipts/independent_audit.json) · [历史文件未变](receipts/historical_integrity.json) · [服务器最终状态](receipts/server_final_state.json)']
lines+=['','## CPU评价精度修复记录','', '首次CPU评价在稀疏mapped float32位置中心化与float64 GT混算时触发Horn/SVD MSE差≤1e-12的断言。冻结的evaluate.py和执行锁均未修改；单独登记evaluate_cpu64.py包装器，将两组输入转换为float64后调用原指标函数。所有18个稀疏组合的Horn校验通过，原1e-12校验门槛、ATE/RPE定义、质量门槛均保持不变，GPU运行零次补跑。主dense路径本已使用float64矩阵，修复不改变主指标定义。见[精度修复登记](receipts/CPU_eval_precision_repair.json)和[逐项诊断](receipts/CPU_eval_precision_diagnosis.json)。','', '[交付清单](delivery_manifest.json) · [封存验证](receipts/delivery_validation.json) · [ZIP回执](receipts/bundle_receipt.json)']
(F/'T17_完整轨迹对照结果与决定.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_CSV_FIGURE_WRITTEN',decision)
