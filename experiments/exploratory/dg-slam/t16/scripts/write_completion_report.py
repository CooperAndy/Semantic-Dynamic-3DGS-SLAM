from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parents[1]
E=json.loads((F/'receipts/replay_evaluation.json').read_text());G=json.loads((F/'receipts/repaired_numerical_gate.json').read_text())
C=json.loads((F/'trace_result.json').read_text());pair=json.loads((F/'receipts/whole_pair_effect.json').read_text());audit=json.loads((F/'receipts/completed_independent_audit.json').read_text(encoding='utf-8'))
assert E['eligible_full_window_candidate'] is None and audit['status']=='PASS'
(F/'figures').mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
zero=next(r['loss'] for r in C['captures'] if r['frame']==47 and r['h']==0)
for axis,label,color in [(3,'Rotation x','#5276b8'),(4,'Rotation y','#bf6b39')]:
 rows=sorted([r for r in C['captures'] if r['frame']==47 and (r['axis']==axis or r['h']==0)],key=lambda r:r['h'])
 axes[0].plot([r['h'] for r in rows],[r['loss']-zero for r in rows],'-o',label=label,color=color,markersize=4)
axes[0].axvspan(.0035,.00375,color='#b94646',alpha=.15,label='Depth-order switch')
axes[0].set(xlabel='Right-local rotation perturbation (degrees)',ylabel='Actual L1 - initial L1',title='Frame47: actual native objective');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
labels=['Coarse T0','ID','FULL','diag(P)','FULL + cap'];colors=['#858585','#5276b8','#bd554d','#a47a3f','#4e9779']
for i,(label,color) in enumerate(zip(labels,colors)):
 values=[]
 for frame in [47,95]:
  if i==0:values.append(next(r['initial_position_error_cm'] for r in E['rows'] if r['frame']==frame))
  else:values.append(next(r['position_error_cm'] for r in E['rows'] if r['frame']==frame and r['arm']==['LOCAL_ID','LOCAL_FULL','LOCAL_DIAG_P','LOCAL_FULL_CAP'][i-1]))
 axes[1].bar(np.arange(2)+(i-2)*.15,values,width=.14,label=label,color=color)
axes[1].set(xticks=[0,1],xticklabels=['DEV frame47','DEV frame95'],ylabel='Fixed-alignment position error (cm)',title='Eight actual-L1 same-map replays');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
fig.savefig(F/'figures/T16_branch_and_replay.png',dpi=160);plt.close(fig)
lines=['# T16：分支数值验证修复与8次DEV回放结果','',
 '结论：已定位并修复阻止回放的验证假设，两个快照的新分支一致性检查通过，8次同图回放全部完成。但FULL退化仍存在，没有候选满足提前登记的两帧质量条件，故不启动完整DEV窗口。T14正式不采用FULL、T15原数值FAIL均保留。','',
 '![原生目标分支变化与同图回放](figures/T16_branch_and_replay.png)','',
 '## 一、真正发生了什么','',
 '第47帧相机从0.0035°旋转到0.00375°时，高斯12245与11777的深度排序改变。改变前两者深度差只有2.3841858e-7m：x轴扰动后深度相同，y轴扰动后前后交换。原生内核按tile/depth的float32键排序，排序切换会改变前后层颜色混合。','',
 '独立CPU混合器重建主要tile的原生输出，RGB最大绝对差约3.6e-7，深度约2.4e-6m，远小于先登记的1e-4/1e-3m重建门槛。主要tile199中没有成员变化或L1残差过零。仅恢复这一对的旧顺序就恢复约3.05的损失，证明这里主要是排序分支改变。','',
 '| 扰动轴 | 实际全图L1变化 | 仅恢复该对排序增加的L1 | 反事实全图变化 |','|---|---:|---:|---:|']
for row in pair:lines.append(f"| {row['axis']} | {row['native_total_change']:.6f} | {row['loss_increment_if_only_pair_order_restored']:.6f} | {row['counterfactual_total_change']:.6f} |")
lines+=['',
 '该排序反事实只用于诊断，没有应用到优化候选。全图损失原本下降约3.8–4.0，仅恢复该对顺序后变成上升约0.34–0.49。不能据此断言它解释了T14全部位姿退化；它解释的是T15第47帧的跨尺度差分异常。','',
 '## 二、修复了哪一部分','',
 '修复对象是DEV数值验证器对全尺度光滑性的假设。原T15把跨排序分支的有限差分变化也作为局部梯度不一致的停止条件，因而停止；T16将同分支导数与跨分支目标变化分开：','',
 '- 独立CPU导数审计固定T0的排序、成员、alpha/终止/clamp决策和L1残差符号，在同一条件分支比较解析梯度与原三个尺度差分。区域为两个旋转轴已登记top8 tile的并集，共10个tile；第95帧使用同一像素区域。该检查覆盖定位到的两个旋转方向，不宣称全图所有方向的精确导数证明。',
 '- 同时保留全图实际L1在固定主尺度h=0.002下的6维梯度、ID/FULL方向导数及P稳定性检查。',
 '- 相对差15%、余弦0.98、同分支尺度差20%等阈值在新GPU验证前冻结；新方案是曝光DEV上的验证修复，不把T15改判PASS。',
 '- 原P、支持、优化方向、单步半径、Armijo、6次回溯和累计半径R=0.25均未更改。所有真实候选仍使用完整原生渲染和实际L1评分，未冻结排序、未使用GT选步。','',
 '| 帧 | 同分支梯度相对差 | 同分支尺度差 | 同分支余弦 | 全图实际梯度相对差 | 新DEV入口 |','|---|---:|---:|---:|---:|---|']
for row in G['cases']:lines.append(f"| {row['frame']} | {row['conditional_relative_L2']:.4%} | {row['conditional_FD_scale_relative']:.4%} | {row['conditional_cosine']:.9f} | {row['global_relative_L2']:.2%} | {row['status']} |")
lines+=['',
 '没有发现需要修改原生内核或P公式的证据。另修复了捕获工具对未使用internal_radii数组的误检，以及跨CUDA backward逐位相等的过严绑定（改为预登记相对差≤1e-5；实测约5.3e-8/1.0e-7）。两者是执行/审计修复，不是位姿质量改善。','',
 '## 三、8次真实同图回放','',
 '每个快照共用真实观测、地图、初值、支持、H/P、随机状态和首次渲染。每次独立进程、seed1219、最多50轮。先封存8次原始产物，再进行本地CPU GT评价。用原T02全部24个粗关键帧拟合一次SE3变换并固定，所有候选不重新对齐。以下为固定对齐的位置误差，不是完整轨迹ATE。','',
 '| 方法 | 第47帧位置误差(cm) | 第95帧位置误差(cm) |','|---|---:|---:|',
 f"| 粗位姿T0 | {E['rows'][0]['initial_position_error_cm']:.4f} | {next(r['initial_position_error_cm'] for r in E['rows'] if r['frame']==95):.4f} |"]
for arm,label in [('LOCAL_ID','ID'),('LOCAL_FULL','FULL'),('LOCAL_DIAG_P','diag(P)'),('LOCAL_FULL_CAP','FULL+累计约束')]:
 values=[next(r['position_error_cm'] for r in E['rows'] if r['frame']==frame and r['arm']==arm) for frame in [47,95]]
 lines.append(f'| {label} | {values[0]:.4f} | {values[1]:.4f} |')
lines+=['',
 'FULL在两个相同地图/初值的案例都明显差于ID，且损失降幅更大：第47帧FULL约6.42%、ID约2.38%，说明更低的真实渲染L1仍可伴随更大的位姿误差。diag(P)两帧也都差于ID，移除非对角项不是充分修复。','',
 '累计约束将第47帧净平移从FULL约3.82cm限制到约0.48cm，位置误差优于ID；第95帧却从ID约1.432cm变为约1.479cm。它没有满足“两帧均比ID改善至少5%”的预登记条件。平均改善不能代替逐帧门槛，因此不把它提升为完整窗口候选。','',
 '四种精修在这两个案例的固定对齐位置误差均高于粗位姿。这个结果支持以后单独评估NO_GS_REFINE完整轨迹，但两帧结果不能当作完整DROID轨迹已经优于GS的证据。旋转守卫使用同一位置拟合的SE3坐标变换；本轮候选均已在位置项失败，不用旋转守卫救回。','',
 '## 四、决定与边界','',
 '1. 数值阻塞已解释并修复，原定8次回放已经完成。','2. 本轮没有候选通过质量筛选，完整DEV窗口0次，新正式实验0次。按照既定停止条件，结束这一轮FULL增强验证，不追加半径/步长/种子搜索。','3. 保持T14 FULL不采用；ID仍只是强对照，不自动正式采用。若继续研究高斯精修是否有必要，应另登记NO_GS_REFINE与ID的完整轨迹对照，而不是直接扩大FULL试验。','4. 两个案例均为曝光DEV，不给显著性、独立样本或跨序列泛化结论。','',
 '## 五、审计与归档','',
 f"8次运行、{audit['camera_iterations']}轮相机更新记录、{audit['accepted_updates']}个接受步骤的CPU独立审计通过。SciPy矩阵指数复核右乘更新，最大float32矩阵差{audit['expm_max_abs_error']:.1e}；CAP最大接受累计距离{audit['maximum_accepted_anchor_distance_CAP']:.9f}，小于0.25。原T15的50个清单文件、T14的7135个文件及队友交接6657个文件全部哈希不变。",
 '', '工程失败留存：首次捕获因unused内部半径误检退出，原脚本/锁/日志保存在capture_attempt0；CPU评分首次JSON输出遇到numpy.bool_，仅转换为Python bool，前后CSV的SHA完全相同，没有GPU重跑或评价公式变化。系统DNS曾超时，备用DNS解析与最近已认证TCP地址一致，主机密钥验证保持启用；没有保存密码。','',
 '## 可复算证据','',
 '- [30条原生捕获](trace_result.json)、[独立分支对照](receipts/branch_comparison.json)、[同一对高斯全图作用](receipts/whole_pair_effect.json)。',
 '- [新验证协议](repair_protocol.json)、[执行锁](repair_execution_lock.json)、[条件差分](receipts/conditional_derivatives.json)、[新数值门槛](receipts/repaired_numerical_gate.json)。',
 '- [8次回放结果CSV](T16_八次同图回放结果.csv)、[候选筛选](receipts/replay_evaluation.json)、[原始结果封存](receipts/replay_raw_seal.json)、[CPU独立审计](receipts/completed_independent_audit.json)。','']
(F/'T16_分支验证修复与8次回放结果.md').write_text('\n'.join(lines),encoding='utf-8')
(F/'README_修复进度.md').write_text('# T16：已完成\n\n分支数值验证已修复，两个快照的新检查通过，8次同图回放全部完成。FULL及两个变体均未通过预登记的候选质量筛选，不启动完整DEV窗口。T14不采用FULL、原T15 FAIL均保留。\n\n详见[完整结果报告](T16_分支验证修复与8次回放结果.md)和[8次结果表](T16_八次同图回放结果.csv)。原服务器关闭期间的准备状态保留在独立准备ZIP中。\n',encoding='utf-8')
print('COMPLETION_REPORT_AND_FIGURE_WRITTEN')
