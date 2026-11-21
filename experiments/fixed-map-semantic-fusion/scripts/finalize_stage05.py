import json,hashlib,tarfile,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'05_条件轻量候选与历史诊断';O=D/'dev_diagnostics_20261007_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(O/'local_independent_verification.json')['status'].startswith('PASS')
dependency=next(r for r in read(R/'04_系统整合与最终归档/final_delivery_20261007_v1/bundle_dependencies.json') if 'vkitti_gt_dev_v2_complete' in r['path'])
dump(D/'dependency_archives.json',[dependency])
rows=read(O/'GT_results.json');deltas=read(O/'candidate_deltas.json');recovery=read(O/'recovery_support_summary.json');depth=read(O/'depth_trigger_diagnostic.json');view=read(O/'V_trigger_diagnostic.json');decision=read(O/'quality_decision.json');GPU=read(O/'GPU_result.json');names=[r['sequence'] for r in depth]
fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
for j,g in enumerate(['S2-V','S2-Z']):axes[0].bar(np.arange(2)+(j-.5)*.3,[100*next(r['mIoU_delta'] for r in deltas if r['sequence']==s and r['candidate']==g) for s in names],.3,label=g)
axes[0].axhline(1,color='red',ls='--',label='1 pp mean gate');axes[0].axhline(0,color='grey',lw=.7);axes[0].set_xticks(range(2),['Scene01','Scene02']);axes[0].set_ylabel('mIoU delta vs S2 (percentage points)');axes[0].set_title('Separate candidates / exposed DEV');axes[0].legend(fontsize=8)
trace=read(O/'history_recovery_trace.json')
for ax,s in zip(axes[1:],names):
 for g in ['S0','S1','S-G','S2','S2-H']:
  rr=[r for r in trace if r['sequence']==s and r['group']==g];ax.plot([r['distinct_frames_processed'] for r in rr],[r['initial_targets_currently_match_unperturbed_prefix']/r['initially_changed_labels'] for r in rr],label=g)
 ax.set_ylim(0,1);ax.set_title(s.split('_')[0]+' early-noise recovery');ax.set_xlabel('Distinct allowed frames processed');ax.set_ylabel('Match unperturbed evolving label');ax.legend(fontsize=8)
fig.savefig(O/'05_candidates_and_recovery.png',dpi=150);plt.close(fig)
table=[]
for s in names:
 for g in ['S2','S2-V','S2-Z']:
  r=next(r for r in rows if r['sequence']==s and r['group']==g);b=next(r for r in rows if r['sequence']==s and r['group']=='S2');table.append(f"|{s.split('_')[0]}|{g}|{r['mIoU_GT_supported_common_classes']*100:.3f}%|{r['coverage']*100:.3f}%|{(r['mIoU_GT_supported_common_classes']-b['mIoU_GT_supported_common_classes'])*100:+.3f}|")
stress=[]
for r in recovery:
 b=next(x for x in rows if x['sequence']==r['sequence'] and x['group']==r['group']);p=next(x for x in rows if x['sequence']==r['sequence'] and x['group']==r['group']+'-EarlyNoise');stress.append(f"|{r['sequence'].split('_')[0]}|{r['group']}|{r['initial_changed_labels']}|{r['final_match_count']} ({100*r['final_recovery_fraction']:.1f}%)|{r['first_match_median_effective_source_support']:.0f}|{100*(p['mIoU_GT_supported_common_classes']-b['mIoU_GT_supported_common_classes']):+.3f}|")
detail='\n'.join(f"- {r['sequence']}：近/中/远诊断像素 {r['trusted_GT_depth_proxy_counts']}，原1米容差权重<0.5的比例分别为 "+'/'.join(f'{v*100:.2f}%' for v in r['rates'])+'。' for r in depth)
pose=read(O/'view_group_pose_audit.json');pose_text=f"时间组内相机最大位移{max(p['max_camera_baseline_m'] for p in pose):.3f}米，最大观察方向变化{max(p['max_view_direction_change_deg'] for p in pose):.3f}度；仅允许融合帧参与分组。"
report='''# 05条件轻量候选与历史诊断完成报告（2026-10-07）

**05限定执行完成：两个真实DEV窗口的触发审计、S2-V/S2-Z独立实验、五组早期错误压力测试、原生GPU评价、本地独立复算及归档。两个候选均未通过预设DEV改善门槛，不替换冻结S0，不宣称新的隔离评价PASS。**

## 输入与范围

使用已曝光的VKITTI2 Scene01/clone330–355和Scene02/clone110–135，各21融合/5评价帧及各自固定Gaussian基图；原有预测标签、标量置信度、SGBM深度、相机和观测缓存不变。GT只用于DEV诊断和评价，未进入候选权重。没有读取KITTI-360已消费最终入口，也没有重开Stage2或训练MLP。

本轮先封存观察规则和单候选参数，再执行GPU，最后读取新候选GT指标。`protocol.json`、`depth_protocol.json`、`quality_policy.json`、源码快照及哈希保留。时间分组固定5本地帧一个块，组总证据上限1；深度参数固定绝对1m+相对0.02×观测深度；没有参数扫描、联合Z+V或根据新结果改规则。

## 触发证据与方法

S2-V在两窗口分别发现'''+str(view[0]['repeated_wrong_with_other_correct_witnesses'])+'、'+str(view[1]['repeated_wrong_with_other_correct_witnesses'])+'''个证据：DEV融合GT关联分布的一致性>=0.8且至少3帧，S2最终类别与其不一致，相同时间组内至少3次同类错误，其他组至少2次正确观测。这里的GT关联共识是诊断代理，不是独立逐Gaussian3D标签；GT不参与时间分组、限额或融合。'''+pose_text+'''

S2-V是离线固定地图方法：对每个Gaussian，在每个时间组内按原S2权重汇总分布；组总权重超过1时统一缩放到1，保留组内相对权重。支持数仍计不同有效源帧，不计为“组数”。持续同类错误仍可能保留。

S2-Z的触发诊断固定近0.5–10m、中10–30m、远30–80m，在融合帧GT非运动且地图深度与GT深度差<=0.1m+0.01zGT的代理集合测量；近远各至少1000像素，远处被原1m残差压低至权重<0.5的比例比近处高至少10个百分点。

'''+detail+'''

S2-Z采用 r=abs(z_map-z_SGBM)/(1+0.02*z_SGBM)，geo=1/(1+r²)，再乘原语义标量。z_map为原生alpha归一化相机z均值，单位米；共同低alpha/无效深度与原预测有效性检查保持原样。重新计算几何伴随权重时原1m权重与缓存在1e-4容差内一致。SGBM不是GT，该模型不是校准方差，也不将残差解释成运动。

## 独立候选结果

主指标为地图渲染二维语义mIoU，沿用原冻结共同类及GT存在类规则；unknown与无地图支持保留主分母。下表各候选独立对比S2。

|窗口|方法|mIoU|标签覆盖|mIoU差，百分点|
|---|---|---:|---:|---:|
'''+ '\n'.join(table)+'''

等权窗口平均：S2-V相对S2 +0.104个百分点，S2-Z +0.183个百分点。均小于预设1个百分点门槛，两个候选均DEV_GATE_FAIL。共同小车辆/非小车辆、边界/内部子组、覆盖和同分风险—覆盖曲线保留在JSON/NPZ；没有用整体mIoU替代子组或把unknown删掉。两个窗口高度相关的像素不能当独立样本，不宣称统计显著性。默认S0与此前隔离评价结论保持冻结。

## 真实DEV早期压力测试

预先固定前三个允许融合帧，稳定GaussianID模100为0的位置，将关联后19类帧分布循环移动一类；其他帧恢复原分布。各组位置、帧、二维输入、可见性和自身原有权重一致，历史一致性全部关闭。**这是真实观测缓存上的关联后标签证据扰动，不是重新生成二维分割，也不是历史反馈网络实验。**

恢复定义为与“同组未扰动的当前累积标签”一致，不等于GT正确。首次恢复可能再次翻转，另列最终恢复和GT mIoU代价；恢复所需支持计不同有效源帧。未扰动Gaussian的证据逐值不变，但其邻近二维像素仍可能受扰动Gaussian的遮挡混合影响。

|窗口|方法|初始改变标签|最终恢复标签|首次恢复支持中位数|真实GT mIoU变化，百分点|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(stress)+'''

S2最终恢复约42%/80%，不能声称累积投票总能纠错。不少未恢复位置仅有前三帧有效支持；没有后续可见观测就没有纠错机会。本诊断支持保持历史反馈关闭，并保留将来使用更新前状态、禁止仅靠历史类别冲突拒绝观测的规则；没有签发历史一致性策略通过，也没有开发遗忘/重置生命周期状态机。

![候选增益与早期恢复曲线](dev_diagnostics_20261007_v2/05_candidates_and_recovery.png)

## 核验、资源与复现

'''+f"GPU原生新渲染{GPU['renders']}份；本地复核120个视角/方法组合，14份新增状态数组对应或CPU重融合、6条完整同分风险曲线。五组旧GT主指标逐值复现，所有共同GT子组mask哈希与旧评价一致，所有新组共享原alpha逐值一致。所有52帧GT源文件以及输入/源码/基图/相机后置核验通过。GPU运行用时{GPU['wall_seconds']:.1f}秒，峰值torch分配{GPU['peak_torch_allocated_bytes']/1024**3:.2f}GiB。"+'''

原始完整包 `stage3_05_raw_complete_20261007_v2.tar.gz`包含状态、几何观测、渲染、指标/子组/风险、协议、源码快照、日志和文件哈希；本补充包保存本地核验、恢复详情、图表、报告和完成状态。旧输入/基图资产引用原VKITTI2完整归档，不重复打包。CPU草稿v1重复错误计数未要求同类，已修复并完整重跑v2，v1只保留开发记录，不用于正式结论。

源码位于阶段3/scripts：stage05_dev_diagnostics.py → stage05_gpu_run.py → stage05_evaluate.py；stage05_pipeline.py执行服务器流程。输出目录保护禁止覆盖，复跑必须另改运行版本并保持原证据。verify_stage05_local.py核验归档和本地CPU一致性；stage05_recovery_detail.py补充逐ID纠错支持数。

05交付完成，新增方法未达到采用门槛。下一步保留S0，06学习/类别运动扩展未启动；若继续研究候选，需要新的开发协议和未消费隔离评价入口，不能回用KITTI-360旧最终窗口调参后再称隔离。
'''
(D/'05_执行完成报告_20261007.md').write_text(report,encoding='utf-8')
(D/'README.md').write_text('# 05_条件轻量候选与历史诊断\n\n2026-10-07限定执行完成；S2-V与S2-Z触发并分别实测，均未通过DEV改善门槛。五组早期标签扰动与纠错支持分析完成，不启用历史反馈。\n\n见[05完成报告](05_执行完成报告_20261007.md)，当前证据目录`dev_diagnostics_20261007_v2`，草稿v1仅开发记录。默认S0保持冻结，06未启动。\n',encoding='utf-8')
(D/'候选启动记录.md').write_text('# 候选启动与结束记录\n\nS2-V：两DEV窗口同类重复错误诊断4283/5302个，独立试验完成，平均mIoU增益0.104个百分点，未过1个百分点门槛。\n\nS2-Z：近远深度权重诊断触发，绝对1m+相对0.02单配置独立试验完成，平均增益0.183个百分点，未过门槛。\n\n未合并Z+V、未使用已消费KITTI360最终窗口调参，未训练MLP。五组历史OFF早期证据扰动完成；恢复仅表示同方法前缀/最终标签一致，GT质量代价另报。详见05完成报告。\n',encoding='utf-8')
dump(D/'05_completion_status.json',{'status':'COMPLETE_05_BOUNDED_DEV_DIAGNOSTICS','S2_V':'TRIGGERED_EXECUTED_DEV_GATE_FAIL','S2_Z':'TRIGGERED_EXECUTED_DEV_GATE_FAIL','early_noise_five_groups':'COMPLETED_HISTORY_OFF','native_GPU_renders':GPU['renders'],'local_independent_verification':'PASS','default_method':'S0_UNCHANGED','new_isolated_final_evaluation':'NOT_AVAILABLE_NOT_CLAIMED','06_started':False})
p=R/'执行进度与下一步.md';text=p.read_text(encoding='utf-8-sig');a=text.index('**最新状态：');b=text.index('\n',a);text=text[:a]+'**最新状态：05限定执行完成。两个轻量候选分别实测，均未达到预设DEV改善门槛；五组真实早期证据扰动、GPU评价和本地复算完成。默认S0不变，06未启动。**'+text[b:]
summary='\n\n## 05条件候选与历史诊断（2026-10-07）\n\n在两已曝光VKITTI2 DEV窗口分别触发并完成S2-V/S2-Z，平均相对S2增益0.104/0.183个百分点，均未达1个百分点门槛。早期标签扰动、纠错支持数、真实GT质量代价和70份新GPU渲染完成，本地120视角/组主指标及完整风险复算通过。未访问KITTI360旧最终入口，默认S0不变。见[05完成报告](LINK)。\n'
if '## 05条件候选与历史诊断（2026-10-07）' not in text:text+=summary.replace('LINK','05_条件轻量候选与历史诊断/05_执行完成报告_20261007.md')
p.write_text(text,encoding='utf-8');rp=R/'reports/阶段3_最终报告.md';text=rp.read_text(encoding='utf-8-sig')
if '## 05条件候选与历史诊断（2026-10-07）' not in text:rp.write_text(text+summary.replace('LINK','../05_条件轻量候选与历史诊断/05_执行完成报告_20261007.md'),encoding='utf-8')
files=[D/'05_执行完成报告_20261007.md',D/'README.md',D/'候选启动记录.md',D/'05_completion_status.json',D/'raw_archive_receipt.json',D/'dependency_archives.json',p,rp]+[O/n for n in ['05_candidates_and_recovery.png','local_independent_verification.json','recovery_support_summary.json','view_group_pose_audit.json']]+list(O.glob('*/S*_recovery_detail.npz'))+[R/'scripts'/n for n in ['verify_stage05_local.py','stage05_recovery_detail.py','finalize_stage05.py']]
hashes={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};mp=D/'completion_05_file_hashes.json';dump(mp,hashes);A=D/'stage3_05_completion_20261007_v2.tar.gz';assert not A.exists()
with tarfile.open(A,'w:gz') as t:
 for p in files+[mp]:t.add(p,arcname=p.relative_to(R).as_posix())
receipt={'path':A.relative_to(R).as_posix(),'sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'bytes':A.stat().st_size,'files':len(files)+1,'raw_archive_sha256':json.loads(read(D/'raw_archive_receipt.json')['stdout'])['sha256']};dump(D/'completion_05_archive_receipt.json',receipt);print(json.dumps(receipt),flush=True)
