"""Create evidence-bound Step02 engineering DEV report and completion sync."""
import csv,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];step=root/'02_基础因子DEV与Oracle诊断';run=step/'dev_sgbm_native_full_20261007_v1'
result=json.loads((run/'result.json').read_text(encoding='utf-8'));config=json.loads((run/'config.json').read_text(encoding='utf-8'))
restart=json.loads((run/'restart_and_shared_support_validation.json').read_text());assert restart['status']=='PASS'
assert result['status']=='COMPLETE_ENGINEERING_DEV_NOT_SCIENTIFIC_ACCEPTANCE' and result['fusion_frames']==180
stats=json.loads((run/'frame_stats.json').read_text());adj=json.loads((run/'adjoint_validation.json').read_text());assert adj['pass']
manifest=json.loads((step/'native_dev_evidence_SHA256_20261007.json').read_text())
assert all(hashlib.sha256((step/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
table='\n'.join(f"| {g['group']} | {g['gaussians_with_evidence']:,} | {100*g['eval_pixel_coverage']:.2f}% | {100*g['eval_unknown_fraction']:.2f}% | {100*g['rejected_observation_fraction']:.2f}% | {100*g['eval_conditional_teacher_agreement']:.2f}% |" for g in result['groups'])
with (step/'五组工程诊断汇总_20261007.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result['groups'][0]));w.writeheader();w.writerows(result['groups'])
report=f'''# 02执行报告：五组真实融合与工程DEV诊断

日期：2026-10-07，负责人CHY。**五组180帧真实融合及20视角诊断已完成；02的科学质量验收尚未完成。** 可运行的工程部分已归档，O-Sem、独立GT准确率、真实风险—覆盖及GT关键子组因无可用独立语义标注而未执行。不能按教师一致性选出“最佳方法”。

## 1. 实际完成与输入版本

运行目录：`dev_sgbm_native_full_20261007_v1`；另保留三帧小规模试运行`dev_sgbm_native_smoke_20261007_v1`。全量使用原始3000–3199中的180个允许语义融合帧，20个i%10==9视角仅作本次融合排除诊断；每组共180个唯一源帧，生成100份评价视角渲染。历史几何、RGB和语义已见这200帧，不能称新Test或未见几何。

唯一907,488点基图SHA256：`{result['base_sha256']}`，运行前后未改变；不增密、不裁剪，不修改RGB、opacity或相机。每组从零独立语义证据出发，历史一致性关闭，未知语义保留遮挡。原生依赖与01核验的安装包装器及二进制哈希匹配。

深度明确采用**SGBM工程DEV新输入版本**，不是恢复历史RAFT实验。扩大数据盘检索记录见`depth_search_20261007.json`：本窗口发现SGBM与其他工程深度副本，未识别可恢复的原RAFT缓存；不声称已穷尽全部压缩包或服务器副本。原始左右目、模型来源及预测缓存沿用00锁定资产，融合时重新核验180×3份标签/置信度/深度文件以及20份诊断标签哈希。若以后改RAFT，应另建输入版本、全组重跑，不能只改某个组。

运行前冻结配置、源码、帧列表与输入哈希见`config.json`、`input_locks.json`及运行内源码快照。本轮参数是工程初值，没有用独立GT调优或签发科学方法冻结。

## 2. 可见关联实现与实测

未使用Gaussian中心点采样替代足迹关联。利用原生颜色通道的线性关系，通过颜色梯度取得每个Gaussian在有效像素掩码中的alpha/透射率可见贡献总和：`d(sum(mask*render))/d(color_g)`。前向和反向都使用原生投影、排序、遮挡及数值通道。这只对颜色特征求导，不对几何、相机或opacity优化。

在真实场景中先验证伴随恒等式：随机颜色前向与像素掩码内积，等于颜色与对应Gaussian梯度内积；相对误差{adj['relative_error']:.3g}，预定容差1e-5，通过。每帧又核对各类别可见质量总和与有效像素原生alpha积分，相对误差最大{max(s['native_adjoint_total_relative_error'] for s in stats):.3g}。缓存保存稳定Gaussian ID与数组对应、19类帧内分布、语义/几何可靠性、可见alpha质量及输入版本，共180份。

有效像素条件对所有组相同：类别0–18、置信度有限且0–1、SGBM深度有效且0.5–80米、地图alpha≥0.05、alpha归一化地图深度有限且>0。无几何加权的组也保留这些共同基础检查。观测深度无效的像素不提供融合证据，不能把由此留下的unknown当成可靠拒判收益。

Gaussian帧内可见alpha质量至少0.001才形成有效源帧观测；按alpha权重汇总硬标签，再归一化为类别分布，基础帧预算为1。因此投影面积影响分布与观测有效性，但不直接增加独立支持或总帧票数。累计支持单位为不同有效源帧。

## 3. 五组权重与深度定义

地图深度为原生渲染相机z的alpha加权平均：`sum(contribution*z_g)/sum(contribution)`，不是射线距离、中心z或深度GT。低alpha像素不关联；前后景混合仍可能使期望深度偏离可见表面。本轮记录这一限制，未擅自引入复杂动态状态机。

逐像素几何可靠性：`1/(1+(abs(z_map_mean-z_SGBM)/1.0m)^2)`；使用固定绝对1米容差，相对容差为0，因此不属于候选S2-Z。它是未校准启发式关联可靠性，不能解释为运动或类别错误。

语义可靠性是共同可见足迹内选中类别置信度的alpha加权平均，几何可靠性也按同一足迹分别加权平均。S0使用权重1，S1使用语义可靠性，S-G使用几何可靠性，S2使用二者乘积；S2-H在乘积≥0.5时给单位权重，否则0。只有硬标签投票，未把标量分数当成完整类别概率。

该关联、帧内归一化与基础检查对所有组统一实施，旧中心采样融合保留为历史源码，不混作本轮S0。没有引入S2-Z/S2-V、类别先验、运动门控或学习网络。

## 4. 五组结果：仅工程覆盖与教师一致性

| 组别 | 获得证据的Gaussian | 全图标签覆盖 | 全图unknown | 有效关联观测拒绝率 | 有标签像素教师一致率 |
| --- | --- | --- | --- | --- | --- |
{table}

教师是本窗口独立于本次融合的二维Mask2Former预测，不是语义GT。教师一致率分母仅为有标签像素；高一致率可能来自高拒绝率或复制教师错误。覆盖使用全部20张图像的共同完整像素分母；不排除未知或难例来缩小它。各组的no-map比例及完整结果见CSV/`result.json`。20视角几何支持掩码逐值一致。

本次诊断中，四个软组覆盖均为97.01%；S-G和S2的教师一致率分别为83.85%和84.03%，二者差距较小。S2-H拒绝68.89%的有效Gaussian—帧关联观测，使覆盖降到93.09%，教师一致率84.06%，与S2接近。这说明当前硬阈值带来明显证据/覆盖损失，不能仅凭条件一致率略高就认定它更好；这些数字也不能证明任何组对真实语义更准确。

固定0..1共21个阈值的“教师不一致—覆盖”诊断见`teacher_disagreement_coverage.json`与图。同分整块接受，无标签/无地图支持不填类，零接受不计风险为0；它不是独立GT风险曲线，也没有据此签发共同覆盖下的错误率改善。逐类教师混淆与IoU仅作预测相互比较，文件明确标注非GT准确率。

全量融合耗时（含缓存、状态保存、评价输出等）{result['wall_seconds']:.1f}秒；PyTorch峰值分配显存{result['peak_torch_gpu_memory_bytes']/1024**2:.1f}MiB；可见观测缓存{result['association_cache_bytes']/1024**2:.1f}MiB。不能把该显存当成驱动/进程总占用，或把总墙钟时间称为纯融合内核速度。每帧与各组状态存储量有单独记录。

## 5. 去重、保存恢复及审计

每个帧更新后立即用同一身份/版本再调用，均为无操作；180帧最终账本长度一致，支持计数不超过180。每30帧保存五组状态，完成后逐个加载30/60/90/120/150/180状态，核对对应身份、账本长度和支持；180帧checkpoint与最终状态逐数组一致。从真实最终状态恢复后，重放最后一个实际缓存帧也不改变证据、支持或累计权重。

20视角几何支持使用独立全1数值通道渲染alpha作为所有组共有分母，避免类别分批浮点求和造成阈值附近的分母漂移；语义通道总质量与该alpha必须满足01固定容差，否则报错。新增3项CPU契约检验共同支持阈值、alpha不一致拒绝和硬拒绝情况下仍保持帧账本。

正式渲染CLI使用相同的独立几何alpha，对S2最终状态的原始3099视角输出全部数组与批量运行逐值相同，补充证据见`cli_shared_alpha_check_20261007_v1/validation.json`。更新后的公共CLI与全量运行内的原始源码快照分别保存，不覆盖原始运行记录。

上述保存恢复证明完整帧状态可安全重放；未实现中途半帧重新接续。发生中断时只使用所有五组均完整且核验通过的同一帧数checkpoint，半帧缓存不得视为已提交。后台主运行未单独捕获操作系统退出码；完整结果、终端日志及后续独立核验保存，不能补造退出回执。小规模命令退出码0已记录。

## 6. 诊断图与归档

![原始3099视角工程对比](dev_sgbm_native_full_20261007_v1/figures/3099_comparison.png)

三张对比图3009/3099/3199包括RGB、二维教师和五组渲染；紫色unknown、黑色无地图支持。图中文字标明已曝光工程DEV与非GT边界。

![教师不一致覆盖诊断](dev_sgbm_native_full_20261007_v1/figures/teacher_disagreement_coverage.png)

全部缓存、六个阶段checkpoint和最终五组状态、100个渲染、源码/配置/日志/哈希及小规模运行保留，并以完整包归档。证据文件清单见`native_dev_evidence_SHA256_20261007.json`，本地及服务器核验一致；图表不能代替这些原始证据。

## 7. 当前门槛及下一步

**工程范围已完成；语义质量范围未通过验收。** 现在独立语义GT真正阻塞O-Sem、GT mIoU、真实风险—覆盖、小目标和边界子组，以及科学方法选择；这些项目明确标为未执行，不能由教师结果顶替。既有计划允许无GT时交付工程与教师诊断，本次按该降级范围执行。

还复查了`stage2_tsdd/data/annotations`内的seq08人工标注包，回执见`annotation_provenance_check_20261007.json`。其中标注任务是物体在世界坐标中是否运动，实例候选类别与mask来自Mask2Former；它们不能直接作为本轮独立像素语义GT，也不据此重开阶段二或改正式评价结果。

后续可固定已验证的工程配置并整理展示/复现；若要进入03的科学冻结与隔离评价，应先获得与目标窗口匹配的独立语义GT或另立可评价数据协议。未来采用新深度/GT或重新选窗口，必须版本化、重跑共同输入比较，不把当前已曝光数据恢复称为新Test。尚无证据触发S2-Z/S2-V或学习扩展，不启动这些候选。
'''
(step/'02_执行报告_20261007.md').write_text(report,encoding='utf-8')
(step/'README.md').write_text('''# 02 基础因子DEV与Oracle诊断

2026-10-07：五组180帧真实融合、20视角工程诊断已完成，状态保存/恢复和共同支持核验通过。最新报告`02_执行报告_20261007.md`，全量证据`dev_sgbm_native_full_20261007_v1`，小规模试运行另存。

独立GT质量验收未完成，O-Sem及GT风险/子组未执行。SGBM为明确的新工程DEV深度版本，不与历史RAFT配置混称。教师一致性不是科学准确率或方法排名。
''',encoding='utf-8')
progress=root/'执行进度与下一步.md';text=progress.read_text(encoding='utf-8')
text=text.replace('**00审计和01原生工程核验已完成；下一步进入02融合准备。**','**00、01工程验收完成；02五组真实工程DEV已完成，独立GT质量验收尚未完成。**')
text=text.replace('现存SGBM仅是候选，不自动替代历史RAFT。','02已明确采用SGBM作为新的工程DEV输入版本，不替代历史RAFT配置。')
marker='02最新报告见' if '02最新报告见' in text else '下一步顺序：02可见关联'
cut=text.index(marker)
text=text[:cut]+'''02最新报告见`02_基础因子DEV与Oracle诊断/02_执行报告_20261007.md`。已使用明确的新SGBM工程输入版本执行180帧五组融合、20视角×五组渲染、可见贡献缓存、30帧间隔checkpoint及实际重复恢复核验。教师一致性/覆盖是诊断，不是GT准确率；原地图和相机未改，180/20仍是已曝光DEV角色。

独立GT现在实际阻塞O-Sem、真实语义mIoU、风险—覆盖、关键GT子组和科学方法选择。02工程交付完成不等于科学验收通过。可整理工程配置和展示复现，但03科学冻结需独立语义评价入口。尚无依据启动S2-Z/S2-V或学习扩展；Stage2冻结与正式FAIL边界保留。
'''
progress.write_text(text,encoding='utf-8')
obs=root/'protocol/observation_schema.json';j=json.loads(obs.read_text(encoding='utf-8'))
j['association_cache']='native color adjoint alpha/transmittance per-Gaussian footprint; real 180-frame engineering cache verified'
j['frame_aggregation_candidate']='implemented shared alpha-weighted hard class histogram, normalized within source frame; unit budget'
j['depth_input_version']='SGBM_ENGINEERING_NEW_VERSION; not historical RAFT replacement'
obs.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
protocol=root/'protocol/02_工程DEV冻结配置_20261007.json';protocol.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=[step/'02_执行报告_20261007.md',step/'README.md',step/'五组工程诊断汇总_20261007.csv',progress,obs,protocol,
    Path(__file__),root/'scripts/run_step02_dev.py',root/'scripts/package_step02_remote.py',root/'scripts/test_step02_contracts.py',root/'scripts/native_semantic_render.py',root/'scripts/download_evidence_parallel.py']
files += [step/name for name in ['depth_search_20261007.json','smoke_command_20261007.json','full_dispatch_command_20261007.json','full_completion_observation_20261007.json','remote_contracts_command_20261007.json','package_command_20261007.json','full_result_snapshot_20261007.json','annotation_provenance_check_20261007.json','archive_range_hashes_20261007.json','archive_integrity_20261007.json','cli_reproduction_command_20261007.json','cli_archive_command_20261007.json']]
files += [p for p in (step/'cli_shared_alpha_check_20261007_v1').rglob('*') if p.is_file()]
hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(step/'completion_manifest_20261007.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
bundle=step/'step02_completion_sync_20261007_v2.tar.gz'
with tarfile.open(bundle,'w:gz') as t:
    for p in files+[step/'completion_manifest_20261007.json']:t.add(p,arcname=p.relative_to(root).as_posix())
print(json.dumps({'evidence_files_verified':len(manifest),'completion_files':len(files),'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()},ensure_ascii=False))
