"""Write Step03 reports after remote replay and local independent recount pass."""
import hashlib,json,tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
step=root/'03_方法冻结与隔离评价'
freeze=step/'engineering_freeze_20261007_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.write_text(s,encoding='utf-8')
audit=read(freeze/'audit/audit_result.json')
replay=read(freeze/'full_cache_replay.json')
local=read(step/'local_independent_recount_20261007.json')
assert audit['status']=='PASS_ENGINEERING_FREEZE_AND_INDEPENDENT_RECOUNT'
assert replay['status']=='PASS_FULL_180_FRAME_REPLAY'
assert local['exit_code']==0 and 'PASS_CROSS_HOST_RECOUNT' in local['stdout']
assert all(sha(freeze/'core'/n)==h for n,h in read(freeze/'core_SHA256.json').items())
rows=read(freeze/'audit/aggregate_teacher_recount.json')
table='\n'.join(f"| {r['group']} | {r['coverage']*100:.4f}% | {r['conditional_teacher_agreement']*100:.4f}% | {r['unknown_fraction']*100:.4f}% | {r['no_map_fraction']*100:.4f}% |" for r in rows)
report=f'''# 03执行报告：工程规则冻结与独立复算

日期：2026-10-07；负责人：CHY。**03可执行的工程冻结与复算完成；独立GT科学方法选择及隔离最终评价未执行。** 五组全部保留，不依据教师一致性指定优胜方法。

## 1. 冻结对象和曝光边界

冻结目录：`engineering_freeze_20261007_v1`。冻结的是02运行前已有的工程配置、生产源码与实际产物，03没有在看到结果后修改权重或拒绝阈值。独立GT的改善门槛、覆盖预算、关键子组约束仍为null，不能称科学协议已经全部冻结。

唯一基图907,488个Gaussian，文件SHA256：`3244884f5a22ebfc6cf94b4ec0e48ddb4f666f5803938bc5e53cc4c62ae6031b`。全部参数字段、稳定ID、五组最终状态、180份关联缓存、100份语义渲染逐文件锁定。生产渲染快照与后来通过共享alpha核验的公共CLI分开记录。

KITTI08原始3000–3199全部被几何、RGB及历史语义使用过。180帧只用于本轮语义融合，20帧只从本轮融合中排除，且已在02消费为DEV；03只复算这些旧结果，没有产生独立最终评价集。`core/frame_roles.json`的最终评价列表为空。清空语义不撤销历史曝光。

深度继续使用明确版本化的SGBM工程输入；它不是原RAFT实验的恢复。五组保持S0、S1、S-G、S2、S2-H，历史一致性OFF，固定几何、RGB、opacity及相机，不调整深度。S0只是复现参考，不是质量获胜者。

## 2. 实际验证

- 服务器CPU从180份原生可见贡献缓存重新创建五组零证据状态；最终类别证据、支持计数、累计权重均与02最终状态**逐值完全相同**，源帧去重账本一致。耗时{replay['seconds']:.1f}秒，无GPU。此项重放缓存，不重新生成可见性。
- 核验唯一基图文件及全部参数数组、五组稳定ID、180个精确融合帧身份、证据总量与权重守恒。
- 用独立19×21整数混淆计数重新统计100份渲染：19列类别，加unknown、无地图支持两列。逐视角计数与02完全一致，汇总浮点指标差小于1e-12；五组共享地图支持掩码逐像素一致。
- 本地从下载后的冻结副本再次运行`--verify-only`，核验封存文件、基图/状态与100份渲染，并再次独立计数；结果`PASS_CROSS_HOST_RECOUNT`。本地没有依赖服务器原预测标签路径。
- 将20个视角分成四个连续轨迹区块，各五视角，仅作描述；它们来自同一场景同一窗口，不能当独立重复实验或生成独立场景置信区间。

## 3. 教师诊断复算表

下表的教师是二维模型预测，**不是独立GT**；条件一致性只在有类别输出像素上计算。覆盖的分母是完整图像，unknown与无地图支持均保留在分母中。

| 组别 | 全图有标签覆盖 | 条件教师一致性 | unknown | 无地图支持 |
|---|---:|---:|---:|---:|
{table}

每组20视角，共9,072,400像素。整数混淆矩阵、逐视角结果、四区块表分别见`audit/teacher_confusion_matrices.npz`、`audit/per_view_teacher_recount.json`、`audit/correlated_block_descriptive.csv`。这些数字能证明实现与复算一致，不能证明真实语义准确率改善。S2-H较高条件一致性伴随覆盖减少，不能据此宣布整体更好。

## 4. 未执行项目与进入条件

`core/scientific_gate.json`标记`NOT_ELIGIBLE_FOR_SCIENTIFIC_FREEZE_OR_FINAL_EVALUATION`。独立语义GT、隔离最终评价入口均未具备；O-Sem、地图渲染的二维语义mIoU、GT风险—覆盖、小目标/边界/稳定表面/动态遮挡共同子组均未执行。详细入口清单见`独立评价入口缺项说明.md`。

旧20视角不可重新命名为Test。以后取得标注或另审新窗口，必须先明确曝光角色、GT映射与ignore、目标地图及共同子组，再在DEV确定判优门槛与风险—覆盖规则，封存后才消费隔离最终评价。没有证据启动S2-Z/S2-V、MLP或历史反馈，也不重开Stage2。

可以继续04工程展示、复现和归档；科学评价仍等待上述入口。02和03的工程完成不能改写为科学PASS。

## 5. 复现与证据

服务器：`/root/autodl-tmp/stage3_chy_20261006/03_方法冻结与隔离评价`。本地同名目录保留封存、服务器命令结果、本地复算回执和下载包哈希。

复算命令：`python 阶段3/scripts/freeze_step03_cpu.py --root 阶段3 --verify-only`。完整缓存重放已记录在`full_cache_replay.json`，不必重复GPU实验。`core_SHA256.json`及`freeze_record.json`锁定核心；完成包另附逐文件SHA256，覆盖报告、协议、脚本和回执。修改方法时另建版本，不覆盖本次封存。
'''
write(step/'03_执行报告_20261007.md',report)
write(step/'独立评价入口缺项说明.md','''# 03科学评价入口缺项与执行门槛

当前状态：NOT_EXECUTED；原因是独立语义GT不可用，且没有已审定的隔离最终评价集合。工程复算已经完成，不能代替以下科学步骤。

| 项目 | 当前状态 | 继续执行前需要 |
|---|---|---|
| 独立二维语义GT | 未发现可用资产 | 来源、许可、逐文件哈希、像素映射、ignore及帧对应；模型预测不是GT |
| 目标地图 | 当前固定离线最终地图，含历史车辆/遮挡风险 | 明确静态长期表面或时刻快照，共同GT定义稳定与变化区；各方法共用分母 |
| DEV与最终评价角色 | 180融合、20已消费DEV，全部历史曝光 | 审计新入口；如仅语义未见，明确几何仍可能已见；完整重建held-out需排除几何/RGB/depth及缓存曝光 |
| O-Sem | 缺GT跳过 | 仅DEV融合帧GT输入，另设DEV评价视角；预测基线使用同一有GT帧子集 |
| 地图渲染的二维语义mIoU | 未运行 | 冻结有效GT像素与类别；unknown按真实类漏判，不能排除；无地图支持另报并锁定主指标处理，不填随意类别 |
| 风险—覆盖 | 未运行 | DEV确定最终证据排序函数、并列规则与共同可达覆盖；固定评价分母、无地图支持不能伪造高覆盖 |
| 小目标/非小目标、边界/内部 | 无独立GT分组 | 共同GT或冻结外部规则；禁止按各组预测Static/拒判掩码缩小分母 |
| 判优门槛 | 改善量、覆盖退化预算、关键子组约束为null | 在有GT的DEV中确定并封存；不能用已见教师一致性挑有利门槛 |
| 统计与资源约束 | 单场景单窗口，仅描述性区块 | 足够独立场景/窗口支持统计，冻结实际预算与硬件记录；连续帧不能当独立样本 |

执行次序：取得并审计GT/曝光 → 确定DEV和最终评价角色 → 冻结评价规则及DEV判优门槛 → 科学方法封存 → 一次隔离评价并保留负结果。

当前无需为03工程归档重新生成RAFT。若科学协议决定改用RAFT或其他输入，应新建版本，所有五组使用一致输入并重新融合。已有SGBM版本完整保留。
''')
write(step/'README.md','''# 03_方法冻结与隔离评价

**工程规则冻结、180帧五组CPU重放、100份渲染独立复算及本地再次核验完成；独立GT科学冻结和隔离最终评价未执行。**

主报告：[03_执行报告_20261007.md](03_执行报告_20261007.md)。入口缺项：[独立评价入口缺项说明.md](独立评价入口缺项说明.md)。核心封存：`engineering_freeze_20261007_v1/core`；审计结果：同目录`audit`。

保留全部五组，不按教师预测一致性选优。旧20视角仍为已消费DEV；未来修改方法或数据另建版本，不改封存。可以继续04工程整合与归档。
''')
protocol=root/'protocol'
previous=protocol/'实验协议_v0.1_历史快照_20261007.md'
if not previous.exists():previous.write_bytes((protocol/'实验协议_v1.md').read_bytes())
write(protocol/'实验协议_v1.md','''# 阶段3工程执行协议 v1

状态：ENGINEERING_RULES_SEALED_ALL_FIVE；SCIENTIFIC_SELECTION_AND_FINAL_EVALUATION_NOT_EXECUTED。2026-10-07，CHY。旧v0.1单独保留为历史快照。

以`03_方法冻结与隔离评价/engineering_freeze_20261007_v1/core/frozen_method_config.json`及核心哈希为唯一机器可读工程配置。封存02运行前配置，无结果后调参。907,488点基图、稳定ID、180帧当前融合、20帧已消费DEV及全部200帧历史曝光详见核心清单。

共同原生可见性/有效输入检查；帧内alpha加权硬标签分布归一化，单位源帧预算；重复源帧不累计，支持计不同有效帧。置信度是选中类别标量分数，不是完整概率分布。S0基础、S1语义、S-G几何、S2二者乘积、S2-H乘积≥0.5单位票；固定绝对深度容差1米，SGBM新工程版本，历史反馈OFF。unknown保留遮挡；地图支持使用独立原生all-one alpha≥0.05，共享分母。原生源码、依赖与01/02核验保留。

工程诊断报告全图覆盖、unknown、无地图支持及条件教师一致性，教师来自模型预测，不是GT。不修改几何/RGB/opacity/相机，不声称RGB改善、在线SLAM或跟踪收益。

科学判优最低改善量、覆盖退化预算、关键子组约束均PENDING/null；尚无独立GT、共同GT子组或隔离最终评价。GT指标及风险—覆盖排序/并列/共同覆盖规则需未来DEV确定。旧20视角不能升级为Test，不用教师一致性宣布科学最佳方法。

科学恢复入口与执行顺序见03目录`独立评价入口缺项说明.md`。Stage2冻结及正式FAIL边界继续保留；S2-Z/S2-V、历史反馈和MLP未启动。当前允许工程展示和归档。
''')
write(protocol/'方法冻结记录.md',f'''# 03方法冻结记录（工程）

状态：SEALED_ENGINEERING_CONFIGURATION_ALL_FIVE，2026-10-07，CHY。不是科学方法选择或正式PASS。

核心清单SHA256：`{sha(freeze/'core_SHA256.json')}`。机器记录见03封存目录`freeze_record.json`。所有五组配置、帧角色、源码、依赖、基图及状态/缓存/渲染哈希已经封存；S0只作为复现参考，quality_winner=null。

服务器180帧五组CPU缓存重放逐值相等；100份渲染独立计数与02一致；本地再次核验通过。科学改善门槛及GT评价未签发，详见`scientific_gate.json`。修改输入、权重、规则或科学协议另建版本，不能覆盖核心清单。报告与同步回执是封存之外的证据补充，不修改核心。
''')
progress=root/'执行进度与下一步.md'
text=progress.read_text(encoding='utf-8')
text=text.replace('**00、01工程验收完成；02五组真实工程DEV已完成，独立GT质量验收尚未完成。** 最新报告见`00_输入与基图_GT曝光审计/00_审计完成报告_20261007.md`。','**00、01、02工程部分完成；03工程冻结与独立复算完成，独立GT科学验收及隔离最终评价未执行。** 最新报告见`03_方法冻结与隔离评价/03_执行报告_20261007.md`。')
text=text.replace('已使用GPU完成01核验，尚未执行真实五组语义融合。','已使用GPU完成01核验；真实五组语义融合已在02执行。')
marker='\n## 03工程冻结与独立复算（2026-10-07）\n'
if marker in text:text=text.split(marker)[0]
text+=marker+'\n封存02原工程配置及五组源码/状态/关联缓存/100渲染哈希。服务器CPU重新融合180帧，五组证据、计数、权重逐值相同；独立整数混淆计数逐视角与02一致，本地复算再次通过。无GPU、无方法调参、无科学优胜者；旧20视角仍为已消费DEV，最终评价列表为空。\n\n科学冻结/最终评价因独立GT与隔离入口缺失未执行；主指标、风险—覆盖及GT子组/判优门槛仍待独立DEV定义。可以进入04工程展示和归档，不能宣称科学PASS，不启动未触发扩展。\n'
write(progress,text)
files=[p for p in freeze.rglob('*') if p.is_file()]
files += [step/n for n in ['README.md','03_执行报告_20261007.md','独立评价入口缺项说明.md','local_independent_recount_20261007.json','remote_freeze_command_20261007.json','download_package_SHA256_20261007.json']]
files += [step/'remote_package_command_20261007.json',progress,Path(__file__),root/'scripts/freeze_step03_cpu.py',root/'scripts/unpack_step03_and_verify.py',protocol/'实验协议_v1.md',protocol/'实验协议_v0.1_历史快照_20261007.md',protocol/'方法冻结记录.md']
manifest={p.relative_to(root).as_posix():sha(p) for p in files}
mp=step/'completion_SHA256_20261007.json';write(mp,json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
archive=step/'step03_completion_20261007_v1.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in files+[mp]:t.add(p,arcname=p.relative_to(root).as_posix())
print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size},ensure_ascii=False))
