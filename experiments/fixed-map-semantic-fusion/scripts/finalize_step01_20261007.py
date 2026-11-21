"""Build Step01 report and sync bundle from verified native audit evidence."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
step=root/'01_语义表示_初始化与渲染审计';run=step/'native_audit_20261007_v1'
result=json.loads((run/'audit_result.json').read_text(encoding='utf-8'))
assert result['status']=='PASS_STEP01_NATIVE_ENGINEERING'
manifest=json.loads((run/'SHA256_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((run/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
assert json.loads((run/'cli_validation.json').read_text())['cli_equals_audited_helper']
cfg=result['config'];groups=result['groups'];frames=result['real_frames']
table='\n'.join(f"| {g['group']} | 通过 | 通过 | {max(g['rgb_max_abs_errors']):.1f} |" for g in groups)
ftable='\n'.join(f"| {f['original_frame']} | {f['supported_pixels']:,} | {f['no_support_pixels']:,} | {f['alpha_max_error_artificial']:.3g} |" for f in frames)
report=f'''# 01完成报告：语义表示、初始化与原生渲染审计

日期：2026-10-07；负责人：CHY。**01工程验收通过。** 本轮使用真实907,488点固定基图和SplaTAM原生CUDA渲染器，完成五组重新初始化、对应身份及参数一致性验证、RGB数值比较、unknown遮挡与数值语义通道测试。尚未进行真实语义融合或语义质量评价。

## 1. 输入、代码与运行范围

基图：`/root/autodl-tmp/stage3_chy_20261006/inputs/base_map/params.npz`；SHA256：`{result['base_sha256']}`。运行前核验全部15字段数组哈希，运行后基图文件哈希不变。

原生运行目录：`01_语义表示_初始化与渲染审计/native_audit_20261007_v1`。相机与几何沿用冻结基图，固定最终地图、所有Gaussian参与，不根据历史timestep裁剪。代表视角为本地0/99/199，对应原始3000/3099/3199；这里只验证工程行为，不改变180/20融合划分。

环境：Python {result['python']}，PyTorch {result['torch']}，CUDA {result['cuda']}，{result['gpu']}。主审计耗时{result['wall_seconds']:.2f}秒，PyTorch峰值分配{result['peak_gpu_memory_bytes']/1024**2:.1f}MiB（不代表所有驱动或进程显存）。完整环境、结果见`audit_result.json`。

审计使用安装的`diff_gaussian_rasterization`及其CUDA扩展，原生相机/变换助手、安装包装器与扩展二进制均已保存并记录哈希，见`source_manifest.json`和`source_snapshots`。当前行为由实测及二进制版本绑定；不据此补造历史构建提交。

## 2. 五组初始化、对应及RGB不变

| 组别 | 语义清空及身份保存恢复 | 全部参数逐值一致 | 三视角RGB最大绝对差 |
| --- | --- | --- | --- |
{table}

每组重新生成独立`semantic_state.npz`，以原始`source_gaussian_index`一一对应；类别证据、不同帧支持计数、累计权重为0，已处理观测账本为空。历史一致性关闭，没有导入旧类别或历史冲突字段。共享不可变基图，每组独立内存副本逐字段比较；渲染器只取几何/RGB/opacity/相机字段，基图历史语义字段未用作初始化。

RGB基准直接调用原生`transformed_params2rendervar`；各组经新的适配器渲染。同相机、同几何、同RGB与opacity、同渲染设置比较，运行前冻结容差`atol={cfg['rgb_atol']}, rtol={cfg['rgb_rtol']}`，15次比较最大差均为0。本轮不以RGB质量改善为收益。此结果覆盖三视角，不声称200视角全量核验。

## 3. 原生遮挡与unknown规则

小场景在同一视线上放置前景unknown Gaussian和背景道路类Gaussian，使用原生投影足迹、深度排序、opacity与透射率，不采用中心投影直接赋类。

中心像素unknown质量为{result['controlled'][0]['center_unknown_mass']:.6f}，背景类别质量仅{result['controlled'][0]['center_background_class_mass']:.6f}，输出unknown。残余背景质量来自原生透明度混合，并非跳过前景。故意把unknown opacity清零的错误对照输出背景道路，复现旧实现漏洞。交换Gaussian数组顺序后输出完全一致，检验实际深度排序不依赖输入数组顺序。

有几何但无语义证据时，unknown有独立数值通道，**保留原opacity及遮挡**；原生足迹内总alpha低于0.05则输出无地图支持。全unknown场景与低alpha场景已单独核验。二维GT的ignore属于评价协议处理，不控制Gaussian是否参与遮挡。

当前工程解码规则：alpha≥0.05为有支持；有支持默认unknown；仅在类别质量非零且unknown质量/总alpha<0.5时输出类别argmax。标签`unknown=-1`、`no_map=-2`、正常类0–18，分数为最大类别质量/总alpha。阈值在本次测试前记录，仍属工程默认，不是通过DEV选择的科学方法参数。

## 4. 数值类别通道与原生alpha

19个类别加unknown共20通道，补齐为21通道，原生每次3通道、共7次前向渲染。每次共享相机、几何、排序、opacity与足迹；背景必须为零。类别从数值质量解码，调色板只可用于展示，不参与类别计算。

受控混合证据测试中，20通道总质量与一次原生全1特征渲染的alpha最大差为{result['controlled'][3]['max_error']:.3g}。真实地图同时测试空语义及以稳定ID生成的人工类别探针，后者只是跨通道工程输入，不是预测、GT或方法结果。

| 原始视角 | alpha≥0.05像素 | 无地图支持像素 | 人工通道总质量与原生alpha最大差 |
| --- | --- | --- | --- |
{ftable}

冻结alpha容差：`atol={cfg['alpha_atol']}, rtol={cfg['alpha_rtol']}`。空语义状态在有支持像素全部输出unknown，无类别质量；三视角空状态alpha差均为0。原生CLI已实际运行视角99，输出逐数组与审计辅助函数完全相同，见`cli_validation.json`。

## 5. 一个观测、帧内汇总与保存恢复

源观测身份是dataset/sequence/camera/original_frame，输入版本绑定文件SHA256，Gaussian对应由基图版本和稳定ID约束。一个源帧先汇总所有Gaussian观测，再整体提交一次；不能把像素块或优化器访问次数当成新的源帧。同一帧同版本再次提交为无操作；版本变化要求新run；证据和帧账本一同原子保存恢复。

支持计数按每个Gaussian的不同有效源帧累计。已实现的共同帧内规则为：使用可见alpha贡献对硬标签汇总类别分布，帧内归一化，单帧基础支持预算为1，随后应用组别权重；大投影面积本身不能把一帧膨胀为更多独立支持。所有组共用该规则，旧中心采样实现作为历史基线保存，不仅给S2改规则。实际可见贡献缓存与真实输入融合尚未执行，不把合成契约当成真实关联有效性结论。

缓存只有硬标签和选中类标量证据比，类别统计采用硬标签投票；标量仅作可靠性候选，不冒充完整19类分布。KITTI08推理回执已在00绑定。历史一致性关闭，后续观测与历史类别不同不自动拒绝。

本地与服务器10项CPU契约测试均通过，涵盖重复帧、恢复和错序ID拒绝、帧内面积归一化、无效输入不提交、共享基础检查、unknown计分、并列风险排序以及关闭历史门控后纠正早期错误。CPU与原生测试作用不同，两类记录分别保留。

## 6. 产物与下一步

本目录保存原始命令回执、冻结配置、机器结果、三视角RGB、空语义数组、人工探针数组、受控场景数组/图、五组新初始化、代码/二进制快照及逐文件SHA256。完整原生证据包`native_audit_20261007_v1.tar.gz` SHA256为`b45d347785b00f7813eae31f6181759073f865e1079aacd9f4fda06830bbf4f0`；本地与服务器逐文件校验一致。

01完成范围是初始化与渲染工程验收；下一步进入02的可见关联、帧内观测及真实融合准备，然后按计划做基础因子对照。尚无真实五组融合质量、方法收益或科学接受结论。缺失资产按用户要求在真正阻塞执行时再提醒，不提前要求补数据。
'''
(step/'01_完成报告_20261007.md').write_text(report,encoding='utf-8')
(step/'README.md').write_text('''# 01 语义表示、初始化与渲染审计

2026-10-07状态：**原生工程验收通过**。最新报告：`01_完成报告_20261007.md`；实际CUDA证据：`native_audit_20261007_v1`。五组初始化和三视角RGB一致性、unknown遮挡及20通道alpha核验通过。10/06仅CPU的报告保留为历史记录。

实际语义融合、可见观测缓存及语义质量评价归后续步骤，不以01工程PASS替代方法有效性。
''',encoding='utf-8')
progress=root/'执行进度与下一步.md'
text=progress.read_text(encoding='utf-8')
text=text.replace('**00输入、唯一基图与GT曝光审计已完成，有明确限制；下一步进入01工程核验。**','**00审计和01原生工程核验已完成；下一步进入02融合准备。**')
start=text.index('01已有CPU契约与适配代码准备')
text=text[:start]+'''01最新结果见`01_语义表示_初始化与渲染审计/01_完成报告_20261007.md`：五组从零重新初始化，Gaussian对应和全部参数一致；原生unknown前景遮挡测试通过；三个真实视角共15次RGB比较最大差0，20数值通道总质量与原生alpha最大差约1.1e-6。本地/服务器10项CPU契约均通过。已使用GPU完成01核验，尚未执行真实五组语义融合。

下一步顺序：02可见关联、帧内观测与真实融合准备 → 基础五组DEV → 03方法冻结与隔离评价。缺失资产按用户要求在真正阻塞后续执行时再提醒。S2-Z/S2-V及学习扩展仍按触发条件启动，Stage2冻结及正式FAIL边界保留。
'''
progress.write_text(text,encoding='utf-8')
obs=root/'protocol/observation_schema.json';j=json.loads(obs.read_text(encoding='utf-8'))
j['provenance_scope']='KITTI08 generation receipt and model hash bound by Step00 20261007; scalar field semantics audited'
j['association_cache']='NOT_CREATED; Step01 native channel rendering verified, per-Gaussian visible observation cache belongs to fusion preparation'
obs.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rules=root/'protocol/rendering_and_scoring_rules.md';text=rules.read_text(encoding='utf-8')
text+='''
## 2026-10-07 原生工程核验

01原生CUDA验收通过，报告见`../01_语义表示_初始化与渲染审计/01_完成报告_20261007.md`。unknown保留opacity，数值通道alpha一致，五组同相机RGB在预先设定容差内一致。工程阈值仍为alpha_min=0.05、unknown_fraction_max=0.5，尚未作科学DEV参数冻结。实际可见关联与融合未由此次渲染核验代替。
'''
rules.write_text(text,encoding='utf-8')
files=[step/'01_完成报告_20261007.md',step/'README.md',progress,obs,rules,
    root/'scripts/native_semantic_render.py',root/'scripts/audit_step01_native.py',Path(__file__),
    step/'cpu_contract_tests_20261007.json']
hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(step/'completion_sync_manifest_20261007.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
bundle=step/'step01_completion_sync_20261007_v1.tar.gz'
with tarfile.open(bundle,'w:gz') as t:
    for p in files+[step/'completion_sync_manifest_20261007.json']:t.add(p,arcname=p.relative_to(root).as_posix())
print(json.dumps({'verified_native_files':len(manifest),'completion_files':len(files),'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}))
