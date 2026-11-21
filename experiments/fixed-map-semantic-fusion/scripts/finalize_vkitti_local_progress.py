"""Update live navigation after verified GT-DEV delivery; retain old reports."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];step=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2';ev=step/'GT_evaluation_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,s):p.write_text(s,encoding='utf-8')
assert read(step/'local_integrity_and_GT_recount.json')['status']=='PASS_LOCAL_FULL_HASH_AND_INDEPENDENT_GT_RECOUNT'
decision=read(ev/'DEV_candidate_decision.json');oracle=read(ev/'O_Sem_diagnostic.json')
report=step/'VKITTI2_GT_DEV补齐执行报告_20261007.md'
write(step/'README.md','''# VKITTI2 GT-DEV v2

两窗口52帧CPU适配、固定GT pose/SGBM建图、五组与O-Sem融合、10个评价视角的独立官方GT评价完成；两个窗口各有小测试。完整870份证据下载后哈希与独立布尔计数复算通过。

报告：[VKITTI2_GT_DEV补齐执行报告_20261007.md](VKITTI2_GT_DEV补齐执行报告_20261007.md)。指标、共同子组、完整风险曲线与判定见GT_evaluation_v1。所有加权候选未满足预声明DEV条件，保留S0；负结果不是工程失败。

曝光身份：历史DEV。原KITTI08仍缺稠密GT，本轮合成场景结果不能替代。隔离最终评价未执行。重跑新建版本，禁止覆盖本目录。

数据访问说明：下载并核验整个官方类别归档；未提取、解码或用于实验/选择Scene20及旧正式Test样本。复用源代码与固定模型另有03来源补充记录。
''')
progress=root/'执行进度与下一步.md';text=progress.read_text(encoding='utf-8')
old='**00、01、02工程部分完成；03工程冻结与独立复算完成，独立GT科学验收及隔离最终评价未执行。** 最新报告见`03_方法冻结与隔离评价/03_执行报告_20261007.md`。'
new='**原KITTI08工程完成；02新增VKITTI2独立GT DEV评价已完成，03新增DEV判据/候选判定与规则结果封存。隔离最终评价尚未执行。** 最新报告见`02_基础因子DEV与Oracle诊断/gt_dev_v2/VKITTI2_GT_DEV补齐执行报告_20261007.md`。'
text=text.replace(old,new)
history='\n## 原KITTI08执行记录（以下为VKITTI2补齐前状态）\n'
if history not in text:text=text.replace('\n本地00–06目录',history+'\n本地00–06目录',1)
marker='\n## VKITTI2独立GT DEV补齐（2026-10-07）\n'
if marker in text:text=text.split(marker)[0]
text+=marker+f'''
官方classSegmentation包补齐并通过官方MD5与SHA256锁定。Scene01/clone 330–355、Scene02/clone 110–135共52帧；每窗口统一GT pose和预测SGBM建固定图，21帧语义融合、5帧排除视角评价。几何见全部26帧、均为历史DEV，不称完整held-out重建。

五组可部署方法与DEV O-Sem实际执行完成。真实GT mIoU、覆盖/unknown/no-map、共同小车辆/边界/时序代理子组及精确同分风险曲线完成。52帧输入与pose后置核验通过；完整870份证据镜像、本地哈希和60份渲染独立GT布尔复算通过。

首次GT指标读取前预声明判据，四个加权候选均未通过；当前DEV参考{decision['selected_DEV_reference']}，不声称加权改善。S1平均近乎持平，S-G/S2略降，S2-H在Scene01损失mIoU、覆盖及部分关键子组。O-Sem从37.08%→61.17%、13.26%→44.51%，支持输入语义质量是限制之一，仍存在地图支持与关联等残余问题。

03规则/结果封存见`gt_dev_rules_freeze_20261007_v2`；DEV判优政策见protocol同名JSON，最终科学接受协议和隔离入口仍待审定。Scene20和旧正式Test样本未提取/解码或用于实验选择（整包下载和哈希校验已执行）；不重开Stage2、不启动MLP或未触发候选。下一步可04工程展示与归档，或另审新数据的隔离最终评价；不可把这两窗口改名Test。
'''
write(progress,text)
third=root/'03_方法冻结与隔离评价'
write(third/'03_GT_DEV补齐状态_20261007.md','''# 03 GT-DEV补齐状态

旧KITTI08封存继续保留。新增VKITTI2 GT评价规则、首次GT指标读取前的DEV判优政策、五组判定与结果哈希已封存，目录gt_dev_rules_freeze_20261007_v2。

02已完成两个历史DEV窗口的独立官方GT评价及O-Sem；所有加权候选未满足预声明条件，保留S0参考，Oracle不参与方法排名。主结果见02/gt_dev_v2报告，误差/覆盖/子组与风险曲线全部保存。本地870份证据完整性及60份渲染独立复算通过。

仍未完成：隔离最终评价及其最终科学接受协议。没有新入口，旧DEV不能恢复成Test；不宣称科学最终PASS或加权算法收益。原KITTI08缺GT也未被合成场景结果替代。
''')
write(third/'README.md','''# 03_方法冻结与隔离评价

旧KITTI08工程冻结与复算完成。新增VKITTI2 GT-DEV规则、预声明判据、候选判定与结果封存完成；隔离最终评价尚未执行。

最新状态：[03_GT_DEV补齐状态_20261007.md](03_GT_DEV补齐状态_20261007.md)。旧报告03_执行报告_20261007.md及engineering_freeze_20261007_v1不改；新封存gt_dev_rules_freeze_20261007_v2完整保留负结果。四个加权候选未过DEV条件，S0作为下一轮参考，不代表质量提升。
''')
second=root/'02_基础因子DEV与Oracle诊断/README.md';previous=second.read_text(encoding='utf-8')
link='\n## VKITTI2 GT-DEV补齐（2026-10-07）\n'
if link in previous:previous=previous.split(link)[0]
write(second,previous+link+'\n五组、O-Sem及真实GT评价完成，详情见`gt_dev_v2/VKITTI2_GT_DEV补齐执行报告_20261007.md`。所有加权候选未满足预声明条件，保留S0。原KITTI08工程版本与缺GT边界继续保留；隔离最终评价未执行。\n')
files=[step/'README.md',step/'local_integrity_and_GT_recount.json',progress,third/'README.md',third/'03_GT_DEV补齐状态_20261007.md',second,Path(__file__),root/'scripts/verify_vkitti_gt_delivery_local.py']
files += [root/n for n in ['vkitti_live_inventory_20261007.json','vkitti_download_dispatch_20261007.json','vkitti_pipeline_dispatch_20261007.json','vkitti_quality_preregistration_receipt_20261007.json','vkitti_gt_evaluation_status_20261007.json','vkitti_dev_candidate_selection_20261007.json','vkitti_GT_contracts_remote_20261007.json','vkitti_gt_results_snapshot_20261007.json','vkitti_package_completion_20261007.json']]
files += [root/'vkitti_reused_source_provenance_20261007.json']
files += [p for p in (third/'gt_dev_reused_sources_supplement_20261007').rglob('*') if p.is_file()]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};mp=step/'local_completion_SHA256.json';write(mp,json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
archive=step/'local_completion_sync_20261007.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files+[mp]:t.add(p,arcname=p.relative_to(root).as_posix())
print(json.dumps({'files':len(files),'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},ensure_ascii=False))
