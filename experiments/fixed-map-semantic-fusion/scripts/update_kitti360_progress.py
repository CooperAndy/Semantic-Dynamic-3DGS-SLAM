"""Update live navigation only after local verification; preserve sealed history."""
import json,hashlib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'03_方法冻结与隔离评价';B=T/'kitti360_isolated_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,s):p.write_text(s,encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(B/'local_integrity_and_independent_recount.json')['status']=='PASS_LOCAL_FULL_HASH_AND_INDEPENDENT_GT_AND_RISK_RECOUNT'
assert read(T/'kitti360_taxonomy_audit_v1/taxonomy_validation.json')['status']=='PASS_OFFICIAL_TAXONOMY_AND_RAW_GT_MAPPING'
summary=read(B/'GT_final_evaluation/equal_drive_descriptive_summary.json');check=read(B/'GT_final_evaluation/fixed_comparison_checks.json');receipt=read(B/'package_result.json')
report='kitti360_isolated_v1/KITTI360隔离评价完成报告_20261007.md'
write(T/'README.md',f'''# 03_方法冻结与隔离评价

**新增KITTI-360两窗口隔离评价已执行，完整证据下载与独立GT/风险曲线复算通过。** 最新报告：[{Path(report).name}]({report})。

每窗口21帧仅用于建图/预测/融合，5个评价视角RGB/深度完全排除；52个外部pose条件下的两个小窗口，是本项目公开train/val数据隔离评价，不是官方隐藏Test。预声明和源码封存后运行五个固定方法，无最终Oracle和调参。

官方类别映射与十份原始语义/实例PNG已独立核验：见`kitti360_taxonomy_audit_v1/类别映射审计与预声明文字澄清.md`。预声明ignore文字已澄清，实现、掩码和指标不变。

仍保留S0作为此前DEV选择的参考。旧KITTI08工程封存、VKITTI2 GT-DEV封存以及负结果完整保留；新增最终窗口不能改变DEV选择或Stage2正式FAIL。

尚不能给出：原KITTI08稠密GT评价、严格稳定/动态遮挡分层、广泛驾驶场景泛化以及整体算法科学PASS。小子组像素/实例不足按NOT_EVALUABLE/INCONCLUSIVE报告，不换帧凑结果。
''')
write(B/'README.md',f'''# KITTI-360 项目隔离评价 v1

报告：[KITTI360隔离评价完成报告_20261007.md](KITTI360隔离评价完成报告_20261007.md)。两个新窗口、十个评价视角×五组原生渲染、共同GT计分/子组及风险曲线完成。

完整归档 `{receipt['files']}` 份文件，SHA256 `{receipt['sha256']}`。本地全文件哈希、50份渲染的独立GT布尔计数、主/次mIoU和完整风险曲线复算通过；服务器CPU逐帧重融合十组状态逐值相同。

逐类GT、无地图支持和unknown在主指标保留。缺少严格动态GT对应，稳定/动态区分层NOT_EVALUABLE。DEV选择仍S0，不在此重新选择方法。不覆盖本目录，不将小公开数据隔离窗口称官方隐藏Test。
''')
progress=R/'执行进度与下一步.md';text=progress.read_text(encoding='utf-8');first='**原KITTI08工程完成；02新增VKITTI2独立GT DEV评价已完成，03新增DEV判据/候选判定与规则结果封存。隔离最终评价尚未执行。** 最新报告见`02_基础因子DEV与Oracle诊断/gt_dev_v2/VKITTI2_GT_DEV补齐执行报告_20261007.md`。'
new='**02 GT-DEV与03新增KITTI-360项目隔离评价已完成，完整证据已同步并独立复算。DEV仍保留S0，不宣称加权方法通过科学验收。** 最新报告见`03_方法冻结与隔离评价/kitti360_isolated_v1/KITTI360隔离评价完成报告_20261007.md`。'
assert first in text;text=text.replace(first,new,1)
text+='''
## KITTI-360隔离评价补齐（最新，2026-10-07）

按未查看语义像素的元数据规则确定drive0000 250–275与drive0002 4391–4416。每窗口21帧建图/预测/融合、5帧只取相机pose和官方GT；评价视角RGB与深度未下载或读取。首次GT像素读取前冻结协议，GPU前封存源码。官方pose是更广泛轨迹的优化估计，实验不证明tracking准确率或全部数据先验隔离。

两份固定基图、五组×两个窗口、50份原生语义渲染完成。mIoU（预冻结共通类与City19 GT存在类）、覆盖、unknown/no-map、同分风险—覆盖、小车辆/非小车辆/边界/内部子组与行人描述项完成。服务器CPU重融合状态一致，本地完整哈希、主/次mIoU、整数GT计数和完整风险曲线复算通过。

这是项目内新的公开train/val小窗口隔离评价，不是KITTI-360官方隐藏Test。最终窗口用于测量固定方法，禁止回头调阈值、重新选择赢家、改变旧DEV或正式FAIL。严格稳定/动态遮挡区分层仍缺独立GT对应，小子组不足明确不可验；原KITTI08语义GT仍缺，不能由新窗口替代。

下一步可整理04展示及05/06结果归档；若要提出新方法，需要新DEV证据和新的隔离评价入口，不能用本次最终窗口作开发后再次声称隔离。无需立即重开Stage2、训练MLP或运行未触发的S2-Z/S2-V。
'''
write(progress,text)
seal={'status':'ISOLATED_PROJECT_HOLDOUT_MEASUREMENT_SEALED','DEV_reference':'S0','postholdout_selection':False,'final_quality_winner':None,'global_scientific_PASS':False,'report':report,'protocol_sha256':sha(R/'protocol/KITTI360隔离评价预声明_20261007_v1.json'),'artifact_archive_sha256':receipt['sha256'],'local_hash_and_GT_risk_recount':True,'old_gt_dev_freeze_preserved':True,'KITTI08_GT':'MISSING','stable_dynamic':'NOT_EVALUABLE','official_hidden_benchmark_Test':False,'summary':summary,'fixed_comparison_checks':check};write(B/'scientific_scope_sealed.json',json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
files=[progress,T/'README.md',B/'README.md',B/'local_integrity_and_independent_recount.json',B/'scientific_scope_sealed.json',B/'cpu_contracts.json',B/'package_result.json',Path(__file__),R/'scripts/verify_kitti360_local.py']
files += [R/n for n in ['kitti360_download_script_audit.json','kitti360_official_script_content.json','kitti360_metadata_audit_receipt.json','kitti360_adapter_launch.json','kitti360_pipeline_launch.json','kitti360_results_snapshot.json','kitti360_package_completion.json'] if (R/n).exists()]
files += [p for p in (T/'kitti360_taxonomy_audit_v1').rglob('*') if p.is_file()]
files += [R/'scripts/audit_kitti360_taxonomy.py']
files += [T/n for n in ['kitti360_taxonomy_audit.log','kitti360_taxonomy_retry1.log']]
files += [R/'kitti360_taxonomy_package_receipt.json']
manifest={str(p.relative_to(R)):sha(p) for p in files};mp=B/'local_completion_SHA256.json';write(mp,json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');archive=B/'local_completion_sync_20261007.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files+[mp]:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive)},ensure_ascii=False))
