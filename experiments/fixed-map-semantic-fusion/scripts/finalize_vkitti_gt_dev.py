"""Verify and archive the bounded VKITTI2 GT-DEV extension without old edits."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
root=Path('/root/autodl-tmp/stage3_chy_20261006');step=root/'02_基础因子DEV与Oracle诊断/gt_dev_v2';data=root/'inputs/vkitti2_gt_dev_v2';ev=step/'GT_evaluation_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(step/'pipeline_status.json')['status']=='COMPLETE_GT_DEV_NOT_ISOLATED_FINAL_TEST'
assert read(ev/'result.json')['status']=='COMPLETE_INDEPENDENT_GT_EXPOSED_DEV_EVALUATION'
manifest=read(data/'manifest.json');checks=[]
for seq in manifest['sequences']:
 d=Path(seq['path']);poses=np.load(d/'c2w.npy');expected=np.asarray([np.linalg.inv(p)@poses[0] for p in poses])
 with np.load(d/'base_map.npz',allow_pickle=False) as p:
  assert np.allclose(p['gt_w2c_all_frames'],expected,atol=1e-4)
  assert np.array_equal(p['source_gaussian_index'],np.arange(len(p['means3D'])))
  n=len(p['means3D'])
 assert sha(d/'base_map.npz')==read(d/'base_map_lock.json')['file_sha256']
 for record in seq['records']:
  for entry in record['files'].values():assert sha(Path(entry['path']))==entry['sha256']
 for mode in ['smoke','full']:
  run=step/(seq['name']+'_'+mode);assert read(run/'result.json')['state_identity_roundtrip']
  assert read(run/'result.json')['geometry_and_RGB_and_camera_unchanged']
 checks.append({'sequence':seq['name'],'gaussians':n,'pose_direction_matches_official_relative':True,'input_hashes_match':True,'smoke_and_full_pass':True})
dump(step/'post_run_validation.json',{'status':'PASS','checks':checks,'old_KITTI08_outputs_untouched':True})
rows=read(ev/'GT_main_results.json');oracle=read(ev/'O_Sem_diagnostic.json');pooled=read(ev/'GT_pooled_descriptive.json')
decision=read(ev/'DEV_candidate_decision.json');policy_path=root/'protocol/VKITTI2_GT_DEV判优预声明_20261007_v2.json';policy=read(policy_path)
table='\n'.join(f"| {r['scene']} | {r['group']} | {r['mIoU_GT_supported_common_classes']*100:.2f}% | {r['coverage']*100:.2f}% | {r['conditional_accuracy']*100:.2f}% | {r['unknown_fraction']*100:.2f}% | {r['no_map_fraction']*100:.2f}% |" for r in rows)
otable='\n'.join(f"| {r['sequence']} | {r['S0_mIoU']*100:.2f}% | {r['O_Sem_mIoU']*100:.2f}% | {r['difference']*100:+.2f}百分点 |" for r in oracle)
report=f'''# VKITTI2 GT-DEV补齐执行报告

日期：2026-10-07，CHY。**02已新增独立官方语义GT的DEV评价：两场景、五个可部署方法及O-Sem诊断，全部真实运行完成。03新增GT评价规则与结果封存；隔离最终评价未执行。** 本轮不会把历史Stage2 FAIL改成PASS，也不替代原KITTI08缺GT的事实。

## 实际执行

官方类别包MD5 `e8658ab49250e61f1caa174625563263`匹配，SHA256见manifest。共享RGB/depth/instance/textgt四包只读，新包及派生输出均归属阶段3。只解出Scene01/clone 330–355、Scene02/clone 110–135，每窗口26帧，CPU适配52帧。数据包来源：https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/ 。

每窗口使用固定官方pose和**预测SGBM深度**建一张共同几何图，10次mapping迭代/帧，无语义驱动几何优化。官方深度只用于相机尺度/GT重投影与评价子组，不作主实验建图或融合深度。几何见全部26帧；每组21帧语义融合、5帧评价，共60份全量渲染。每窗口另有3帧融合/1视角的小测试。整个实验为离线固定地图与GT-pose控制实验，不证明tracking改善。

五组预测输入不使用GT有效掩码。O-Sem独立状态只在允许融合的同21帧用GT硬标签单位票，评价视角不写入状态。预测S0与Oracle使用相同帧集合、几何、pose、基础可见检查；GT ignore不提供Oracle观测。它是DEV输入瓶颈诊断，不是可部署方法或严格上限。

保存/恢复稳定ID、重复源帧no-op、原生可见贡献质量守恒、共享支持、固定参数逐值不变均通过。后置审计也验证了保存的相机pose与官方relative w2c一致，以及所有派生输入哈希。运行命令、返回码和错误日志保留在各receipt与log中。

## GT指标与覆盖

GT按官方RGB colors解码，Tree/Vegetation合并vegetation，Van/Car合并car；GuardRail、Misc、Undefined为冻结ignore，不能冒充与Cityscapes严格同构。评价共同类为road/building/pole/light/sign/vegetation/terrain/sky/car/truck；各场景GT无支持类NOT_EVALUABLE。无行人GT，不宣称行人表现。

主指标名为**地图渲染的二维语义mIoU**；unknown和无地图支持在有效GT上都计真实类漏判，同时另报覆盖。条件准确率只在有标签输出区计算，不能单独代表完整质量。

| 场景 | 方法 | mIoU | GT有效区覆盖 | 条件准确率 | unknown | 无地图支持 |
|---|---|---:|---:|---:|---:|---:|
{table}

逐类混淆、支持数见`GT_evaluation_v1/GT_main_results.json`；完整风险—覆盖曲线保存npz，按最终最大类别质量/独立原生alpha排序，同分全部一起接受，GT只计算错误，不参与排序。共同覆盖表见`GT_common_risk_coverage.json`，no-map与unknown不靠随意类别补齐。

小/非小车辆、边界/内部、官方moving/nonmoving车辆均用所有组共用的GT规则。小车辆面积≤图像0.25%，边界半径3像素。额外以相邻帧GT深度/类别重投影定义temporal_stable与temporal_change代理，无法归类区域另报且保留在全图主指标中；不把代理当完美动态/遮挡真值。详见子组表和共同掩码哈希。

## O-Sem结果

| 窗口 | 预测S0 mIoU | GT输入O-Sem mIoU | 差值 |
|---|---:|---:|---:|
{otable}

若GT输入改善，支持输入标签质量是限制之一；若仍然较差，需检查地图支持、遮挡、关联、动态时序与类别映射。不能据一次差结果认定唯一原因，也不直接启动MLP。

## 科学范围与03状态

两个窗口是已曝光DEV，全部相关视角不当独立重复。不计算像素级独立置信区间；按场景分别报告，两场景合并只是描述。本轮完成独立GT评价链与结果封存，不签发广泛泛化结论或隔离Test PASS。Scene20及旧正式Test未读。

03新增冻结目录`03_方法冻结与隔离评价/gt_dev_rules_freeze_20261007_v2`记录指标、排序/并列、分母、子组、方法/输入及结果哈希。**本轮DEV判据在首次GT指标读取前声明**：等权两场景平均mIoU至少提升1个百分点，单场景mIoU和覆盖损失各最多2个百分点，车辆/小目标/边界等最多退化2个百分点，并设最低GT支持及批量资源约束。这是初始实用项目容差，不是已校准显著性标准或最终Test PASS。

DEV判定：`{decision['status']}`；下一轮DEV参考：**{decision['selected_DEV_reference']}**。具体未满足门槛和支持缺项见`GT_evaluation_v1/DEV_candidate_decision.json`；所有五组与负结果完整保留，Oracle不参与排名。若保留S0仅表示候选没有满足预声明条件，不代表新的质量改善。最终科学接受阈值与隔离数据协议仍需另审，不能以DEV判定代替。

已实现命令与完整执行回执见`run_vkitti_gt_pipeline.py`及各阶段receipt。重跑必须另建版本目录，不能原样再次启动覆盖本轮。服务器专属根：`/root/autodl-tmp/stage3_chy_20261006`。
'''
(step/'VKITTI2_GT_DEV补齐执行报告_20261007.md').write_text(report,encoding='utf-8')
freeze=root/'03_方法冻结与隔离评价/gt_dev_rules_freeze_20261007_v2';freeze.mkdir()
dump(freeze/'scientific_scope.json',{'status':'GT_DEV_RULES_AND_RESULTS_SEALED_FINAL_TEST_PENDING','evaluation_protocol_file':str(ev/'GT_evaluation_protocol.json'),'evaluation_protocol_sha256':sha(ev/'GT_evaluation_protocol.json'),'all_five_retained':True,'DEV_selected_reference':decision['selected_DEV_reference'],'DEV_policy_sha256':sha(policy_path),'DEV_minimum_quality_gain':policy['minimum_equal_scene_mean_mIoU_gain'],'DEV_coverage_loss_budget':policy['maximum_single_scene_coverage_loss'],'final_quality_winner':None,'final_acceptance_protocol':'PENDING','final_split':[],'reason':'historically exposed DEV only; no final-Test PASS','main_claim':'official independent semantic GT DEV evaluation, not final generalization acceptance'})
files=[p for p in step.rglob('*') if p.is_file()]+[p for p in data.rglob('*') if p.is_file() and p!=data/'vkitti_2.0.3_classSegmentation.tar' and p.name!='download.log']
files += [policy_path]+[root/'scripts'/n for n in ['prepare_vkitti_gt_dev.py','run_vkitti_gt_fusion.py','run_vkitti_gt_pipeline.py','evaluate_vkitti_gt_dev.py','select_vkitti_gt_dev_candidate.py','test_gt_evaluator_contracts.py','finalize_vkitti_gt_dev.py','vkitti_mapping_template.py','semantic_cpu.py','native_semantic_render.py']]
dump(freeze/'artifact_SHA256.json',{p.relative_to(root).as_posix():sha(p) for p in files})
dump(freeze/'freeze_record.json',{'owner':'CHY','date':'2026-10-07','artifact_manifest_sha256':sha(freeze/'artifact_SHA256.json'),'archive_exclusions':'official shared archives referenced only; new official class archive checksum locked in manifest; no archive copy locally required'})
files += list(freeze.iterdir())
archive=root/'02_基础因子DEV与Oracle诊断/vkitti_gt_dev_v2_complete_20261007.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p.relative_to(root).as_posix())
result={'status':'PACKAGED_GT_DEV','files':len(files),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}
dump(step/'package_receipt.json',result);print(json.dumps(result),flush=True)
