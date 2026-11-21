"""Produce durable evidence-bound CPU-phase reports; never imports CUDA."""
import csv,hashlib,json,platform,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FYP=ROOT.parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8')
def dump(rel,obj):write(rel,json.dumps(obj,ensure_ascii=False,indent=2))

def main():
    snapshot=ROOT/'01_语义表示_初始化与渲染审计/source_snapshots';snapshot.mkdir(exist_ok=True)
    sources=[FYP/'9.16补漏/S3优化后/deployment/fuse_semantic_gaussians_kitti08.py',
      FYP/'2026.9.1/第10步_Mask2Former_200帧/execution_bundle/scripts/generate_mask2former_semantics_kitti.py',
      FYP/'9.16补漏/S3_KITTI08三维Gaussian地图展示/remote_work/prepare_seq08_subset.py',
      FYP/'阶段2/08_M3验收与复现归档/archive_20261001_v1/source/SplaTAM_t25_tsdd_loader/utils/recon_helpers.py']
    source_manifest=[]
    for source in sources:
        target=snapshot/source.name;shutil.copy2(source,target)
        source_manifest.append({'source':str(source),'snapshot':str(target.relative_to(ROOT)),'sha256':sha(target),
              'attribution':'historical shared project source; not newly authored in Stage 3'})
    dump('01_语义表示_初始化与渲染审计/source_manifest.json',source_manifest)
    audit=json.loads((ROOT/'00_输入与基图_GT曝光审计/remote_inputs.json').read_text(encoding='utf-8'))
    idmap=audit['metadata']['/root/autodl-tmp/s3_seq08_work/subset/semantic/id2label.json']
    dump('protocol/class_mapping.json',{'prediction_ids':idmap,'gt_mapping':None,'gt_mapping_status':'PENDING_INDEPENDENT_GT','unknown':-1,'no_map_support':-2,'gt_ignore':255})
    manifest=json.loads((ROOT/'inputs/base_map_manifest.json').read_text(encoding='utf-8'))
    dump('inputs/exposure_manifest.json',{'dataset':'KITTI08','original_window':[3000,3199],
        'exposure':'EXPOSED_DEV_AND_DISPLAY','all_frames_used_in_geometry_and_appearance':True,
        'prior_all_frames_semantic_fusion':True,'test_status':'NOT_NEW_TEST',
        'historical_stage1_test_already_consumed':True,'stage3_reset_does_not_remove_exposure':True})
    dump('protocol/split_manifest.json',{'status':'ENGINEERING_DEV_ONLY','frame_conversion':'original=local+3000',
        'fusion_local':[i for i in range(200) if i%10!=9],'dev_render_local':[i for i in range(200) if i%10==9],
        'exclusion_scope':'semantic fusion observations only; geometry used all 200','formal_test':[]})
    dump('protocol/observation_schema.json',{'source_frame_identity':['dataset','sequence','camera','original_frame'],
        'input_version':'per-input SHA256 lock; version change requires another run',
        'gaussian_identity':'base source_gaussian_index','support_unit':'distinct effective source frame per Gaussian',
        'replay':'same frame/version no-op; changed version raises error','checkpoint':'evidence and processed ledger committed together using atomic file replace',
        'frame_aggregation_candidate':'alpha-weighted within-frame class distribution; unit effective support budget',
        'association_cache':'PENDING_NATIVE_VISIBLE_CONTRIBUTION_GENERATION_OR_VALIDATED_CACHE',
        'cached_2d_label':'hard Cityscapes-19 selected class',
        'cached_2d_confidence':'selected semantic class evidence / sum of 19 class evidence; scalar, not full vector or calibrated probability',
        'provenance_scope':'local audited producer definition; exact KITTI08 inference receipt still to be bound',
        'full_class_distribution_available':False,'history_consistency':'disabled'})
    rows=[]
    for entry in audit['files']:rows.append({'type':'historical_remote_file',**entry})
    for group in audit['semantic_arrays']:
        for entry in group['entries']:rows.append({'type':group['kind'],'path':entry['file'],'exists':True,'sha256':entry['sha256'],'shape':str(entry['shape']),'dtype':entry['dtype'],'finite':entry['finite']})
    rows.append({'type':'frozen_stage3_base','path':manifest['base_local'],'exists':True,'sha256':manifest['file_sha256']})
    keys=sorted({k for row in rows for k in row})
    for rel in ['inputs/input_inventory.csv','00_输入与基图_GT曝光审计/input_inventory.csv']:
        with open(ROOT/rel,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    dump('00_输入与基图_GT曝光审计/semantic_array_validation.json',{'label_count':200,'confidence_count':200,
       'all_shapes_370x1226':all(e['shape']==[370,1226] for g in audit['semantic_arrays'] for e in g['entries']),
       'all_finite':all(e['finite'] for g in audit['semantic_arrays'] for e in g['entries']),
       'label_ranges_valid':all(0<=e['min']<=e['max']<=18 for e in audit['semantic_arrays'][0]['entries']),
       'confidence_ranges_valid':all(0<=e['min']<=e['max']<=1 for e in audit['semantic_arrays'][1]['entries'])})
    write('00_输入与基图_GT曝光审计/数据与GT审计.md','''# 输入、基图、GT与曝光审计（CPU，2026-10-06）

服务器专属目录为 `/root/autodl-tmp/stage3_chy_20261006`；本轮只执行目录创建、读取、传输及CPU脚本，未启动GPU推理、建图或渲染。

唯一基图为本地核验后的 KITTI08 P1_all200_recovery800_resume_v1，907,488个Gaussian，SHA256 `3244884f5a22ebfc6cf94b4ec0e48ddb4f666f5803938bc5e53cc4c62ae6031b`。本轮复制到 `inputs/base_map/params.npz`；原服务器历史文件已不存在，恢复只写阶段3目录，不覆盖旧目录。

几何、RGB/opacity及历史语义均使用过3000–3199；位姿为KITTI GT，不是ORB估计。本轮180帧重新融合、20帧仅语义观测排除的DEV渲染划分，仍是已曝光窗口，不是新正式Test。字段哈希和稳定ID见base_map_manifest.json。

服务器200张二维硬语义标签及200张confidence全部CPU读取、hash并核验。实际RGB目录为 `s3_seq08_work/subset/sequences/08_s3_3000_3199/image_2`，200个链接有效；初始脚本误用`sequences/08`得到0张，后续路径核查已纠正，保留原审计记录。RAFT左目深度数组当前不在旧raft_left目录，仅保留预览/统计；不能假设深度权重输入已就绪。

所审计KITTI目录未发现对应独立二维语义GT；现有static_valid_masks来自预测类别排除，不是独立GT，不用于缩小主分母。VKITTI2已有归档是RGB、depth、instanceSegmentation和textgt；Scene02/clone的frames也只有上述三种图像，没有classSegmentation。colors.txt有类别名称不等于已拥有逐像素完整语义标签。实例/车辆信息只能用于另立有限类别诊断，不能跨数据集拿来评价KITTI地图。未读取旧Scene20预测或重新消费其Test。

结论：基图和二维预测工程输入可用；独立语义GT、对应完整映射、原生关联缓存及实际深度权重输入未就绪。可完成初始化和CPU契约核验；S0/S1/S-G/S2/S2-H真实语义收益、O-Sem及正式冻结评价暂不能执行。GT短期不可用时按计划交付工程准备和限制，不强行扩建数据集。
''')
    write('01_语义表示_初始化与渲染审计/代码审计与核验报告.md','''# 语义表示、初始化与渲染审计（CPU）

## 实际发现

1. 历史融合在入口创建零evidence和observations，单次顺序200帧内每Gaussian中心最多更新一次；关联为中心采样而非整个足迹多像素融合。硬conf阈值、排sky、中心/渲染深度容差、opacity和confidence权重同时存在，不能直接叫作无可靠性加权的S0。
2. 历史semantic_rendervar把opacity乘semantic_valid，使unknown高斯不遮挡；本轮不得用于定量语义评价。
3. 历史语义图是palette颜色alpha混合的可视化，不能从颜色反查类别计算mIoU。
4. 历史checkpoint仅保存最终label、最大类别份额、evidence_sum和observation_count，没有完整类别证据或去重账本，无法无损继续本轮融合。其source_gaussian_index是compact数组位置；基图source_gaussian_index是完整图源ID。旧标签不能按相同字段名直接假定ID等价。
5. 历史代码未提供重复源帧/断点恢复的幂等账本；顺序单次运行计数合理，不代表可安全重复执行。
6. 本地生产脚本定义confidence为选中类别的semantic evidence占19类总和；不是实例置信度，也不是已保存的完整类别分布。该定义与缓存范围一致，但KITTI08具体推理receipt尚需绑定。

## 已执行

真实907,488点基图哈希核验、逐字段hash和五组独立参数副本逐值比较已完成。五组零语义sidecar保存/恢复通过稳定Gaussian ID对应、零证据/支持/权重及空去重账本核验。旧语义没有导入；共享冻结基图文件，避免保存五份重复参数。

semantic_cpu.py提供独立证据账本、帧内归一化、五组因子权重、unknown/无地图支持计分和整块并列风险—覆盖函数。10项受控CPU测试通过，包括重复帧与断点、身份错配拒绝、前景unknown遮挡、面积归一化、错误早期标签可被后续纠正。

native_semantic_render.py已准备原生渲染器的数值类别通道适配：保留所有opacity，unknown单独通道，分批三通道渲染，零背景，输出class_mass/unknown_mass/alpha/label/score。未执行GPU路径；CPU测试通过只证明合成参考和适配参数契约，不能证明原生CUDA实现遮挡测试或RGB数值不变性已经通过。

## 待原生渲染核验

实际Gaussian小场景unknown前景测试、各组同相机RGB在冻结容差内一致、原生多通道pass alpha一致性及真实帧可见贡献缓存。RGB容差尚未实测，不虚填数字。
''')
    write('protocol/rendering_and_scoring_rules.md','''# 工程规则草案 v0.1：非科学方法冻结

- 唯一基图及180/20已曝光DEV角色由清单绑定；所有语义证据从零开始。
- 固定最终地图包含全部基图Gaussian；不按历史timestep删点，不能称逐时刻动态快照或增量重建。
- 原生排序、足迹、alpha/透射率共享；unknown保持opacity。数值类别和unknown各有通道，palette仅展示。
- CPU参考阈值：alpha_min=0.05，unknown_fraction_max=0.5。它们是待DEV核验的工程初值，非已验收参数。支持足够且unknown占比低于阈值才可输出类别。
- 主计分E为映射后非ignore的有效GT；unknown(-1)与no-map(-2)都构成真实类别FN；只平均有GT支持的共同类，预测到无GT类仍计入真实类FN并附FP表。GT类表未知，实际mIoU未运行。
- 分数为最终最大类别质量/总alpha质量，unknown进入分母；不是校准概率。风险排序整块接受同分，零覆盖风险不可评价，无地图支持不可补类。共同覆盖仅取各方法可达点，不插补高覆盖。
- 长期稳定表面作为优先目标；动态/遮挡变化、小目标、边界/内部需要独立共同分组。当前无GT，分组像素、尺寸阈值和boundary带宽未科学冻结，不用预测Static掩码替代。
- S0/S1/S-G/S2共享基础有效性，历史一致性关闭。硬拒绝默认0.5只用于CPU契约示例，最终阈值需DEV选择。
- S2-Z/S2-V尚无对应真实DEV触发证据，本轮不自动加入。
''')
    write('protocol/实验协议_v1.md','''# 阶段3工程入口协议 v0.1

状态：CPU_PREPARATION_COMPLETE_WITH_DATA_AND_GPU_GATES；不是科学方法冻结或正式PASS。

输入和工程规则见base_map_manifest、split_manifest、observation_schema与rendering_and_scoring_rules。五组初始化完成，历史算法仅作source snapshot和历史对照，不冒充S0。

下一关先建立独立二维语义GT及共同子组/类映射，确认观测关联缓存和几何可靠性输入；再运行原生渲染小测试及RGB容差核验，之后才做真实DEV方法比较。准确率改善量、覆盖退化预算、关键子组约束在DEV支持审计后冻结，当前均PENDING；无GT不能填写虚假判优门槛。

O-Sem缺GT跳过。原KITTI08窗口已曝光，只供DEV/展示；独立正式评价需要另审数据来源及曝光，不复用旧Scene20 one-shot。
''')
    dump('protocol/工程执行清单.json',{'phase':'CPU_ONLY','groups':['S0','S1','S-G','S2','S2-H'],
        'map_lock':manifest['file_sha256'],'init_status':'PASS_CPU','cpu_contracts':'10/10 PASS',
        'native_renderer':'PREPARED_NOT_EXECUTED','rgb_invariance':'PENDING_GPU','real_dev_comparison':'PENDING_INPUT_AND_NATIVE_CACHE',
        'independent_gt':'NOT_FOUND_IN_AUDITED_INPUTS','oracle':'SKIPPED_NO_GT','formal_freeze':'NOT_ISSUED',
        'learning':'NOT_STARTED','environment':{'python':platform.python_version(),'numpy':np.__version__}})
    steps=[('00_输入与基图_GT曝光审计','CPU审计完成；GT和深度/关联入口仍不足'),
        ('01_语义表示_初始化与渲染审计','CPU初始化/契约通过；原生GPU核验待执行'),
        ('02_基础因子DEV与Oracle诊断','CPU工具准备完成；真实五组DEV未跑，O-Sem缺GT跳过'),
        ('03_方法冻结与隔离评价','工程草案就绪；科学冻结/隔离评价尚未签发'),
        ('04_系统整合与最终归档','本轮CPU成果归档；完整系统最终交付未完成'),
        ('05_条件轻量候选与历史诊断','历史纠错参考测试通过；S2-Z/S2-V无真实触发证据暂不启动'),
        ('06_类别运动与学习扩展','依计划后置，本轮未启动')]
    for name,status in steps:
        write(name+'/README.md',f'# {name}\n\n状态：{status}。\n\n本目录对应计划步骤；已有CPU产物与未执行的GPU/数据依赖分开记录。总进度见根目录执行进度与下一步.md。')
    write('02_基础因子DEV与Oracle诊断/待执行入口.md','''# 真实DEV入口

不能将CPU合成样例或旧palette图作为真实DEV成绩。已具备五组权重、证据账本和CPU评价器；尚需真实可见关联缓存、几何可靠性输入、独立语义GT。只有预测教师一致性时另表诊断，不称准确率。

O-Sem：本轮无对应GT，跳过。既不将KITTI预测换成GT，也不将另一数据集GT贴到KITTI地图。
''')
    write('05_条件轻量候选与历史诊断/候选启动记录.md','''# 候选与历史诊断

CPU受控测试证明关闭历史反馈的累积投票可被后续不同源帧纠正早期错误；这是参考契约，不是实际DEV压力评价。真实扰动帧/位置、正常标签代价仍待数据入口。

S2-Z：缺实际深度/DEV误差分析，尚未满足远近阈值问题触发条件。
S2-V：缺真实多视角误差审计，尚未满足相关错误过度计数触发条件。
不同时叠加二者，不启动MLP。
''')
    write('执行进度与下一步.md','''# 阶段3执行进度（2026-10-06，CPU阶段）

本地逐步目录00–06已建立；服务器独立目录 `/root/autodl-tmp/stage3_chy_20261006` 已建立。本轮不启动GPU任务。

已完成：远端目录/数据审计，200标签+200置信数组读取与SHA256，真实907,488点唯一基图冻结，五组语义重置与保存恢复身份核验，源码归档与问题定位，CPU融合/计分契约实现及10项测试，原生数值语义通道适配准备，输入/曝光/观测和评价工程草案。

重要发现：旧unknown opacity被清零，palette混色不适合类别评价；旧服务器基图已清理，已从本地核验备份恢复到阶段3专属路径。旧数组仅有硬标签和选中类别置信度，未保存完整类别分布。KITTI08基图使用GT位姿且200帧已曝光，不能证明跟踪改善或新Test。

当前未完成：独立语义GT和共同子组、可见关联缓存、实际深度输入、GPU原生遮挡及RGB容差核验、真实五组DEV/Oracle和科学冻结评价。CPU参考通过不签发原生渲染或语义质量PASS。

下一步顺序：先确定独立GT可行入口（短期无GT则按计划降级交付工程和限制）→补齐输入/原生缓存并执行GPU小核验→五组真实DEV→冻结与隔离评价。S2-Z/S2-V及学习扩展仅在条件成立时启动。

详情见00审计、01代码审计、protocol工程规则以及每步README。服务器同步核验另见04目录的同步记录。
''')
    dump('04_系统整合与最终归档/cpu_phase_status.json',{'cpu_work':'COMPLETE_FOR_AVAILABLE_INPUTS','scientific_stage3_complete':False,
       'gpu_jobs_started':0,'base_gaussians':907488,'initialization_groups':5,'cpu_tests_passed':10,
       'independent_gt':'NOT_FOUND_IN_AUDITED_INPUTS','native_gpu_tests':'NOT_RUN','real_dev_metrics':'NOT_RUN'})
    print(json.dumps({'report_status':'COMPLETE','inventory_rows':len(rows),'steps':len(steps)}))

if __name__=='__main__':main()
