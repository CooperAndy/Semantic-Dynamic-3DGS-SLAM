"""Consolidate observed CPU audit evidence; no inference or GPU imports."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np

root = Path(__file__).resolve().parents[1]
step = root / '00_输入与基图_GT曝光审计'
e = step / 'reaudit_20261007_v1'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
rows = list(csv.DictReader((e / 'input_inventory.csv').open(encoding='utf-8')))
old = read(root / 'inputs/base_map_manifest.json')
base = read(e / 'base_validation.json')
rgb = read(e / 'input_rgb_sha256.json')['rgb_sha256']
matches = sum(r['sha256'] == rgb.get(f"{int(r['original_frame']):06d}.png") for r in rows if r['kind'] == 'image_2')
field_match = base['fields'] == old['fields']
model = read(e / 'model_provenance.json')
checkpoint_match = model['checkpoint_live']['sha256'] == model['generation']['checkpoint']['sha256']
assert len(rows) == 1000 and matches == 200 and field_match and checkpoint_match
assert base['stable_ids_unique'] and base['sha256'] == old['file_sha256']
cal = read(e / 'calibration_pose_validation.json')
p2 = np.array(cal['calib']['P2']).reshape(3, 4)
p3 = np.array(cal['calib']['P3']).reshape(3, 4)
c2 = -np.linalg.solve(p2[:, :3], p2[:, 3])
c3 = -np.linalg.solve(p3[:, :3], p3[:, 3])
depth = [r for r in rows if r['kind'] == 'depth_sgbm_candidate']
checks = {
    'date': '2026-10-07', 'step00_status': 'AUDIT_COMPLETE_WITH_LIMITS',
    'next_gate': 'READY_FOR_STEP01_ENGINEERING_ONLY', 'science_ready': False,
    'files_checked': len(rows), 'historical_input_rgb_hash_matches': matches,
    'all_base_fields_match': field_match, 'base_field_count': len(base['fields']),
    'stable_gaussian_ids_unique': base['stable_ids_unique'],
    'checkpoint_hash_matches_generation_record': checkpoint_match,
    'legacy_parser_baseline_m': cal['stereo_baseline_m'],
    'full_projection_camera_centers': [c2.tolist(), c3.tolist()],
    'full_projection_center_distance_m': float(np.linalg.norm(c3-c2)),
    'calibration_limit': 'Legacy depth parser ignores nonzero projection tz; current source snapshot is not a frozen generation config. Do not alter cached depths or base cameras during audit.',
    'depth_positive_fraction_min': min(float(r['positive_fraction']) for r in depth),
    'depth_positive_fraction_max': max(float(r['positive_fraction']) for r in depth),
    'independent_GT': 'NOT_FOUND_IN_AUDITED_INPUTS',
    'historical_RAFT': 'OLD_PATH_REMOVED', 'SGBM': 'CANDIDATE_ONLY',
    'gpu_compute_started': False,
}
write(e / 'completion_checks.json', checks)
manifest = {
    'version': 'kitti08_source_locked_20261007_v1',
    'owner': 'CHY', 'endpoint': 'SERVER_HOST:16680',
    'stage3_remote_root': '/root/autodl-tmp/stage3_chy_20261006',
    'reference_root': '/root/autodl-tmp/stage3_chy_20261006/inputs/kitti08_source_locked_20261007_v1',
    'reference_type': 'symlinks to read-only shared originals; not byte backups; recheck file hashes before use',
    'sequence': 'KITTI08', 'original_frames': list(range(3000, 3200)),
    'base_map_remote': old['base_remote'], 'base_sha256': old['file_sha256'],
    'base_gaussians': base['num_gaussians'],
    'geometry_and_historical_semantic_exposure': list(range(3000, 3200)),
    'allowed_semantic_fusion_original_frames': [3000+i for i in range(200) if i % 10 != 9],
    'excluded_from_current_semantic_fusion_original_frames': [3000+i for i in range(200) if i % 10 == 9],
    'evaluation_role': 'HISTORICALLY_EXPOSED_DEV; semantic-excluded views are not unseen geometry or formal Test',
    'confidence_definition': model['generation']['confidence_definition'],
    'confidence_limit': 'scalar selected-class evidence ratio; not full class distribution or calibrated probability',
    'labels': '19 Cityscapes classes, IDs 0..18; frozen id2label.json; reject invalid IDs',
    'GT_status': checks['independent_GT'], 'O_Sem': 'SKIP_UNTIL_INDEPENDENT_GT_AVAILABLE',
    'depth_candidate': 'SGBM float32 meters, zero invalid; not adopted as historical RAFT replacement',
    'pose_convention': 'stored relative w2c: inv(c2w_i) @ c2w_first; uses KITTI odometry GT source, not ORB estimated poses',
    'semantic_reset': 'Each method starts from immutable base geometry and new zero evidence/support/weight/conflict sidecar, keyed by source_gaussian_index; historic exposure remains.',
    'audit_evidence': '00_输入与基图_GT曝光审计/reaudit_20261007_v1',
    'readiness': checks['next_gate'],
}
write(root / 'inputs/current_input_manifest_20261007.json', manifest)
# Preserve the earlier dated manifest and inventory rather than rewriting their evidence.
shutil.copy2(e / 'input_inventory.csv', root / 'inputs/input_inventory_20261007.csv')
report = f'''# 步骤00完成记录：输入、唯一基图与GT曝光审计

日期：2026-10-07；负责人：CHY。结论：**00审计完成，有明确限制；可进入01工程核验，尚不具备独立语义准确率评价条件。** 本轮仅CPU读取、哈希、元数据/代码核对与目录恢复，未启动GPU推理、训练、融合或渲染实验。

## 1. 本次验收

| 项目 | 核验结果 | 判定 |
| --- | --- | --- |
| 服务器独立工作区 | `/root/autodl-tmp/stage3_chy_20261006`，本次端口16680，旧工作区保留 | 完成 |
| 输入文件 | 左/右RGB各200，标签200、置信度200、候选SGBM深度200；共1000份存在、形状、数组有限性及适用范围检查，问题清单为空 | 完成 |
| 原语义输入对应 | 200份左目RGB与历史推理输入哈希一致；标签/置信度400份与10/06缓存哈希一致 | 完成 |
| 唯一基图 | 907,488 Gaussian，文件及{len(base['fields'])}字段哈希均匹配原冻结清单，稳定ID唯一 | 完成 |
| 模型来源 | 检查点实测哈希与生成记录匹配；生成命令、类别表、processor与生产源码已归档 | 完成，有来源限制 |
| 标定与位姿 | RGB尺寸370×1226，基图K与P2内参一致，200位姿有限，时间间隔0.1037–0.1042秒；相对首帧位姿误差最大5.46e-5 | 完成，有坐标限制 |
| 独立语义GT | 审计输入中未找到；预测标签不能作为GT | 已查明缺项 |
| 原RAFT深度 | 原`s3_seq08_work`目录消失；当前SGBM不自动替代 | 已查明缺项 |

## 2. 输入和基图锁定

唯一基图服务器文件：`{old['base_remote']}`；本地镜像：`inputs/base_map/params.npz`。

SHA256：`{base['sha256']}`。保存字段、形状、类型和各数组哈希见`base_validation.json`及原`inputs/base_map_manifest.json`。`source_gaussian_index`作为对应身份，不能只检查数组长度。

原始源目录：`/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/08`。本次恢复的版本化入口：`{manifest['reference_root']}`；本地编号000000–000199对应原始003000–003199。完整逐文件源路径、引用路径、字节数、哈希、形状和检查结果见`reaudit_20261007_v1/input_inventory.csv`。

入口是指向共享原始数据的符号链接，不是数据备份。共享数据未修改。运行前必须用本清单复核源文件存在及哈希；失效或变更应停止并建立新输入版本。旧10/06记录及缓存哈希保留，不将消失的旧路径继续写为可用。

## 3. 模型、字段及预处理

输入来自`facebook/mask2former-swin-small-cityscapes-semantic`，revision `91281d0492f2175fc409c66a231bc2ff9c4ebaf0`；历史生成时间2026-09-15，生成窗口3000–3599，本轮取前200帧。模型权重SHA256：`{model['checkpoint_live']['sha256']}`。

19类映射见`id2label.json`，硬标签ID为0–18。置信度是“选中类别语义证据 / 该像素19类证据之和”，不是实例置信度，不是完整类别分布，也未证明概率校准。后续可做硬标签加标量软权重，不能据此虚构19类概率。

归档processor配置启用resize/rescale/normalize，配置size为384×384、divisor32，mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)。当前生产源码调用processor并将语义输出还原到原图尺寸。历史生成记录未锁定生产源码及processor文件哈希；本次快照只能证明当前配置和代码，不能补造历史生成时版本证据。生成摘要的待验证状态需结合后续`verification_report.json`及本次实测哈希解释。

## 4. 位姿、深度与标定边界

基图历史使用KITTI08里程计GT位姿，不是ORB估计位姿。存储`gt_w2c_all_frames`对应`inv(c2w_i) @ c2w_first`，不能当成绝对世界位姿。现有地图相机约定在本轮固定；P2/P3完整投影第四列保留，不能仅因K匹配就宣称全部相机坐标变换已验证。后续关联审计需核查里程计相机与左彩色相机约定，任何修正另开统一输入版本。

现存SGBM候选是float32米制深度，0为无效。200帧有效像素比例范围{checks['depth_positive_fraction_min']:.4f}–{checks['depth_positive_fraction_max']:.4f}。当前源码默认裁剪0.5–80米，但缺少原次生成配置回执，不能认定所有参数已冻结。原解析器采用x投影项差除fx计算baseline={cal['stereo_baseline_m']:.9f}米；完整P矩阵求相机中心距离={checks['full_projection_center_distance_m']:.9f}米，原公式未处理非零tz。本次不重算缓存、不改相机或地图。

**SGBM仅作为新候选资产。** 后续S-G/S2前须确定并冻结深度输入版本和无效/混合区域处理；恢复RAFT或采用SGBM均需新记录、各组共用，不能与历史RAFT结果直接混称同一配置。

## 5. 曝光与划分

几何、RGB以及历史语义均已见原始3000–3199全部200帧。本轮保留既有180/20工程划分：本地i%10!=9允许融合；i%10==9的20视角排除本轮语义融合。精确原始帧列表见`inputs/current_input_manifest_20261007.json`。

20视角只能称“本轮语义融合排除视角”，不能称几何未见、独立新Test。清空语义不消除历史数据曝光。各组重新创建零类别证据、零支持、零累计权重和零历史冲突状态；禁止重新读入基图旧语义作初始化。初始化/保存恢复实现及原生渲染一致性由01验收，本次输入审计不替代它。

## 6. GT检索与评价门槛

审计范围为`slam_project/data/kitti/dataset`和`datasets`，排除Scene20、输出目录及预测根；范围记录见`gt_discovery.json`。找到的semantic_mask2former标签属于预测。现有VKITTI2实例标签或colors类别表不能直接算作本窗口独立像素语义GT。本结论仅为审计范围内未找到，不声称服务器或数据集绝不存在GT。

没有独立GT时，O-Sem跳过；不得报告真实语义mIoU、风险—覆盖准确率或方法科学排名。稳定/动态变化、小目标/非小目标、边界/内部等共同评价子组也不能先用预测掩码冒充GT冻结。工程输出及教师一致性若后续开展，须单列其证据边界。步骤00已经交付GT缺项审计，不等于GT条件通过。

## 7. 下一步与验收状态

下一步为01：从同一基图创建各组语义副本，核验清空字段和Gaussian身份；审计原生遮挡及数值语义通道，做前景unknown遮挡背景的受控测试及各组RGB容差检查。CPU契约测试与源码准备已有10/06记录，实际原生渲染核验尚未执行。

进入真实S0/S1/S-G/S2/S2-H实验前，还需冻结可见关联缓存、深度输入、独立GT与共同子组。候选S2-Z/S2-V、历史冲突反馈及学习扩展不提前启动。本次没有修改Stage2冻结结果。

机器验收：`reaudit_20261007_v1/completion_checks.json`；远端原始命令回执在00根目录。输入证据包v2 SHA256：`4cbde689407238c4b6d0159a0403ffcaee4868d6910c0ada64fee9c2763e9346`。本次结论与更新文件另以逐文件哈希同步到阶段3专属目录。
'''
(step / '00_审计完成报告_20261007.md').write_text(report, encoding='utf-8')
(step / 'README.md').write_text('''# 00 输入、基图与GT曝光审计

2026-10-07审计完成，有明确限制。最新入口：`00_审计完成报告_20261007.md`，逐文件清单及回执见`reaudit_20261007_v1`，活动输入版本见`../inputs/current_input_manifest_20261007.json`。

可进入01工程核验；独立语义GT缺失、原RAFT深度旧路径消失，科学准确率实验尚不具备条件。SGBM候选不自动替换历史RAFT。10/06记录保留作为历史快照。
''', encoding='utf-8')
(root / '执行进度与下一步.md').write_text('''# 阶段3执行进度（2026-10-07）

**00输入、唯一基图与GT曝光审计已完成，有明确限制；下一步进入01工程核验。** 最新报告见`00_输入与基图_GT曝光审计/00_审计完成报告_20261007.md`。

本地00–06目录及服务器专属目录`/root/autodl-tmp/stage3_chy_20261006`保留。本次已连接新端口16680并恢复原始输入的版本化引用：左右目各200张、标签200份、置信度200份、候选SGBM深度200份，共1000份核验。200份左目与历史推理输入哈希一致，400份语义/置信数组与原缓存一致；907,488点唯一基图及全部字段哈希匹配；模型检查点哈希匹配生成记录。

活动输入入口是`inputs/current_input_manifest_20261007.json`，新逐文件清单是`inputs/input_inventory_20261007.csv`；旧清单与10/06归档保留。几何和历史语义已见全部200帧，180/20只区分本轮允许语义融合与排除视角，均属于已曝光DEV，不是新Test。

独立语义GT在审计范围内未找到；原`s3_seq08_work`及RAFT输入旧路径已消失。现存SGBM仅是候选，不自动替代历史RAFT。没有独立GT时跳过O-Sem，不签发语义准确率、风险—覆盖质量或方法科学排名。

01已有CPU契约与适配代码准备，10/06的10项CPU测试通过；尚未完成真实原生渲染、unknown遮挡、RGB容差、可见关联缓存等核验。本次没有GPU计算，也没有执行五组真实DEV。

下一步顺序：01初始化/原生渲染工程核验 → 确定并冻结深度、可见关联及GT评价条件 → 02真实五组DEV（GT可用才执行Oracle）→ 03方法冻结与隔离评价。S2-Z/S2-V及学习扩展仅在触发条件成立后启动。Stage2冻结及正式FAIL边界保留。
''', encoding='utf-8')
plan = root / '阶段3_语义高斯融合与可靠性验证_新计划_20261006.md'
text = plan.read_text(encoding='utf-8')
text = text.replace('“使用KITTI08地图”不足以确定实验输入。当前尚未选定具体基图文件，必须在步骤0审计后填写，不以本计划假定某个资产已合格。', '“使用KITTI08地图”不足以确定实验输入。2026-10-07步骤00已锁定具体基图与输入版本，详见`00_输入与基图_GT曝光审计/00_审计完成报告_20261007.md`及`inputs/current_input_manifest_20261007.json`；当前仅通过输入工程审计，独立语义GT与历史RAFT深度条件仍缺项，不据此签发科学评价合格。')
plan.write_text(text, encoding='utf-8')
files = [p for p in e.rglob('*') if p.is_file()]
files += [step/'00_审计完成报告_20261007.md', step/'README.md', root/'执行进度与下一步.md', plan,
          root/'inputs/current_input_manifest_20261007.json', root/'inputs/input_inventory_20261007.csv',
          root/'scripts/audit_step00_20261007.py', Path(__file__), root/'scripts/remote_session.py']
hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
write(step/'step00_sync_manifest_20261007.json', hashes)
bundle = step / 'step00_completion_sync_20261007_v1.tar.gz'
with tarfile.open(bundle, 'w:gz') as tar:
    for p in files + [step/'step00_sync_manifest_20261007.json']:
        tar.add(p, arcname=p.relative_to(root).as_posix())
print(json.dumps({'checks': checks, 'sync_file_count': len(hashes), 'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}, ensure_ascii=False))
