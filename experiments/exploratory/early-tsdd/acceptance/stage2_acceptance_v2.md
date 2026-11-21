# 阶段二 M3 验收裁决 v2（2026-10-01）

这是冻结裁决的本地便捷副本。包内路径以 [archive_20261001_v1](archive_20261001_v1/REPRODUCE.md) 为根；完整包及本地校验记录与本文件同目录。


**08审查和复现归档已完成；M3未完成，不能签正式PASS。**状态为 `M3_NOT_COMPLETE_ENGINEERING_ARCHIVE_COMPLETE`。CHY独立执行；按用户指示06保持暂停/FAIL。08只检查、登记、保存已有记录，未新增推理、改标签、调参、消费test或重置旧one-shot。

## 验收依据

无CARLA方案的五条M3建议条件按真实证据分别审核。早期01/03/05是9/25历史版本；之后Vehicle-only v2与三次批准的例外只改变车辆Scene20子协议，不把IMS/Pedestrian或完整三态地图一并升级。旧文档与记录均保留，不回填签字。负责人的批准原件放在evidence目录；不要求队友替CHY的独立归档签字，也不虚构独立盲审。

| 条件 | 状态 | 依据/不足 |
|---|---|---|
| C1 Contracts, GT, splits and source independence | PARTIAL | G2 18-frame engineering passed; later Vehicle-only protocol restricted. IMS conflict/full label support and independent KITTI object GT not completed. |
| C2 Pre-registered binary/coverage Gate B for eligible classes | FAIL | Scene20 primary clone F1=0.096195<.75; Pedestrian not formally evaluable. |
| C3 Temporal/rejection and transition GT evaluation | NOT_COMPLETE | T2.3 engineering output exists, no formal calibrated transition/coverage acceptance. |
| C4 ORB/map system benefit and ghosting | ENGINEERING_COMPLETE_WITH_LIMITATIONS | No ORB 25% ATE improvement; shared-view D3 does not reliably improve quality; ghost ROI support insufficient and no vacated GT. |
| C5 Reproducible evidence and failure preservation | ARCHIVE_AUDIT_PASS | 19 P6 receipts schema/semantic validated and immutably registered; original raw prediction manifests retained; full07 dependencies hash checked. |

## 01–08当前完成层次

| 步骤 | 已完成 | 未完成或限制 |
|---|---|---|
| 01 数据路线 | 三数据源、CHY范围、来源路径记录 | 旧双方签收栏不补写；本次按用户独立执行范围归档 |
| 02 G0 | 数据/支持度/容量/来源重叠审计 | 样本哈希不证明全部来源独立；行人支持/冲突未解决 |
| 03 G1 | KITTI08固定曝光窗口、VKITTI scene隔离；后来Vehicle-only受限GT/split | 非双类正式GT；KITTI08不是新盲测 |
| 04 G2 | 三源18帧adapter smoke及后续VKITTI开发内容核查 | 旧smoke不等于全三源正式开发GT通过 |
| 05 开发/时序 | motion几何预检、IMS smoke；后来车辆候选/技术锁与DEV诊断 | 完整时序/拒判/转换GT验收未完成；工程阈值不迁移为科学PASS |
| 06 Test | 三次消费全保留，第三次完成评分 | 正式FAIL；修复已暂停，无可重跑许可 |
| 07 系统 | O0–O3、D0–D3、O3→D3、200视图补充指标/VRAM/日志和完整本地备份 | 无可靠收益；无物体vacated GT；完整候选晋升/track-to-Gaussian/历史清理未证明 |
| 08 审核/归档 | 五条件裁决、19份P6回溯收据、校验/注册、SHA、环境与依赖包 | 完成的是验收任务和归档，不能把M3未达标改成PASS |

## 06消费与正式失败

| 次数 | 原run ID | 结果 | 可覆盖/再用？ |
|---|---|---|---|
| 1 | scene20_vehicle_20260928_01 | TECHNICAL_ENTRY_FAIL_NO_FORMAL_SCORE | 否；保留原消费 |
| 2 | scene20_vehicle_20260928_repair_02 | SHUTDOWN_INTERRUPTED_NO_FORMAL_SCORE | 否；保留原消费 |
| 3 | scene20_vehicle_20260929_recovery_03 | FORMAL_MODEL_FAIL | 否；保留原消费 |

第三次primary Scene20/clone：F1 **0.096195**，precision 0.070754，recall 0.150202，PR-AUC 0.208593；有效GT对象相邻帧对支持 15376，正例支持 4940，不是独立事件数量。
全部变体Vehicle F1 0.090270，prediction coverage 0.219066；各变体来自同一Scene20，不能加成独立场景。模型和双类正式PASS标志均false。三次消费registry、技术失败封存、关机封存、批准与最终指标原字节归档。

## 07最新工程结果

| split | D0 | D1 | D2 | D3 | O3→D3 | D0–D3共同有效 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 15/20 | 4/20 | 15/20 | 15/20 | 16/20 | 4 |
| exposed_heldout | 15/20 | 7/20 | 14/20 | 14/20 | 13/20 | 7 |

KITTI08 3000–3199已曝光，Train160与Dev20/留出20本轮不交；留出不是新的盲Test。7个共同有效留出视图D3全图PSNR14.9907、D0为15.0274dB，无可靠D3收益。full/static proxy指标定义、单尺度SSIM、normalize=True LPIPS、SGBM一致性、VRAM采样范围均见归档07报告，不与旧masked指标混用。Ghost代理Dev只1个、留出0个视图满足128ROI支持，不满足预登记3视图条件，不能把未估计填0。

## P6与可复现证据

- 19个新P6 ID是08命名空间中的回溯工程收据/归档收据：15个10/1既有07执行、3个Scene20历史尝试的ARCHIVE_ONLY审计、1个08总体归档。不是19次新实验。
- 19份均通过原P6 schema和语义校验，唯一注册到本次独立registry。相同ID再次注册退出4，registry SHA前后相同；不修改旧注册表。
- ARCHIVE_ONLY的PASS只表示取证成功；Scene20原模型仍FAIL。原P6校验器不检查所有文件内容，所以另做evidence_index所列证据、431个冻结输入、15个退出码、5个评价地图及完整07两包的实时SHA核验。
- 08环境为10/1重启后捕获，不是所有9/28–29运行环境的完整证明。15项时间仅保留日期精度，成本来自原日志；不从mtime伪造精确开始时间。
- P5源码快照无.git，写NOT_A_GIT_WORKTREE，保存源码和Python SHA；float export与之前CHY helper的实际diff另归档。ORB若有Git则保存真实commit/status/diff。依赖版本列表只反映安装快照，未验证干净主机重新安装成功。
- 完整07包中有808个指向服务器原数据的软链接，Windows只解普通文件。原始KITTI/SGBM/预训练权重仍为外部输入，需要按原路径和哈希另准备。08轻量包不假装含全部原始数据。

## 当前交付与复核入口

- `stage2_acceptance_v2.json/md`：验收裁决。
- `manifests/*.json`、`run_registry_v1.json`、`p6_validation.log`、`registry_immutability_check.json`：P6与唯一性证据。
- `evidence_index.json`、`archive_verification.json`：原件路径、SHA、字节数与实际核验。
- `one_shot_history.json`：三次消费失败沿革；原件在evidence/06*/vehicle_only_v2内。
- `environment.json`、`dependency_versions.txt`、`source_receipts.json`、`p5_float_export.diff`：环境/源码来源。
- `REPRODUCE.md`、`verify_stage08_bundle.py`：安全重验方式；默认只读校验，无推理和重新注册。
- 本地08工作目录为 `C:\Users\LOCAL_USER\Desktop\fyp\阶段2\08_M3验收与复现归档`，独立备份在 `9.28赶进度/阶段二_20261001_08_M3验收与复现归档`。
- 必需大证据依赖是上一批07完整包与DEV矩阵包，按ARCHIVE_DEPENDENCIES.json中的相对备份位置获取；两包已有完整本地校验。

## 后续研究缺口（本轮不执行）

1. 06：按用户意愿暂停。将来若有新算法思路只能先做DEV，不重用Scene20消费记录。
2. 物体状态/转换/vacated GT与行人支持度、来源独立性仍需解决；否则维持受限类别/工程主张。
3. 地图层需实现或收窄真正实例到Gaussian、Uncertain候选晋升、历史Dynamic清理等主张；当前birth-frame字段不是object track ID。
4. 在新的未曝光窗口和冻结协议上验证系统收益，完整统计单位/coverage/失败支持；不得以改门槛、换最好场景或高拒判追认PASS。

**本次裁决执行者：Codex代CHY整理取证；未代签独立评审或负责人接受声明。用户未提出恢复06，本次只完成08审查与归档。**

## 历史运行登记范围

19份P6覆盖本次明确列出的回溯/归档收据，不声明全部历史DEV探索都已补写统一P6格式。其余旧run_manifest保留原生格式和源哈希；O0–O3旧轨迹/feature/console额外归档并核对原轨迹、feature SHA，见orb_historical_evidence_index.json。原计数重建primary/overall F1与保存值一致，无新推理或正式重评分。

