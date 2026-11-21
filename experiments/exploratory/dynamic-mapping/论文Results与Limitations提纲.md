# FYP Results 与 Limitations 中文提纲

## Results 1：静态重建与深度可靠性

描述固定的 ORB-SLAM3 Stereo 估计位姿、SGBM / RAFT-Stereo 深度和 SplaTAM 建图；E6a 依据有效深度、天空掩码、左右一致性、光度和时序几何一致性构造置信度，采用冻结的 `C≥0.80` hard gate。KITTI00 和 KITTI07 分别在各自 0–199 帧中留出 10 帧，留出帧不参与建点、mapping loss 或关键帧。临近帧仍参与建图，结论限定为同序列邻近视角。

| 序列 / 方法 | 留出帧数 | PSNR dB | MS-SSIM | LPIPS | Gaussian 数 |
|---|---:|---:|---:|---:|---:|
| 00 E3-SGBM | 10 | 16.419 | 0.7467 | 0.3223 | 1,762,556 |
| 00 E4-RAFT | 10 | 14.147 | 0.6080 | 0.4846 | 1,908,598 |
| 00 E6a | 10 | 24.066 | 0.9091 | 0.1439 | 3,235,712 |
| 07 E3-SGBM | 10 | 18.074 | 0.7977 | 0.2565 | 2,195,795 |
| 07 E6a | 10 | 26.054 | 0.9307 | 0.1075 | 3,400,900 |

E6a seq00/seq07 mapping 约 3.29/3.39 s/frame，历史 peak CUDA 2.72/2.77 GiB；不可称为端到端 FPS。参见第十六轮 `FYP_results_table.csv` 和原始 `fair_holdout_summary.json` / `cross_scene_summary.json`。原 evaluator 的 depth RMSE 文件实际等价逐像素绝对误差均值，须称 Depth L1。

## Results 2：动态检测与运动分类

正式动态分类 Scene20 F1=0.096195 是**失败记录**；该数值来自用户提供及历史摘要，本轮未读取/重新评分正式 Test 原始数据，论文引用前与已有正式封存摘要核对。保留 Scene20 FAIL，不写 PASS。

Scene01/clone 0–50 的离线审计有 1,959 个有效 GT 相邻帧对：1,256 两端无合格匹配、142 仅一端匹配、561 两端匹配（544 预测 ID 相同、17 不同）；这是指定匹配规则下的帧对，非独立车辆。ID32 34→35 的跨 GT 身份继承会改变局部 3D 特征支持；第四轮 GT 身份 oracle 局部对照没有改变当时 v5/v7 决策，不能推广为其他错误无影响，也不能证明修 tracker 提升总体 F1。v5/v7 是历史 DEV 候选而非正式通过的系统。

10 月 5 日概率融合是已曝光的条件 DEV：21 个 episode / 21 个物理 track（Static 10、Dynamic 11），另有 2 个 episode 缺失。原 P1 该条件子集 F1 0.9091；跨场景 train Scene01→validate Scene02，Logistic Regression 与 Gradient Boosting 各 11 episode、FP 4、FN 0、F1 0.75；反向 10 episode，Logistic F1 0.75（FP 0/FN 2）、Gradient Boosting F1 0.6667（FP 1/FN 2）。不能把条件支持子集分数当成全体 GT 或独立测试性能；不称为部署改进。详见 `results/概率融合DEV结果.csv` 和融合原始 `results.json`。

## Results 3：动态 Gaussian 创建过滤

使用第十六轮结果图和候选表，写明首帧及后续候选形成方式不同，统计的分母是“原始实际候选 ∩ 有效预测深度”，GT 只参与离线评价。Scene06/clone 130–137 中，GT Moving 候选 24,885 像素，Dynamic 拒绝 24,636；GT Static 候选 297，Dynamic 拒绝 1。Scene02/clone 206–231 中，GT Moving 候选 35,098，Dynamic 拒绝 25,432；GT Static 候选 2,507，Dynamic 拒绝 1,783（71.12%）。Uncertain 分开报告，不能和 Dynamic 合并；GT 未知不写成静态背景。

两场景均为已曝光 DEV，不同时刻像素不独立；Scene06 候选像素是事后复算且仅逐帧计数与原日志吻合。Scene06 track12 标注运动切换对应 3D 位移较小；不能把官方 isMoving 标签直接解释为明显物理运动或腾空。

Scene02 D0/D3 26 帧最终地图对建图视角回看分别为 PSNR 19.035/19.530、MS-SSIM 0.86647/0.88074、LPIPS 0.20474/0.19964，Gaussian 88,317/87,049；均使用相同预测 RAFT-Stereo 深度、ORB 估计位姿与 seed。不能把它们写成留出视角或 ghost 改善。其他 3 帧、40 帧和 Scene06 表见 `阶段2/第十六轮_动态创建过滤与成果_20261031/results/FYP_结果说明.md`。

## Results 4：受限 Oracle 生命周期

Scene06 130–137 的标签驱动 Oracle 将出生于 GT mask 的 296 个 Gaussian 标为候选并于 135 帧停用，真实 rasterizer 与参数冻结机制运行成功；GT 深度离线复核发现 23/296 点误差>1m，且车辆运动与 vacated 背景支持不足，物理归属不可保证。第十五轮使用每帧后快照在线评价：135 帧 D3/Oracle PSNR 19.5513/19.5435；136–137 均值 20.2282/20.2115。它是当前帧参与优化的在线训练视角，不是独立测试。Prediction-only 停用、可靠归属和完整恢复未验证，ghosting NOT_EVALUABLE。

## Limitations 与可用主张

1. 深度可靠性实验仅验证两个 KITTI 序列各自邻近留出帧，场景、点数、资源与其他实验不应跨协议直接比较。
2. 动态分类正式 FAIL；检测漏匹配、实例碎片和 Track ID 继承会降低覆盖或污染运动历史；不能将 DEV 条件子集写成全分母指标。
3. E6b、E6c 没有达到整体采用条件；E6c 在 50 帧配对中 PSNR/MS-SSIM 下降、LPIPS 改善，属于折中而非无条件成功。
4. 动态创建过滤对 Scene02 静态车辆候选有明显拒绝；预测状态可靠性、其他/未知背景和帧间依赖必须报告。
5. 真实对象归属、vacated-background GT、Prediction-only 生命周期与原生 P5 checkpoint/优化器完整续跑未完成；不得宣称 ghosting 减少或正式 M3 PASS。
6. PLY 是保留 3DGS 参数的导出，viewer 交互导入未经验证；视频是实际 Gaussian rasterizer 的固定轨迹回放，中心点轨迹图仅为空间诊断。

建议论文将贡献表述为“可靠深度输入筛选及其留出视角实验”“动态候选过滤的工程实现与带分母 GT 诊断”“生命周期可逆停用原型及其失败边界”，并对历史负结果单列。所有效果表旁写清场景、帧数、输入、是否见过、评估视角及分母。