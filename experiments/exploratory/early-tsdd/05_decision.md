# 05：T2.2 development 与 T2.3 工程烟测结论

日期：2026-09-25；执行人：CHY。状态：`T22_DEV_PREFLIGHT_PASS / T23_ENGINEERING_SMOKE_PASS / G3_NOT_FROZEN / NO_FORMAL_TEST`。

## 实际运行

1. 复用已曝光的 InstanceMotSeg drive0009、源帧 355–362 的 8 帧工程包。原包 `SHA256SUMS.txt` 核对 34/34 通过。其旧 RAFT motion evidence 分为 `GT-oracle instance` 与 `Mask2Former-predicted instance` 两路，每路 7 个相邻帧对；均使用 OXTS 派生的 cam2 pose、SGBM 伪深度和同一 RAFT 光流。`external_input_SHA256SUMS.txt` 另记录 31 个本轮直接使用的原包清单/证据文件哈希。旧 evidence 的 `probability_center=3.0`、`temperature=1.0` 只是工程设置，未经过新 GT 校准。
2. `stage05_preflight.py` 调用 CHY 副本 `source/compute_motion_evidence_P1.py` 的刚体流函数，验证 identity pose 零流、相机 +x 位移预期负 x 光流、零深度无效；再对 G2 的 VKITTI2/IMS/KITTI08 各 3 个真实相邻帧对计算刚体流，9/9 有有效投影和有限数值。KITTI08 使用已修正的 `P0 c2w -> P2 c2w` 位姿。它还重读两路 7 对 RAFT/刚体流/残差 NPZ：shape、有效像素、有限值及概率范围通过；两路观测光流逐像素一致，候选实例来源仍分开。报告为 `preflight_report.json`。
3. 从共享 P1 实现与配置只读复制到 CHY `source/`、`configs/`，在 **predicted** 路线运行 `ENGINEERING_SMOKE` 的 `run_tsdd_P1.py`，输出 `runs/ims0009_pred_engineering_smoke/`。8 帧产生 55 个实例观测、其中 39 个有效证据；状态计数 Static 10 / Uncertain 35 / Dynamic 10，Uncertain 比率 63.6%。首帧全为 Uncertain；逐帧状态图 `uint8 [375,1242]` 且值域仅 `{0,1,2,255}`。这些是**预测输出分布**，不是正确率或物理 GT。未运行 `FORMAL`、test 或阈值搜索。

## 重要限制

- 现有 IMS 8 帧来自同一 drive、同一短窗口，并曾用于工程诊断；不能当独立样本量。G2 发现 oracle mask 中不同 track 完全相同的冲突，本轮保留旧 T2.2 证据做数值/接口预检，**不对这些 track 评分**。
- `GT-oracle` 路线只用于隔离几何问题；T2.3 本轮使用的 predicted 路线含上游检测/关联误差。KITTI08 的 6 帧仅作相机几何预检；已有 3000–3199 窗口曾曝光，且没有物体状态 GT。
- 没有可用于新路线开发校准的人工确认 Static/Dynamic、停走事件和行人负例；因此不计算 F1/PR-AUC、转换准确率或三态 accuracy，不选择新的 motion threshold，不冻结 `lambda/theta/confirm/missing`。旧 KITTI/SGBM 阈值及历史 seq07 FAIL 不迁移、不改写。

## G3 何时能冻结

先补 G1 的真实物体状态/转换标签和 G2 的预定开发集覆盖，处理 IMS 冲突 mask、候选实例与跨帧 track；再仅用 development 冻结 T2.2 的流/深度/pose/分数校准、T2.3 搜索矩阵与 Uncertain 覆盖预算。要保存全部候选结果、逐 track/事件表、失败例、配置与 SHA。此后才可形成 `tsdd_config_frozen.yaml` 和 G3 签收。当前工程 midpoint YAML **不是**该冻结文件，正式 one-shot 不放行。

本地目录是服务器 CHY 本轮运行的轻量备份；共享数据和旧代码原件未修改。`05_SHA256SUMS.txt` 校验本目录产物。
