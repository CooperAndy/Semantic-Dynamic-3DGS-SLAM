# 阶段 3 双目高斯 SLAM · KITTI 演示迁移手册

适用于在其他 KITTI Odometry 序列上复用 **MASS 位姿反馈 + H3 历史关键帧回放**，生成轨迹、RGB 高斯地图和可选的语义地图。以下服务器路径已在 `SERVER_HOST:15889` 核对。新序列需要重新准备输入和运行配置。

## 1. 复用哪套代码

服务器上的 S23 实验根目录和实际运行参考组如下。后文的 `$S23` 和 `$REF` 均指这两个绝对路径。

```bash
S23=/root/autodl-tmp/s23_effects_20261105_p16247
REF="$S23/E3_K05_S0_R3_serial"
G21=/root/autodl-tmp/stage3_g21_inputs_20261029_p47149
STEREO=/root/autodl-tmp/stage3_g21_stereo_20261029_p47149
ORB=/root/autodl-tmp/stage3_g19_causal_20261029_p47149
```

| 功能 | 服务器文件 |
|---|---|
| 双目 ORB 前端、词袋和标定模板 | `$ORB/bin/causal_stereo`、`$ORB/runtime/ORBvoc.txt`、`$ORB/inputs/settings.yaml` |
| KITTI 输入窗口、帧序和时间戳准备 | `$G21/prepare.py`、`$G21/common.py`，ORB 调用示例在 `$G21/controller.py` |
| 左／右 RAFT-Stereo 深度及实际运行参数 | `$STEREO/source/left.py`、`$STEREO/source/right.py`、`$STEREO/execution_cpu_registration.json` |
| 深度置信度和后端输入清单 | `$S23/confidence_bundle.py`，已跑示例为 `$S23/bundles/05/runtime.json` |
| 后端入口及高斯 SLAM 主循环 | `$REF/scripts/worker.py`、`$REF/source/scripts/splatam_external_tm.py` |
| MASS 双目提案、匹配与 PnP | `$REF/scripts/stereo_proposal.py`、`$REF/scripts/feature_pose.py` |
| MASS 验收和位姿反馈 | `$REF/scripts/backend.py`、`$REF/scripts/visibility.py`、`$REF/scripts/pose_chain.py`、`$REF/scripts/stable_pose.py` |
| H3 回放、建点与地图优化 | `$REF/scripts/factor_hooks.py`、`$REF/scripts/stable_birth.py`、`$REF/source/scripts/splatam_external_tm.py` |
| 配置模板 | `$REF/source/configs/kitti/B.py`（MASS + H3，保留标签融合）、`$REF/source/configs/kitti/TMD.py`（再启用语义定位、语义建图和动态过滤） |
| 语义输入与高斯语义状态 | `$S23/prepare_bundle.py`、`$S23/export_observations.py`、`$REF/scripts/semantic_bridge.py`、`$REF/scripts/semantic_runtime.py`、`$REF/scripts/online_state.py` |
| 运行目录生成、公共调度封存、地图评价 | `$S23/run_registered_matrix.py`、`$S23/matrix_evaluation.py` |

## 2. 跟踪与建图的实现

**Tracking · MASS。** 双目 ORB 提供米制名义位姿。后端用当前帧和过去参考帧的图像匹配，结合参考帧有效双目深度构建 3D–2D 对应，再用 PnP 生成修正提案。`Backend.refine` 对预测位姿和候选位姿分别渲染，在相同支持区域上检查残差和位姿变化。MASS 还要求候选保留至少 `0.98` 的连续不透明度质量。通过后将选中位姿用于当前建图，并与后续 ORB 相对运动组合进行预测。拒绝时恢复本步预测位姿，保留此前已传播的修正。`worker.py` 会安装 `stereo_proposal.py` 的双目适配，复用时应保留该入口的安装顺序。

**Mapping · H3。** 每个建图帧执行 60 次地图优化。有合格历史关键帧时，`factor_hooks.py` 将零基迭代 `15、35、55` 改为回放至少早于当前帧 200 帧的关键帧；其余迭代走原调度。回放使用历史帧的 RGB、深度、置信度和已选位姿。地图更新优化高斯参数，保持相机位姿固定。首帧新建地图，每帧新增候选上限为 3000。前 200 帧以内的短测可验证管线，但要检查回放实际触发，需要继续跑到出现合格历史关键帧。

**语义与动态模块。** MASS + H3 是几何主干。S23 的 `B` 组关闭 `T/M/D`，仍读取语义观测并融合高斯标签，适合先生成 RGB 和语义演示地图。`TMD` 组额外启用类别一致性筛选、历史语义几何损失和动态掩码过滤。沿用 TSDD 时使用阶段 2 的观测生产链。Pi³MOS 路径的参考配置在 `$S23/S23P_K05_S0/source/configs/kitti/TMD.py`，其车辆动静观测生成见 `$S23/S23P_20261110/s23p_inputs.py`。

## 3. 迁移到新的 KITTI 序列

1. **准备数据窗口。** 原始数据根目录是 `/root/autodl-tmp/slam_project/data/kitti/dataset/sequences`。核对目标序列的 `image_2`、`image_3`、`calib.txt`、`times.txt`。先选连续 1000 帧，沿用现有数据长度接口。保留原始帧编号映射，确保左右目、时间戳、深度、位姿和语义都对应同一帧。使用新序列的 P2/P3 内参、基线和分辨率，按 P2 左目坐标输出米制位姿。可用 `P2[0,3] - P3[0,3]` 重新计算 `projection_beta.txt`，单位为像素·米。`$G21/prepare.py` 会校验与旧标定模板一致；新序列标定不同时，应先生成对应的 ORB `settings.yaml`，再做校验。
2. **重建后端输入。** 按 G21 的窗口和 ORB 调用样例生成 `external_causal_P2_c2w.txt`，再用 G21 的左右双目深度脚本与 S23 的置信度脚本生成新窗口的 `runtime.json`。深度采用左目对齐的米制 `float32`，无效值为 0，有效范围为 0.5–80 m；时间置信度仅依赖当前帧和上一帧。参照 `prepare_bundle.py` 的生产顺序生成 `observations/manifest.json`。新 bundle 至少提供 `runtime.json`、`observations/manifest.json`、`calib.txt` 和 `projection_beta.txt`，清单引用的数据也需在服务器上可读。现有 `prepare_bundle.py --sequence` 只接受 `08/05/10`，G21 准备脚本也写死了窗口，应在独立演示目录适配序列与路径。若要改为不同长度，还需同步适配置信度、运行目录生成和输入检查中的固定长度。
3. **生成独立运行目录。** 参考 `$S23/run_registered_matrix.py` 的 `prepare_cell`，从 `$S23/candidate_v4_fp64accum` 复制 `source/`、`scripts/`，将运行目录绑定到新 `scripts/common.py`、`inputs/runtime/input_manifest.json`、`source/configs/data/g20_manifest.yaml` 和组配置。同步填写 `data.sequence`、`num_frames`、图像尺寸、`mapping.residual_fb`、语义清单、输出目录及 `registration.json`。参考配置的 `feedback_policy='MASS'`、`factorial.history_budget=3`、`mapping.num_iters=60` 和 `load_checkpoint=False` 可直接作为迁移起点。沿用留出评价时，同步配置 `eval_only_frames`；现有回放和审计还依赖每 20 帧末帧留出的规则。
4. **先跑 B，再按需要跑 TMD。** B 在新窗口生成本组的 `target_schedule.npz` 和 `addition_masks/`。若需要 TMD，对照 `run_registered_matrix.py` 封存本窗口的 `receipts/donor.json`，再启动 TMD。TMD 的 `donor_variant='B'` 需要同一窗口和种子的调度、候选及其哈希清单。旧序列的调度和建点候选与旧帧绑定。

完成上述准备后，可用下面的入口直接运行 B。`DEMO` 设为已准备好的新运行目录。需要 TMD 时，在封存 B 的输入后将末尾参数改为 `TMD`。此处调用单组 worker，适合将演示产物留在独立目录。

```bash
PY=/root/autodl-tmp/stage3_g15_20261028_p47149/runtime/venv/bin/python
export S23_RASTER_ROOT="$S23/raster_fp64_accum_v1"
export PYTHONPATH="$S23_RASTER_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg WANDB_MODE=disabled
cd "$DEMO/scripts"
"$PY" -u worker.py B
```

## 4. 演示验收与地图交付

- **输入检查。** 核对帧数、原始帧编号、标定、图像尺寸、清单哈希、深度和置信度有效域。检查 `tracking.use_gt_poses=False`，轨迹 GT、LiDAR 或语义 GT 仅用于后续评价。
- **跟踪检查。** 在 `runs/<组>/backend_events.jsonl` 检查 `geometry_ok`、`accepted`、`reason` 和 `selected_w2c`。如果没有接受记录，演示可展示运行轨迹，是否产生 MASS 位姿修正应据账本说明。
- **建图检查。** 在 `runs/<组>/mapping_updates.jsonl` 核对 `history_slot`、`frame`、`target`，确认出现有效回放且历史目标满足 200 帧间隔。实际帧数和运行状态查看 `runs/<组>/receipt.json`。
- **保留地图。** 将 `runs/<组>/params.npz`、`semantic_state_final.npz`、`source_gaussian_index.npy` 放入同一个演示目录并核验 SHA-256。`params.npz` 是高斯地图，另外两份分别保存语义状态和高斯 ID 对应关系。需要回放建图过程时，另保留 `params<帧号>.npz` 与同帧的 `semantic_state_<帧号>.npz`。正式实验的归档流程可能在回传后释放服务器大文件，演示副本应放在独立目录。

历史回放已在 KITTI08 长序列上验证能保持早期地图质量。MASS 在同一开发序列改善了整体轨迹，逐帧相对误差改善较小。新序列的演示结果可用同一组留出相机检查地图，并用 ORB 轨迹作定位对照。本手册核对了参考源码与服务器路径，新序列尚未按此手册实跑。
