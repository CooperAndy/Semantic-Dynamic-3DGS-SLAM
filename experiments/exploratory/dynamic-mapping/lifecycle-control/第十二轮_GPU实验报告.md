# 第十二轮：40 帧 D0/D3、P5 active 分支及 Scene02 事件

## 结论

完成已曝光 KITTI08 Train 的 40 帧 D0/D3 配对，D3 在其自身运行中拒绝 13,474 个有效深度候选，最终地图比 D0 少 4,651 个 Gaussian。D3 在本窗口全图平均 PSNR 和 MS-SSIM 略低，LPIPS 略低；不能据此宣布质量提升或 ghost 改善。

完成独立 P5 active 渲染和保护 optimizer step 的初版，并运行 5 帧固定 ID 注入：64 个 Gaussian 在帧 1–2 停用、帧 3 恢复；实际 checkpoint 中停用期五组 Gaussian 参数逐值不变，恢复帧均发生更新。但**原生 checkpoint 恢复、出生帧/关联证据完整保存、prune/clone/split 和 Adam step 的完整语义尚未解决**；代码显式禁止 resume、prune、densify，不能标记完整生命周期实现。D3+Lifecycle 的真实车辆实验前置条件不满足，本轮未执行 Oracle 或 prediction-only 生命周期对照。Mask2Former 训练保持 BLOCKED。

## 40 帧实验预先固定

KITTI08 Train160 索引 68–107；原始帧 3085–3133 中按冻结 split 删除 Dev/heldout，实际帧列表和 9 个断点保存在 `results/input_freeze.json`。包括 3085–3088、3090–3093 等 4 帧连续片段，不能称为 40 个连续生命周期帧；3089 等留出帧不进入优化。D0/D3 同样使用 192×640、RGB、SGBM 预测深度、O0 ORB 估计位姿、seed 20260929、每帧 1 次 mapping iteration、相同 checkpoint 与评估频率。40 帧选择依据为原 Train 索引连续区间，先于结果固定；不是新盲测。

40 帧 RGB/深度各与旧 train160 输入审计哈希相符，状态文件与各自侧车哈希一致；window 轨迹由原 Train 轨迹对应行生成。120 条输入记录、源帧、路径及 SHA 在 input_freeze.json；源状态与旁侧车哈希一致不等于独立可信状态。P5 源目录和独立入口在启动前快照并保存 source_sha256.txt；历史 `base_mapping_config.py` / `base_train_config.py` 仍由配置动态导入，不在该源快照内，故不能宣称所有配置依赖均已完整冻结。记录的实际 config 在两组输出中，后续需将这些依赖一并固定。

| 40 帧指标 | D0 | D3 |
|---|---:|---:|
| 最终 Gaussian | 216,973 | 212,322 |
| 自身有效深度候选被状态拒绝 | 0 | 13,474 |
| 平均 PSNR dB | 15.9320 | 15.8876 |
| 平均 MS-SSIM | 0.75821 | 0.75767 |
| 平均 LPIPS | 0.35964 | 0.35845 |
| 进程墙钟秒 | 54.56 | 53.03 |
| 逐帧记录 peak allocated 最大 MiB | 217.36 | 216.27 |

D3 拒绝中 Dynamic 8,085，Uncertain 5,389，Ignore 0；这些是 D3 自身候选中的数量，不等于两组最终地图差，后续候选会随地图改变。历史评估器对无效输入深度置零，表中为 40 帧算术平均。按状态与有效深度像素交集、silhouette >0.5 加权统计：Static 两组 99.9905% / 99.9894%，Uncertain 99.9968% / 97.2001%，Dynamic 99.9667% / 85.7593%（D0/D3）；状态来自预测侧车，不能视为真实物理标签，且无效深度不在分母。逐帧 CSV 提供支持、创建数、状态拒绝、时间、显存、指标及区域。原审计代码未记录逐帧 loss，日志中的 loss 无法事后补齐；逐帧 peak 从每帧循环开始，不含首帧初始化，且不是整机显存峰值。所有 40 帧原始评估图、浮点渲染和 checkpoint 在服务器，精简包取固定帧 0/10/20/30/39 的渲染，未复制大地图。

## P5 active 初版和局限

独立源码 `p5_integrated_dev` 从第九轮 DEV 代码另建：在主程序及独立 eval helper 将 CUDA renderer 包装为仅输入 active 行并把 radius 散回原索引；tracking/mapping 渲染、loss 渲染、新增候选渲染和本支路评价调用该 wrapper。两处 optimizer.step 使用逐行快照恢复保护全 Gaussian 参数及匹配形状 Adam 状态；相机参数不按 Gaussian 行处理。代码 diff 与事件日志均已保存。5 帧测试固定窗口前五个 Train 帧、帧 1 注入停用 ID 0–63、帧 3 注入恢复，不涉及真实车辆。checkpoint 0–4 的总数为 82,757、86,969、89,847、92,063、96,972；inactive 数依次 0、64、64、0、0，ID 数量对齐。实际 checkpoint 参数对照：从 0→1、1→2，五组停用行最大差均 0；2→3 恢复后五组最大变化分别约 1e-4、0.0025、0.001、0.05、0.001。没有与固定 checkpoint 的相同视角独立分支再次测渲染恢复；第十一轮独立 rasterizer 对该属性的受控测试通过，但不同于本次整条 P5 路径。

额外固定 checkpoint 真实 rasterizer 五组 Adam 工程测试：inactive 参数、exp_avg 和 exp_avg_sq 保持不变，五个活跃参数组均改变；GPU 1.89 秒。测试只覆盖一步默认 Adam，标量 step 会在 inactive 行冻结期间增加，恢复后偏置校正语义尚未解决；不支持 AMSGrad 等变体。P5 5 帧墙钟 11.02 秒、最后区间 peak allocated 144.40 MiB；不包括所有 CUDA 扩展的显存。未做原生重启恢复后继续建点，不支持 actual prune/clone/split：元数据当前将 next_id 设为当前 N，只适用于禁用删除和重排的本次 append-only run；没有真实物体证据及完整 birth/state/事件持久化。显式禁止不安全路径，真实生命周期仍 NOT READY。

## Scene02 事件及资源

冻结 CPU 离线核查 Scene02/clone 206–231；官方 track 9 在 214→215 由 `isMoving=True`→`False`，状态之后维持到 231。对每帧在已有历史预测中寻找单帧最大 mask IoU：26/26 帧 IoU≥0.5，最佳预测 track ID 均为 110；214 和 215 的 IoU 分别 0.7656、0.7506。该统计逐帧独立取最大值，不是全局一对一匹配或物理身份保证，GT 仅用于离线诊断；`scene02_track9_206_231.csv` 保存逐帧 GT 和预测文件 SHA。

现有 bridge 清单声明 0–232 使用 RAFT-Stereo 预测深度和 ORB-SLAM3 Stereo 估计位姿、T2.3 预测状态；206–231 各 RGB、深度、侧车及整体轨迹的 SHA 对照 bridge manifest 通过。该数据已有历史曝光，侧车实例列表为空。真实 Gaussian 与车辆的可信几何归属尚未建立，持续预测 ID 不足以消除跨车污染；标注变化也不证明车辆离开或腾空。无 vacated GT，ghosting NOT_EVALUABLE。依赖前述完整生命周期实现，故 Oracle 与 prediction-only 生命周期对照均尚未执行；不能把这些覆盖统计当作 D3+Lifecycle 结果。

## GPU 与文件

本批 GPU 进程墙钟：D0 54.56 秒、D3 53.03 秒、五组受控测试 1.89 秒、5 帧 P5 初版 11.02 秒，合计约 120.50 秒，远低于授权累计 2 小时；时间不含准备和 CPU 审计，不表示服务器计费时长。服务器末次 `/root/autodl-tmp` 约剩 28 GB，95% 使用。执行结束后没有满足完整生命周期条件的后续实验，停止而非运行无意义任务。

完整服务器输出：`/root/autodl-tmp/stage2_round12_20261031_v1/maps/{D0,D3}`（逐帧 params 和 eval）、`/root/autodl-tmp/stage2_round12_p5_integrated_v1`、`/root/autodl-tmp/stage2_round12_lifecycle_control_v1`。最终地图 SHA-256：D0 `62f23cc1216e9d2ecbb596eefc0a5c824c3fc5a980f54b33c1190904c558dddb`；D3 `3d79cc07848865cbbc6288d8b0965d463146b50bff3f8b7c43b4cbf1ce2e6139`；5 帧集成 `08a9b9da346c8114da6d53329aa76286a877bea24af6292aca37c40869640f07`。历史 D0–D3、FAIL、Scene20 和正式规则未改。
