# T04 阶段A实验启动记录

日期：2026-10-12。阶段A：异常增点资格修复；GS-SLAM观测一致性筛选和20米限制未启用。

已部署独立 `source_v3_depthguard`，核对V2父文件及补丁SHA256；原V2和队友研究分支未修改。现服务器为RTX5090 / sm120，PyTorch2.8.0+cu128。

旧扩展第一次运行出现no kernel image；独立重编译DROID、LieTorch、高斯渲染后，第二次运行在FAISS处遇到同类错误。最终复制现有T14的同版本FAISS1.10.0 sm120兼容二进制及C++运行库到本T04目录，SHA256及来源留档；仅复用兼容组件，不使用其研究结果。所有修改通过本次子进程PYTHONPATH/LD_PRELOAD生效，共享环境未修改。

40帧测试 `KITTI00_200_A_smoke40_20261012_03` 已COMPLETED，退出0，24关键帧，58,963高斯。检查了每次mapping后的有限性、相对当前相机距离和激活尺度，未出现旧百万米级异常；本窗口最大当前相机距离、尺度及耗时见launch_receipt.json。尺度仍需几何质量检查；工程阈值不代表地图质量验收。

200帧 `KITTI00_200_A_on_20261012_01` 已实际启动。远端回执：`/root/autodl-tmp/t04_chy_kitti00_200_20261012/stage_a/cuda_sm120/resume_receipt.json`；日志 `stage_a/full200_ready.log`，退出码 `stage_a/full200_ready.exit`；结果 `04_reproduction/KITTI00_200_A_on_20261012_01/`。启动时local full200_live_manifest.json只是一份运行中快照，不能视为完成证据。

后续必须回收完整manifest、exit、轨迹/pose stages/地图/范围历史及日志，验证帧数与哈希，再独立CPU评测ATE/RPE。不得依据40帧工程通过宣称精度改善。A将历史部分8米采样路径扩大到40米，且GPU/扩展架构发生变化，报告须列出这些条件；必要时用同一5090兼容环境重跑V2对照。

管线在本轮200帧后结束，不会自动启用B/C。若范围异常仍存在，先定位首次异常出现于采样还是地图优化，不能直接删除离群点后宣布修复。

## 完成回执

200帧已COMPLETED，退出0，完整轨迹/地图及身份哈希验证通过。评价报告见 `05_报告与归档/阶段A增点修复评测_20261012/阶段A结果与下一步.md`。极端远点未再出现，tracking指标混合，未采用为精度改进。没有启动B/C或V2复跑。
