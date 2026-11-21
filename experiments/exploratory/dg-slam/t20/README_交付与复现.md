# T20 交付与复核

首先阅读 [结果报告](T20_GPU补全诊断结果与下一步.md)。本轮完成 T19 登记的五组真实 GPU 机制诊断，不是新正式采用验收。关键帧输出约束通过，非关键帧质量问题未解决。

完整新结果、状态快照、逐轮 NPZ、日志、候选源码、协议、执行锁及 CPU 评分均在本目录。`dependencies/T17_scoring` 是原始 T17 的小型评分依赖快照，保留原数学函数和评价参考哈希。运行时不读取这个 GT 参考，只有封存后的 CPU 评分使用。

CPU 依赖为 Python、NumPy、SciPy；画图另需 Matplotlib。无需 Torch/CUDA。已生成结果只读保留；重新评分时指定新的输出目录：

```powershell
$env:T20_T17_ROOT = (Resolve-Path './dependencies/T17_scoring').Path
$env:T20_SCORE_OUTPUT_ROOT = Join-Path (Get-Location) 'new_cpu_check'
python ./scripts/score_diagnostics_local.py
Remove-Item Env:T20_T17_ROOT
Remove-Item Env:T20_SCORE_OUTPUT_ROOT
```

脚本会先核验全部 raw seal 文件，再访问 GT 参考。输出目录必须尚未存在；不要覆盖当前 `independent_cpu` 或原始 CSV。frame158 深度描述可只读复核：

```powershell
python ./scripts/source_depth_inspection.py
```

`independent_audit.py` 的历史 T17/T19 全量未变检查及旧轨迹比较，需要同级原文件夹。报告指向交接文档的链接也依赖同级目录；独立拷走 T20 时不应将这些外部路径当作包内文件。

GPU 环境依赖已封存的 Linux T14 sm120 原生构建与环境，不在本轮重复拷贝所有二进制。执行锁记录这些依赖的路径及 SHA256；不能在 Windows 直接运行原生 GPU worker，或将旧 sm86 二进制替换进去。完整重建需原数据、模型权重和 T14 环境。两份 `native_state.pt` 只保证本段固定状态补全回放所需输入，不包含继续完整 SLAM／高斯优化的全部前端、地图优化器状态。

`launch.py` 和 `candidate_gpu_smoke.py` 是保留的首次失败工程尝试；实际完成路径见 `launch_v2.py`、`candidate_gpu_smoke_v2.py`、`execution_lock_v2.json`。不要重新运行已有输出上的 launcher。所有 remote_console／下载脚本通过不回显提示接收认证信息，密码没有存盘；旧主机和端口只是本轮连接信息，之后应先确认仍有效。

原 T19 登记和 `anchor_policy.json` 的 `GPU_validated=False` 是保留的执行前候选标记；本轮真实执行是否通过，查看新的候选冒烟、各例 engineering_gate 和 launch_v2 回执，不能将冻结的旧标记当作当前进度。

本轮原始 ZIP 单独保存，完整交付 ZIP 另外包含新增报告、图表及本地独立审计。SHA256 和成员数见回执；源文件级复核使用 `delivery_manifest.json`。本地临时传输分片及中断的部分 ZIP 在验证完整原始 ZIP 后清理，不属于科学结果。
