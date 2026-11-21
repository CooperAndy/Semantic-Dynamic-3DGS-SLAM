# T19 CPU 验证及接入

依赖：Python 3、CPU PyTorch、NumPy、SciPy。无需 CUDA、DG-SLAM 扩展、网络或 GPU。内附源码引用原作者/前序实现，CHY 只拥有 T19 的改动和诊断工作，不替队友作验收签字。

在本目录执行 CPU 测试，使用包内小型输入快照：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:T19_BASE_ROOT = (Resolve-Path './base_snapshot').Path
python ./scripts/validate_cpu.py
Remove-Item Env:T19_BASE_ROOT
```

测试替身明确隔离 SE3/网络/相关模块/native BA，CUDA 初始化调用会失败。每次测试写入新 `receipts/cpu_cases_*`，避免覆盖先前样例；`CPU_validation.json` 记录最近一次验证。测试重新运行后交付 manifest 需要重新封存，原交付 ZIP 不受影响。

## 未来 GPU runner 的接入片段（本轮不执行）

先将六份 candidate/dg_slam 源码放入独立的新 source/dg_slam 目录，其余依赖沿用已校验版本。使用 T19 独立 runner，完整跟踪后、作者 terminate_woBA 前接入。**去掉 T17 runner 原先覆盖 video.poses 的语句**，否则会在 adapter 前就破坏 native state。

```python
from dg_slam.filler_adapter import configure_filler

# matrices: 当前 mapping 保存的 (N,4,4) CPU NumPy c2w。
# 先保存完整的注入前视频/网络缓冲状态和 RNG，后续每个 case 都恢复。
# output 必须是新 T19 路径；本片段不能直接替换冻结 T17 runner 后原地运行。
configure_filler(
    model, matrices, arm="NO_GS_REFINE", policy="native_final_DROID",
    output=output / "N_NATIVE_FIXED", preserve_keyframes=True,
)
trajectory32 = model.terminate_woBA(data[144:176])
```

ID mapping 锚点对照必须显式传 `policy="mapping_snapshot", mixed_geometry_diagnostic=True`；旧逻辑对照另传 `preserve_keyframes=False`。不同 case 使用新输出目录、同一份恢复后的原生视频状态，并核对哈希；此 adapter 本身不代替状态快照恢复器或完整 runner。

adapter 是实际可调用的候选接入函数，但尚未部署、尚未验证真实 CUDA。ID 的一致几何重求解尚缺，混合诊断标记不能解释为已经解决。

未来有真实日志后，CPU 读取示例：

```powershell
python ./scripts/analyze_trace.py ./YOUR_CASE/trace --output ./trace_steps.csv
```

输出必须是新文件。每轮固定 learned target 残差仅作诊断代理量，不能据此宣布真实轨迹改善。没有真实日志时不要把 receipts 中的 CPU 测试替身样例用于科学结论。
