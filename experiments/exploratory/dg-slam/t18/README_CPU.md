# T18 CPU 复现

本包由 CHY 接手分析工作独立生成。T17 的轨迹与源码仅作为历史输入保留，不改写队友产物或冻结结果。诊断只覆盖六次 DEV200，不能外推完整 TUM 序列或未曝光测试。

依赖：Python 3、NumPy、SciPy、Matplotlib。无需 Torch、CUDA、网络或 GPU。环境版本见 `receipts/independent_CPU_validation.json`。

默认 `scripts/analyze_filler.py` 读取同级原 T17 目录并核验其 281 文件 manifest。独立解压本包后，可以使用内附小型快照：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:T18_INPUT_ROOT = (Resolve-Path './inputs_snapshot').Path
python ./scripts/analyze_filler.py
Remove-Item Env:T18_INPUT_ROOT
```

快照模式核验 `receipts/input_snapshot_manifest.json`，不是声称快照包含原 T17 的所有交付文件。内附原 `delivery_manifest.json` 仅用于核对来源及冻结指纹。`audit_and_report.py` 另外需要同级 T17 与 T14 原始包，以便重新获取历史源码；本包已附其核验结果、报告、图和输入，不运行该脚本也能重算数值。

`source_reference/` 与 `inputs_snapshot/source/`、快照中的原 run_tracking.py 含 GPU 代码，只供文本核对，不要执行。CPU 复现只执行 T18 自有分析脚本。

“插值初值”是 float64 数学反事实，不是保存的原 GPU 中间位姿；“恢复 KF”也是离线诊断反事实。二者都不是新的通过实验。固定对齐取自各次运行最终 DROID 的 24 个锚点，与各阶段独立对齐的 ATE 分开计分。
