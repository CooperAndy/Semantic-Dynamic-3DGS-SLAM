# T19：补全接口修复与 CPU 验证

已完成独立候选代码、CPU 控制逻辑测试及后续诊断登记。**没有启动 GPU，没有连接服务器，没有部署或改写 T17/T18。** 当前状态为 `CPU_CANDIDATE_VALIDATED_GPU_NOT_RUN`。

## 已实现的改动

1. **关键帧直接返回固定锚点**。`trajectory_filler.py` 在原来的 16 输入帧批次内识别关键帧，只为非关键帧建立临时节点。输出按原输入顺序重组，精确关键帧时刻不再返回优化过的副本；全关键帧批次无需进入网络补全。显式的 `preserve_keyframes=False` 仅供后续旧逻辑仪表化对照。
2. **异常清理和身份检查**。拒绝重复/逆序/小数帧号、无左锚点及临时缓冲溢出。优化或日志失败时，通过 `finally` 恢复 video 计数和原回调；原锚点被移动则报错终止。保持六轮更新、BA 参数、神经权重、分割掩码和插值公式。
3. **显式的锚点来源契约**。NO_GS 的 `native_final_DROID` 路径保留原 DROID 位姿与几何状态，避免用 mapping 当时的粗位姿覆盖最终位姿。它是新的管线候选，不等同于冻结的 T17 NO_GS，也没有精度通过结论。`mapping_snapshot` 只在显式标记混合几何诊断后允许注入；ID 不能通过改用原 DROID 锚点来悄悄丢掉精修。**一致的 ID 几何衔接仍未实现，现阶段不宣称修复完成。**
4. **逐轮旁路记录**。选中的批次保存临时初值、固定锚点、源视差和特征、learned delta/weight、分割掩码作用后的实际 native 权重、每轮 BA 前后位姿和重投影残差。日志不调整目标、权重或步长，不使用 GT 决定更新。256 MiB 是每个 case 的未压缩负载预留上限，超限明确终止。

## CPU 验证结果

18 组检查通过。六份现有运行的实际 Python 路由均为 **24 个固定锚点 + 176 个待补全帧**，保留原 16 输入帧边界；40 个随机窗口覆盖交错锚点、全锚点批次、单锚点及尾批。还执行了候选的实际 `__call__` / `__fill` 控制流程、六轮模拟优化、优化异常/日志失败清理和临时缓冲溢出检查。

用相同 CPU 后端替身分别执行原版和候选的 FactorGraph.update / DepthVideo.ba，比较传给 native BA 的 target、有效权重、边和优化参数以及返回状态，一致。日志读取器对已知位移的 CPU 样例正确计算 c2w 步长与固定 learned-target 残差，并拒绝把不完整 mock 日志当作真实网络诊断。

测试使用真实 CPU PyTorch 张量，但 **SE(3)、网络/相关模块和 native BA 使用测试替身**。CUDA lazy initialization 被主动阻止，最终 `CUDA_initialized=false`。因此这里只证明 Python 接口控制和日志链路，不证明实际 CUDA 兼容、真实网络更新行为或位姿精度改善。没有重跑 T17，也没有给候选生成伪造的 ATE/RPE。

详见 `receipts/CPU_validation.json`。内附原始小型测试输入及逐文件哈希，可在 CPU 独立复现。

## 逐轮日志如何读

`scripts/analyze_trace.py` 只使用 NumPy/SciPy，读取每轮四个真实阶段：graph_before_ba、native_before_ba、native_after_ba、graph_after_ba。输出最大 c2w 平移/旋转步长、有效权重覆盖率、learned delta 最大值，以及本轮固定 learned target / 有效权重下的前后残差。

这个残差是诊断代理量，不是完整 native BA 目标；不同轮的 target/weight 会变化，不能把跨轮下降直接当作同一目标下降，更不能替代 GT 轨迹精度。CPU 测试样例日志只证明记录器和读取器，不是真实神经补全结果。

## 下一轮已登记，尚未运行

`diagnostic_registration.json` 将未来 GPU 工作限定为已曝光 DEV 的机制诊断：选择已知异常种子 2718；需要分别重建 NO_GS 与 ID 的完整 0–199 前端状态，再只补全 **144–175**，覆盖异常附近原来的两个 16 帧批次。T17 未保存完整视差/特征/上下文，不能仅凭已有位姿数组重演真实补全。

五个诊断 case 为 NO_GS mapping 锚点的原逻辑/保留关键帧逻辑、NO_GS 最终 DROID 锚点的保留逻辑，以及 ID mapping 锚点的原逻辑/保留逻辑。每个分支都必须恢复同一份注入前视频状态和 RNG，核对活跃位姿、视差、特征、上下文、掩码与内参哈希，不能接着上一分支已改动的状态运行。

登记已冻结窗口、种子、case、状态控制、日志上限与工程门槛；创建时间在 CPU 测试和 T18 分析之后、任何 T19 GPU 实验之前，明确不是盲测预注册。GT 只允许在原始输出封存后离线评分。此窗口是按已知误差选出的诊断窗口，任何改善都不能当作正式采用通过。

## 文件与接入方式

- `candidate/dg_slam/`：三个改动文件与三个新增模块，共六份候选源码。
- `patches/T19_vs_T17.patch`：相对冻结 T17 源码的完整差异。
- `base_snapshot/`：原三个源码与六份关键帧测试输入，不含 GT。
- `scripts/validate_cpu.py`：CPU 复现测试。
- `scripts/analyze_trace.py`：未来逐轮日志的 CPU 读取器。
- `receipts/historical_integrity.json`：冻结 T17/T18 的完整逐文件核验。
- [README_CPU与接入.md](README_CPU与接入.md)：CPU 命令与未来独立 runner 的接入片段。

候选须应用到新工作目录，使用新的 runner、输出路径和登记文件；不能覆盖冻结目录或复用 T17 采用标签。T14 FULL 不采用、T15 原 FAIL、T16 负面结果、T17 两者未资格通过的状态均保留。
