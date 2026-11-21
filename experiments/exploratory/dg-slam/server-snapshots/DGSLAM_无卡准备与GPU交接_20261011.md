# DG-SLAM：无卡准备与 GPU 交接

更新：2026-10-11。范围：TUM walking_xyz 官方参考的工程准备。研究主线仍是利用高斯提高 tracking / mapping；本轮不恢复旧 06 的类别／运动消融。

> **当前状态更新：**用户已更换3080 Ti并启动GPU，原生扩展和兼容性验收已完成。最新命令与证据见[3080 Ti环境修复与验收](DGSLAM_3080Ti_环境修复与验收_20261011.md)。下文保留14:21无卡阶段的历史快照。

> 无卡资料、模型与入口准备已收尾，数据／模型／scatter 的 CPU 检查通过。**原生扩展尚未编译完成，环境未达到可运行 DG-SLAM 的验收标准。**无卡规格固定为 2GB RAM / 0.5 核，用户已要求保留编译现场；已停止进一步编译。GPU 未启动，预测掩码与 SLAM 均为 0。最终回执见 `05_reports/cpu_prepare/final_cpu_preparation.json`。

## 1. 目录与环境

服务器根目录：`/root/autodl-tmp/dg_slam_chy_20261011`。以下所有相对路径均相对此目录。

| 路径 | 内容 |
|---|---|
| `01_source_env/DG-SLAM` | 固定 DG-SLAM 原始提交的完整文件快照与单独记录的兼容补丁 |
| `01_source_env/dependencies` | 固定提交的依赖源码、私有 Clang；DROID-SLAM 为保留的编译试验，DROID-SLAM_runtime 为独立源码候选 |
| `01_source_env/venv_dg` | 本轮轻量隔离环境；只读复用共享 base 的 torch / CUDA Python 依赖，新增与降级包写入本环境 |
| `01_source_env/weights` | DROID、OneFormer、ADE20K 类别说明 |
| `01_source_env/patches` | 兼容改动 diff；不覆盖已有 SplaTAM / MonoGS / 阶段结果 |
| `scripts` | 数据接口、CPU 验收、预测掩码、30/200 帧运行、GPU 启动入口 |
| `05_reports/cpu_prepare` | 安装／编译日志、CPU 检查与锁文件 |

本环境不是全依赖独立副本：共享 torch 为 `2.8.0+cu128`，torchvision 为 `0.23.0+cu128`；DG 环境使用 `numpy==1.26.4`。启动脚本检查关键版本变化，变化后须重验／重编译。没有对共享 base 执行安装或降级。

AutoDL 无卡容器实际 cgroup 内存上限 **2GB**、CPU 配额 **0.5 核**，不能把主机 meminfo / nproc 当成可用额度。本轮曾尝试单进程、Clang、精简头文件和拆分编译；19 个 CPU 运算单元曾通过编译，但 CUDA 编译继续被终止，后续头文件分离试验还存在编译错误。**这些试验没有形成通过验收的扩展，不能视作已验证的兼容移植。**试验源码、生成文件与日志全部保留，旧 `build_extensions.sh` 已加停止保护。

另外从固定 git 提交导出 `dependencies/DROID-SLAM_runtime`，仅移除 lietorch 旧架构硬编码，交由环境变量指定 `sm_120`；没有应用拆分试验。它和 `build_runtime_extensions.sh` 只是未来构建候选，尚未编译验证。新入口在当前 2GB 限额下会直接退出。编译日志出现进程被杀，但当前 `memory.events` 的 `oom_kill=0`，不能断言已证实为内核 OOM；具体终止来源尚未定位。

## 2. 源码与模型来源

| 项目 | 固定版本／来源 |
|---|---|
| [DG-SLAM](https://github.com/fudan-zvg/DG-SLAM) | `fb28a20caa74f376ebd4a5f3c473cd57e7f5d9b4`，91 个原始跟踪文件的哈希已有记录 |
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99`；当前扩展源码提供 torch 2.x 兼容基础，DG Python 前端保持原版本 |
| lietorch / Eigen / scatter | 子模块提交见 `01_source_env/dependency_commits.json` |
| scatter 已安装版本 | PyG 官方 `torch_scatter==2.1.2+pt28cu128` / cp312 Linux wheel，SHA256 `372005c58c8ab7a96c8e971abe008bfbe5933f5c526d5d3419a48482381d1d47`；cuobjdump 已检查 CUDA kernel 库含 PTX，可向前 JIT；GPU 调用尚未验证 |
| [FAISS](https://github.com/facebookresearch/faiss/tree/v1.10.0) | `v1.10.0` / `189a9d4461233de798c2eb17fb6d354ceda542e6`，源码已准备，后续需为 CUDA 12.8 / sm_120 构建；尚未编译 |
| GLM | 从现有工程复制头文件，原提交 `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110`；1529 文件复制前后哈希逐项一致 |
| DROID 权重 | [DROID 官方 README 下载链接](https://github.com/princeton-vl/DROID-SLAM#demos)，16,061,701 字节，SHA256 `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526` |
| [OneFormer ADE20K Swin-L](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) | revision `4a5bac8e64f82681a12db2e151a4c2f4ce6092b2`；仅下载 Transformers 版本，避免重复下载原 Detectron2 权重 |
| OneFormer 权重 | `pytorch_model.bin`，879,517,517 字节；SHA256 `c0b2fe11dfecee6f2f1f315f466946e96f4e94813f3f6d660ff3747b83c28cc9`，与该模型版本 LFS 对象哈希一致 |
| ADE20K 类别说明 | `shi-labs/oneformer_demo` dataset revision `4d683bd5bf84e9c8b5537dce306230bde409fe89` |

DG 作者只说明使用 OneFormer，未明确发布分割模型、检查点与预处理细节。本轮选择 Swin-L ADE20K，已核对 **类别 12 = person**；应报告这一选择，不能声称预测掩码与作者原实验逐像素相同。

额外核查：下载检查过 `faiss-gpu-cu12==1.10.0` wheel，其 GPU 动态库只有旧架构 cubin、无 PTX，**未安装**，不能用于 5090。scatter 的 PTX 与该情况不同；普通 PTX 的向前兼容依据为 [NVIDIA Blackwell 兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)。这些均不代替实际 GPU 验证。

兼容补丁包括：移除从未调用的 `simple_knn.distCUDA2` 导入（所有实际邻域搜索仍使用 GPU FAISS）；解除 lietorch 的旧 GPU 架构硬编码；精简 torch 头文件并显式补齐所需头；更新已移除的 THC 头。补丁不将 DG 专用光栅器替换成 SplaTAM 光栅器。

## 3. 已通过的数据检查与入口

CPU 检查通过作者 `run_tum.py` 中原始关联／选帧函数独立执行，得到 **828 帧**。30/200 窗口的 RGB、depth 与 GT 关联逐项匹配。200 对 RGB／depth 已实际解码；深度为 `uint16`、640×480、5000 单位／米，有效深度比例约 46.35%–75.28%。本次没有重复解码其余全部图像；全序列完整性沿用原下载回执。

- `tum_window.py`：显式读取已有 30/200 CSV，核对是作者索引前缀；保持作者 RGB、深度、内参和二值掩码预处理。
- `run_window.py`：构造模型前设置 `n_img`；输出独立目录；记录输入帧数、DROID 关键帧数、GS tracking / mapping 调用、GS 位姿优化与渲染次数。
- `generate_masks.py`：真正推理后保存原分辨率 ADE20K 类别 ID PNG，并登记模型、RGB 和 mask 哈希；不创建空掩码。缺失或哈希不一致时停止。
- `launch_gpu.sh`：先检查 CPU 验收与关键版本，再检查 GPU；首次 smoke 包括 30 帧掩码与 30 帧 SLAM，总时限 20 分钟。

R0 保留作者先提取 person 二值图、8×8 膨胀、再用默认线性插值缩放并转 uint8 的方式。此处**不缩放原始类别 ID 图**。若后续改为最近邻二值图缩放，应另记预处理变体。

## 4. 后续顺序与命令，当前不执行

先解决编译资源。GPU 本身不是编译所必需的；如果 AutoDL 只有有卡模式才提供足够 CPU／内存，则下一次开机需要把扩展构建作为前置步骤，不能开机后直接跑 SLAM。本次没有启动或调整实例。

在内存更充足、路径与版本栈一致的环境中，先尝试以下候选入口，并处理实际编译报错。具体内存需求尚未测定，不能保证一次构建成功：

```bash
R=/root/autodl-tmp/dg_slam_chy_20261011
set -o pipefail
bash "$R/scripts/build_runtime_extensions.sh" 2>&1 | tee "$R/05_reports/cpu_prepare/build_runtime.log"
```

只有末尾原生导入／CPU 算子检查生成 `cpu_check_native.json` 的 PASS 回执后，才在有卡模式执行：

```bash
bash "$R/scripts/launch_gpu.sh" smoke
```

`launch_gpu.sh` 不会自动编译；当前缺少 native PASS 回执，会拒绝运行。20 分钟时限仅针对 GPU smoke，不含未来构建时间。

仅当 30 帧工程 smoke 通过，或确认 30 帧只完成 DROID 预热后，才决定是否进入 200 帧；不会自动扩跑：

```bash
bash "$R/scripts/launch_gpu.sh" masks200
bash "$R/scripts/launch_gpu.sh" slam200
```

本轮入口只支持 30/200 帧，不冒充已准备全 828 帧正式实验。完整序列入口在 DEV 通过后再扩展。所有结果写入新时间戳目录，已有结果不覆盖。

## 5. 验收与边界

| ID | 责任建议 | 验收标准 | 当前状态 |
|---|---|---|---|
| REF-01 | B 主、A 配合 | 完整固定源码、原生扩展导入、DROID 权重严格匹配 | 源码与权重就绪；DROID 参数严格匹配通过；原生编译／导入未完成 |
| REF-02 | B 主 | 作者选帧吻合；200 对图像解码；窗口构造前设长度；独立输出与计数 | CPU 数据检查通过，GPU 调用未执行 |
| REF-03a | A 主 | 模型与类别映射固定、处理器 CPU 检查、模型参数严格匹配 | CPU 断网处理器与 OneFormer 参数严格匹配通过；未推理 |
| REF-03b | A 主 | 真实预测 30/200 张语义图，尺寸／标签／来源一致 | GPU 待执行 |
| REF-04 | A/B | 无 CUDA 错误；GS 渲染、建图与位姿优化计数均大于 0；有限轨迹与地图输出 | GPU 待执行 |
| REF-05 | A/B | 200 帧 DEV 后再确定完整序列、GT-free 与机制对照 | 待执行 |

CPU 回执：`inputs`（828 作者索引、30/200 前缀、200 对解码）、`models`（OneFormer 断网处理器与参数匹配）、`droid_weights`（102 个参数张量严格匹配）、`scatter`（CPU 前向／反向）均为 PASS，CUDA 初始化均为 false。这些检查没有执行 OneFormer／DROID 网络前向，不能代替原生算子或 GPU 验证。

保留现场：`dependencies/DROID-SLAM`、`05_reports/cpu_prepare/build*.log`、`cuda_header_probe.log`、`preserved_*` 哈希／diff、`01_source_env/patches`。原生扩展仍待验收：lietorch、DROID backend、DG 专用 rasterizer、GPU FAISS。

R0 明确保留作者的 GT 辅助初始化、早期位姿注入及 `scale=True` 评价，不能作为 GT-free 系统或 KITTI 度量跟踪改善证据。30 帧读完、残差下降、接口导入均不等于 tracking/mapping 质量提升。

| 风险 | 停止／退化条件 | 应对 |
|---|---|---|
| 无卡容器内存有限 | 编译被终止；CPU 参数检查超过 2GB | 规格固定，按用户要求停止重试并保留现场；后续资源充足时再构建，未承诺构建成功 |
| GPU 运行尚未验证 | CUDA 符号、架构或算子行为错误 | 20 分钟 smoke 内定位；失败不扩跑 200/828 |
| 短窗口关键帧不足 | GS 位姿优化或渲染计数为 0 | 标为未覆盖，保留作者 warmup，再评估 200 帧 |
| 磁盘余量 | 剩余 <5GiB | 停止新增大文件，先清点本轮可恢复缓存；不删除旧实验 |
| 掩码模型选择不完全可追溯作者 | 作者没有发布精确检查点 | 报告本轮版本，不声称完全相同的论文复现 |

最终回执：2026-10-11 14:21（服务器北京时间），数据盘剩余约 15.75GiB；原有 14 份源码／配置／manifest 哈希与准备前一致，另 1 个原本不存在的路径仍不存在。CPU 检查、编译日志与试验 diff 已同步到本地 `DGSLAM_方案核查_20261011/cpu_receipts`。

角色分工是建议，尚未替队友接受任务。Stage2 正式 FAIL、阶段3 00–05 结果与旧06 GPU 未执行状态均不改变。
