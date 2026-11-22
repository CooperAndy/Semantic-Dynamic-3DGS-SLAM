# S23 复现手册

| 任务 | 分工 | 固定窗口与验收 |
| --- | --- | --- |
| S23-INPUT | B 制作双目深度与位姿，A 制作实例语义和动静状态 | KITTI08 源帧 3000–3999，KITTI05/10 源帧 0–999。每窗 1,000 对双目、1,000 条 P2 米制位姿和 1,000 份语义观测。 |
| S23-E2 | A/B | KITTI08 种子 0，B/T/M/TM/D/TD/MD/TMD 按注册次序运行。 |
| S23-E3 | A/B | KITTI05/10 各用种子 0、1、2，每个单元 B/TM/D/TMD。B 生成同单元共用的回放调度与出生候选。 |
| S23-EVAL | A 负责地图和语义，B 负责轨迹 | 每组 57,000 次更新、零在线 GT 访问与零留出帧建图。20 张留出图各用三组相机渲染，共 60 次，另做全窗 SE(3) ATE/RPE。 |

原始数据、配对差异及采用阈值见 [结果目录](report/) 和 [冻结定义](formal_method_freeze.json)。正式定位目标为窗等权 ATE 改善 ≥10%、平移 RPE 改善 ≥5%，单窗退化上限为 5%。地图条件见各组注册中的 limits。历史 32 组已完成，联合质量采用门未通过。

## 安装

使用 Linux、Python 3.12、CUDA 12.8 工具链及相应的 GPU 驱动。从仓库根目录执行。

~~~bash
study=experiments/online-semantic-dynamic
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$study/requirements-reproduction.txt"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.6
python "$study/build_rasterizer.py"
~~~

S23 正式扩展使用 sm_86。RTX 5090 的架构设为 12.0。构建脚本从固定上游归档提取 GLM 头文件，核验 CUDA、C++、头文件及 Python 接口的 SHA-256，再编译仓库里的 FP64 累加分支。用 `--upstream-archive` 提供离线 ZIP，或用 `--source-only` 只核验源码。每个运行单元记下本次编译的二进制身份。

## 准备输入

将 KITTI Odometry 的彩色双目和标定解压到 `KITTI_ROOT/sequences/NN/`，官方轨迹解压到 `KITTI_ROOT/poses/`。评分还要用 `KITTI_ROOT/sequences/NN/velodyne/` 和 `LABELS_ROOT/sequences/NN/labels/`。`WORK` 是仓库之外的独立工作目录。`ORB_POSES` 指向团队独立归档的该窗口 P2 ORB 即时估计，也可用同一前端协议生成。输入为 1,000 行 KITTI 12 列米制位姿，以窗口首帧为锚。官方位姿、点云与点标签只用于评分。 [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) · [SemanticKITTI](https://semantic-kitti.org/dataset.html) · [因果 ORB 前端协议](../../preprocessing/server-snapshots/orbslam3-causal/PROTOCOL.md)

**重新生成预测缓存。** 使用 RAFT-Stereo 的 Middlebury 权重、Cityscapes 语义模型 `91281d04`、实例模型 `4c0fb9b2`，以及 TorchVision RAFT-Large 的默认权重。模型文件身份记录在 [阶段 2 锁](stage2_frozen_sources.json) 和 [双目锁](../../preprocessing/server-snapshots/causal-stereo/registration_cpu.json)。实例模型使用 Transformers 4.57.6 的键布局。从原始 Hugging Face 权重重建时，可用 [转换脚本](../../preprocessing/tsdd-s23/scripts/convert_mask2former_checkpoint.py)，并传 `--converted-instance` 记录新序列化文件的哈希。离线机器需预置 TorchVision RAFT 默认权重和评估用 AlexNet 权重。 [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) · [Mask2Former 语义模型](https://huggingface.co/facebook/mask2former-swin-small-cityscapes-semantic/tree/91281d04) · [Mask2Former 实例模型](https://huggingface.co/facebook/mask2former-swin-small-cityscapes-instance/tree/4c0fb9b2)

~~~bash
python "$study/reproduce.py" inputs --workdir "$WORK" --sequence 05 \
  --kitti-root "$KITTI_ROOT" --orb-poses "$ORB_POSES" --raft-checkpoint "$RAFT_STEREO_WEIGHTS" \
  --semantic-model "$SEMANTIC_MODEL" --instance-model "$INSTANCE_MODEL"
~~~

P1 运动证据读取原始双目深度。建图和跟踪提案使用 G19C 因果置信度深度。KITTI08 再生成沿用同一因果公式和权重。需要输入逐字节一致时，使用缓存恢复入口。

**恢复已封存的预测缓存。** `CACHE` 下应有 `cache/depth_confhard/`、`cache/confidence/` 和 `observations/`，文件按窗口内帧编号，例如 `000000_depth.npy`、`000000_confidence.npy` 和 `000000.npz`。B 组科学包中的 `input_provenance/observation_cache/` 提供观测 NPZ。深度及置信度来自团队对应窗口的输入归档。05/10 的正式清单随仓库发布。恢复 08 时，通过 `--runtime-lock` 和 `--observation-lock` 提供原始科学包的两份清单。

~~~bash
python "$study/reproduce.py" inputs --workdir "$WORK" --sequence 05 \
  --kitti-root "$KITTI_ROOT" --orb-poses "$ORB_POSES" --cache "$CACHE"
python "$study/reproduce.py" check --workdir "$WORK" --sequence 05 --verify-hashes
~~~

## 执行与评分

~~~bash
python "$study/reproduce.py" run --workdir "$WORK" --sequence 05 --seed 0 \
  --group all --evaluate --kitti-root "$KITTI_ROOT" --labels-root "$LABELS_ROOT"
~~~

命令按注册次序执行一个单元。单组可用 `--group B` 等指定，后续组读取该单元的 B 组封存结果。也可先建图，再用 `evaluate` 和相同的 `--sequence`、`--seed`、`--group` 和数据路径评分。每组保存数值审计、输入哈希、渲染和轨迹评分。单组起跑至少需要 23 GiB 空闲磁盘，运行保留 8 GiB，单组预算 9,000 秒。完整矩阵保留地图、评分文件与源码快照需额外空间。已有组及日志保持原样，失败后可用新的工作目录复核。

完成三个窗口的全部注册组和评分后生成结果表。

~~~bash
S23_WORKDIR="$WORK" python "$study/create_summary.py"
~~~

| 风险 | 检查 |
| --- | --- |
| 输入缓存或帧序错配 | 运行 `check --verify-hashes`，核对左右图、深度、置信度、观测与 ORB 轨迹。 |
| CUDA ABI 不匹配 | 用执行建图的同一 Python 与 PyTorch 构建，按实际显卡指定架构。 |
| GPU 变化或权重重新转换 | 记录本次输入与构建哈希，用原始表格核对变化。 |
