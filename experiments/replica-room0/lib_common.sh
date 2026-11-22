#!/usr/bin/env bash
# =============================================================================
# T0.3 SplaTAM 复现 - 公共配置与函数库
# 被 01_setup_env.sh / 02_download_replica.sh / 03_run_replica.sh /
# 04_eval.sh / run_all.sh 共同 source。
#
# 所有可调参数集中在此处；如需改路径/场景/版本，只改这一个文件。
# =============================================================================

# ---- 可调参数 ---------------------------------------------------------------
# 项目工作区（与 阶段0_技术执行文档.md 2.3 节约定一致）
WORKSPACE="${WORKSPACE:-$HOME/workspace/slam_project}"
# SplaTAM 仓库目录
SPLATAM_DIR="${SPLATAM_DIR:-$WORKSPACE/SplaTAM}"
# SplaTAM 上游仓库
SPLATAM_REPO="${SPLATAM_REPO:-https://github.com/spla-tam/SplaTAM.git}"
SPLATAM_REV="${SPLATAM_REV:-da6bbcd24c248dc884ac7f49d62e91b841b26ccc}"
# 专用 conda 环境名（必须独立，SplaTAM 的光栅化内核与 MonoGS 版本不同）
ENV_NAME="${ENV_NAME:-splatam}"
PY_VER="${PY_VER:-3.10}"

# 计算栈版本（与 技术执行文档 一致：CUDA 12.1 / torch 2.1.2）
CUDA_LABEL="${CUDA_LABEL:-nvidia/label/cuda-12.1.0}"   # 提供可编译用的 nvcc 12.1
TORCH_SPEC="${TORCH_SPEC:-torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
SCATTER_INDEX="${SCATTER_INDEX:-https://data.pyg.org/whl/torch-2.1.0+cu121.html}"

# RTX 4060 = Ada Lovelace, sm_89。锁定架构可加速编译、避免多架构编译失败。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

# 复现场景与数据/日志路径
SCENE="${SCENE:-room0}"
DATA_DIR="${DATA_DIR:-$SPLATAM_DIR/data/Replica}"   # SplaTAM 默认数据位置
LOG_DIR="${LOG_DIR:-$WORKSPACE/logs/t0.3}"

# ---- 日志辅助 ---------------------------------------------------------------
log()  { echo -e "\033[1;32m[T0.3]\033[0m $*"; }
warn() { echo -e "\033[1;33m[T0.3][WARN]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[T0.3][ERR]\033[0m $*" >&2; exit 1; }

# ---- conda 激活（兼容非交互式 bash，~/.bashrc 未加载的情况）------------------
activate_conda() {
  local conda_sh=""
  local c
  for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && conda_sh="$c" && break
  done
  if [ -z "$conda_sh" ] && command -v conda >/dev/null 2>&1; then
    conda_sh="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
  fi
  [ -f "$conda_sh" ] || die "找不到 conda.sh，请确认已安装 Miniconda（参考技术执行文档 2.1 步骤3）"
  # shellcheck disable=SC1090
  source "$conda_sh"
}

# 激活本任务专用环境
activate_env() {
  activate_conda
  conda activate "$ENV_NAME" 2>/dev/null \
    || die "conda 环境 '$ENV_NAME' 不存在，请先运行 01_setup_env.sh"
}

# GPU 自检（非致命，仅提示）
check_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  else
    warn "未检测到 nvidia-smi，请确认 WSL 能访问 GPU"
  fi
}
