#!/usr/bin/env bash
# =============================================================================
# 01 - 环境搭建：conda 环境 + torch + CUDA 编译链 + 克隆 SplaTAM + 编译光栅化内核
#
# 幂等：重复运行会跳过已完成的步骤。
# 用法： bash 01_setup_env.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_common.sh"

log "===== 步骤 1/8：检查 conda 与 GPU ====="
activate_conda
check_gpu

log "===== 步骤 2/8：创建 conda 环境 '$ENV_NAME' (python $PY_VER) ====="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "环境已存在，跳过创建"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

log "===== 步骤 3/8：安装 PyTorch (cu121) ====="
if python -c "import torch" 2>/dev/null; then
  log "torch 已安装：$(python -c 'import torch;print(torch.__version__)')，跳过"
else
  # shellcheck disable=SC2086
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
fi
# torch 2.1.2 由 NumPy 1.x 编译；若环境里是 NumPy 2.x 会报 "_ARRAY_API not found" 并在运行时崩。
log "锁定 numpy<2 以兼容 torch 2.1.2"
pip install "numpy<2"

log "===== 步骤 4/8：安装 CUDA 编译链 (nvcc 12.1 + 兼容 gcc 12) ====="
# SplaTAM 需从源码编译 diff-gaussian-rasterization → 必须有 nvcc。
# WSL 默认无 nvcc，这里用 conda 把整套工具链装进环境内（无需 sudo，自洽隔离）。
if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
  conda install -y -c "$CUDA_LABEL" cuda-toolkit
else
  log "nvcc 已存在：$("$CONDA_PREFIX/bin/nvcc" --version | tail -1)，跳过"
fi
# 用 conda-forge 装 gcc-11 作 nvcc 宿主编译器，并通过 CUDAHOSTCXX 指定。
# 为什么是 11 而非 12/13：torch 2.1.2 自带的旧版 pybind11，在 gcc≥12 下用 nvcc 编译
# rasterize_points.cu 会报 "expected template-name before '<' token"（pybind11 cast.h）。
# gcc-11 能正确解析该 pybind11，是 3DGS 系 CUDA 扩展编译的稳妥版本。
# 用 conda 显式装 11（若环境里已是别的版本会自动降级），idempotent。
conda install -y -c conda-forge "gxx=11" "gcc=11"

# 导出编译期环境变量（本 shell 内编译光栅化内核时生效）
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
CONDA_GXX="$(ls "$CONDA_PREFIX"/bin/*-g++ 2>/dev/null | head -1 || true)"
if [ -n "$CONDA_GXX" ]; then
  export CUDAHOSTCXX="$CONDA_GXX"
  export CXX="$CONDA_GXX"
  export CC="${CONDA_GXX%g++}gcc"
  log "宿主编译器: $CUDAHOSTCXX"
fi
log "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -2 || die "nvcc 不可用"

log "===== 步骤 5/8：克隆 SplaTAM 仓库 ====="
# SplaTAM 的子模块用 SSH 地址 (git@github.com:...)，但通常没配 SSH 密钥、
# 且 SSH 不走 HTTP 代理 → 会卡死。强制把 SSH 改写为 HTTPS（走代理可达）。
mkdir -p "$WORKSPACE"
if [ -d "$SPLATAM_DIR/.git" ]; then
  [ "$(git -C "$SPLATAM_DIR" rev-parse HEAD)" = "$SPLATAM_REV" ] || \
    die "现有 SplaTAM checkout 与登记版本不同，请为复现指定一个新的 SPLATAM_DIR"
else
  git clone "$SPLATAM_REPO" "$SPLATAM_DIR"
  git -C "$SPLATAM_DIR" checkout --detach "$SPLATAM_REV"
fi
git -C "$SPLATAM_DIR" -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive

log "===== 步骤 6/8：安装 SplaTAM Python 依赖 ====="
# 以仓库自带依赖清单为准（避免硬编码过时的包列表）。torch 已在步骤3锁定。
cd "$SPLATAM_DIR"
# 准备构建依赖（步骤7用 --no-build-isolation 编译光栅化内核时需要这些在环境内）。
# 关键：setuptools 必须 <70 —— torch 2.1.2 的 cpp_extension 仍用 pkg_resources.packaging，
# 新版 setuptools(≥70) 已删除该接口，否则编译报 "No module named 'pkg_resources'"。
pip install "numpy<2" "setuptools<70" wheel ninja
if [ -f requirements.txt ]; then
  log "使用 requirements.txt（剔除其中的 git+ 光栅化内核，改到步骤7单独编译）"
  # requirements.txt 里把光栅化内核写成 git+ 依赖，pip 会用构建隔离编译 → import torch 失败。
  # 这里剔除该行，其余正常装；光栅化内核留到步骤7用 --no-build-isolation 编译。
  grep -vi 'rasteriz' requirements.txt > /tmp/req_no_rast.txt
  pip install -r /tmp/req_no_rast.txt
elif [ -f environment.yml ]; then
  warn "仓库仅提供 environment.yml；将更新环境（注意其可能尝试改动 torch/cuda，如冲突请见 runbook 排错）"
  conda env update -n "$ENV_NAME" -f environment.yml || warn "environment.yml 部分项更新失败，请人工检查"
else
  warn "未找到 requirements.txt / environment.yml，按已知常用依赖兜底安装"
  pip install opencv-python open3d imageio imageio-ffmpeg matplotlib kornia \
              natsort plyfile pyyaml scipy scikit-image tqdm rich lpips wandb cyclonedds
fi
# 兜底：上面装 requirements 时可能把这些关键版本顶掉，编译/运行前统一再锁定一次：
#   numpy<2        —— 匹配 torch 2.1.2 的 numpy1 ABI
#   setuptools<70  —— 保留 pkg_resources.packaging 供 torch 编译用
#   opencv<4.11    —— 4.11+ 强制要求 numpy>=2，会与本环境冲突
pip install "numpy<2" "setuptools<70" "opencv-python==4.10.0.84"

log "===== 步骤 7/8：编译 SplaTAM 的 diff-gaussian-rasterization ====="
# 自动定位光栅化子模块目录（不同版本命名可能不同：含 depth/silhouette 的 fork）
RAST_DIR="$(find "$SPLATAM_DIR" -maxdepth 3 -type d \
            \( -iname '*rasteriz*' -o -iname '*diff-gaussian*' \) \
            -exec test -e '{}/setup.py' ';' -print 2>/dev/null | head -1 || true)"
[ -n "$RAST_DIR" ] || die "未找到光栅化内核源码目录（含 setup.py），请检查 submodule 是否拉取成功"
log "光栅化内核目录: $RAST_DIR"
# 清理上次失败的编译残留（旧 gcc 编出的 .o 缓存会干扰重编）
rm -rf "$RAST_DIR/build"
# 必须 --no-build-isolation：否则 pip 在隔离环境里 import torch 失败（setup.py 依赖 torch）
pip install --no-build-isolation "$RAST_DIR"

# torch-scatter（SplaTAM 依赖，需匹配 torch/cuda 版本）
if ! python -c "import torch_scatter" 2>/dev/null; then
  pip install torch-scatter -f "$SCATTER_INDEX" || \
    warn "torch-scatter 安装失败；若运行时报缺失，请按 runbook 用对应 torch 版本的 wheel 重装"
fi

log "===== 步骤 8/8：环境自检 ====="
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
ok = True
for m in ["diff_gaussian_rasterization"]:
    try:
        __import__(m); print(f"import {m}: OK")
    except Exception as e:
        ok = False; print(f"import {m}: FAIL -> {e}")
try:
    import torch_scatter; print("import torch_scatter: OK")
except Exception as e:
    print(f"import torch_scatter: WARN -> {e}")
assert torch.cuda.is_available(), "CUDA 不可用！"
assert ok, "光栅化内核导入失败"
print(">>> 环境自检通过")
PY

log "✅ 01 环境搭建完成。下一步： bash 02_download_replica.sh"
