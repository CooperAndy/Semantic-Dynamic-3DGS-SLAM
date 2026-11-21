#!/usr/bin/env bash
# Conservative fixed-source build. Kept separate from the preserved low-memory trial.
set -euo pipefail
R=/root/autodl-tmp/dg_slam_chy_20261011
LIMIT=max
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [[ -f "$f" ]]; then LIMIT=$(cat "$f"); break; fi
done
if [[ "$LIMIT" != max ]] && (( LIMIT <= 2147483648 )); then
  echo 'STOP: 2GB build trials are preserved. Use a larger-memory environment before compiling.' >&2
  exit 2
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$R/01_source_env/venv_dg/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${DG_TARGET_ARCH:-8.6} MAX_JOBS=${DG_BUILD_JOBS:-4} CMAKE_BUILD_PARALLEL_LEVEL=${DG_BUILD_JOBS:-4}
[[ "$TORCH_CUDA_ARCH_LIST" == 8.6 ]] || { echo 'This validated preset requires sm_86; update wheel/build validation for other GPUs.' >&2; exit 2; }
export DG_REPORT_DIR="$R/05_reports/3080ti_prepare"
export CUDA_VISIBLE_DEVICES=""
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
python -m pip install 'scikit-image==0.24.0' 'numpy==1.26.4'
python - <<'PY'
from pathlib import Path
import hashlib
p=next(Path('/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/dependencies/wheels').glob('faiss_gpu_cu12*.whl'))
assert hashlib.sha256(p.read_bytes()).hexdigest()=='8bf6f8d667a5de94b0ddef76ccb54b6b5ee06a21655804924d4f91fde4ca26fa'
PY
python -m pip install --no-deps "$R/01_source_env/dependencies/wheels/faiss_gpu_cu12-1.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
D="$R/01_source_env/dependencies/DROID-SLAM_runtime"
test -f "$D/thirdparty/lietorch/eigen/Eigen/Core"
python -m pip install --no-build-isolation --no-deps "$D/thirdparty/lietorch"
python -m pip install --no-build-isolation --no-deps "$R/01_source_env/dependencies/DROID-backend_dg_api"
python -m pip install --no-build-isolation --no-deps "$R/01_source_env/DG-SLAM/submodules/diff-gaussian-rasterization_pose"
python -c 'import faiss; assert faiss.__version__=="1.10.0" and hasattr(faiss,"StandardGpuResources")'
python "$R/scripts/cpu_checks.py" native
echo NATIVE_IMPORTS_AND_CPU_CHECKS_PASS
