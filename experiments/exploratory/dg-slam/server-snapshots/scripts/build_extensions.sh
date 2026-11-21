#!/usr/bin/env bash
# EXPERIMENTAL LOW-MEMORY TRIAL ONLY. Preserved by user request; not a verified runtime build.
# Use the separate clean source candidate and build_runtime_extensions.sh after obtaining more RAM.
set -euo pipefail
echo 'STOP: low-memory trial retained for diagnosis; do not resume automatically.' >&2
exit 2
R=/root/autodl-tmp/dg_slam_chy_20261011
export CUDA_HOME=/usr/local/cuda
export PATH="$R/01_source_env/venv_dg/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export CUDA_VISIBLE_DEVICES=""
C="$R/01_source_env/dependencies/clang14/root"
export CC="$C/usr/bin/clang-14" CXX="$C/usr/bin/clang++-14" CUDAHOSTCXX="$C/usr/bin/clang++-14"
export LD_LIBRARY_PATH="$C/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
D="$R/01_source_env/dependencies/DROID-SLAM"
python -m pip install --no-cache-dir swig==4.3.0
test -f "$D/thirdparty/lietorch/eigen/Eigen/Core"
python "$R/scripts/standalone_torch_build.py" "$D/thirdparty/lietorch" --probe-gpu-first
python "$R/scripts/verify_runtime_wheels.py"
python -m pip install --no-deps "$R/01_source_env/dependencies/wheels/torch_scatter-2.1.2+pt28cu128-cp312-cp312-linux_x86_64.whl"
python "$R/scripts/standalone_torch_build.py" "$D"
python "$R/scripts/standalone_torch_build.py" "$R/01_source_env/DG-SLAM/submodules/diff-gaussian-rasterization_pose"
echo NATIVE_EXTENSIONS_BUILT
F="$R/01_source_env/dependencies/faiss"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
cmake -S "$F" -B "$F/build" -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_C_API=OFF -DBUILD_TESTING=OFF -DFAISS_OPT_LEVEL=generic -DCMAKE_CUDA_ARCHITECTURES=120 -DCUDAToolkit_ROOT="$CUDA_HOME" -DPython_EXECUTABLE="$(command -v python)" -DPython3_EXECUTABLE="$(command -v python)" -DSWIG_EXECUTABLE="$(command -v swig)" -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Generic -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3 -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so.3
cmake --build "$F/build" --target swigfaiss --parallel 1
(cd "$F/build/faiss/python" && python -m pip install --no-build-isolation --no-deps .)
echo ALL_EXTENSIONS_BUILT
