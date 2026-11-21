#!/usr/bin/env bash
set -u
R=/root/autodl-tmp/dg_slam_chy_20261011
P="$R/05_reports/t01_pose_ablation_20261011"
export PATH="$R/01_source_env/venv_dg/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
test ! -e "$P/off.run_name" || exit 30
available=$(df -Pk "$R" | awk 'NR==2 {print $4}')
test "$available" -gt 5242880 || exit 31
stamp=$(date +%Y%m%d_%H%M%S)
off="T01_200_OFF_${stamp}"
on="T01_200_ON_${stamp}"
printf '%s\n' "$off" > "$P/off.run_name"
printf '%s\n' "$on" > "$P/on.run_name"
python "$R/scripts/t01/run_t01.py" --mode OFF --run-name "$off" > "$P/off.log" 2>&1
code=$?
printf '%s\n' "$code" > "$P/off.exit"
test "$code" -eq 0 || exit "$code"
python "$R/scripts/t01/run_t01.py" --mode ON --run-name "$on" > "$P/on.log" 2>&1
code=$?
printf '%s\n' "$code" > "$P/on.exit"
exit "$code"
