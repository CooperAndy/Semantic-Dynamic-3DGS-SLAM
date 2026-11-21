#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/t04_chy_kitti00_200_20261012
export PATH=/root/autodl-tmp/dg_slam_chy_20261011/01_source_env/venv_dg/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$R"
python - <<'PY'
import subprocess,shutil,json
from pathlib import Path
r=Path('/root/autodl-tmp/t04_chy_kitti00_200_20261012')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Shared GPU occupied'
assert shutil.disk_usage(r).free>5*1024**3
assert json.loads((r/'projection_rasterizer_audit.json').read_text())['status']=='PASS'
PY
for mode in off on; do
  name="KITTI00_200_V2_${mode}_$(date -u +%Y%m%d_%H%M%S)"
  printf '%s\n' "$name" > "$R/v2_${mode}.run_name"
  set +e
  timeout --signal=TERM --kill-after=30s 30m python "$R/scripts/run_kitti_v2.py" --run-name "$name" --gs-refine "$mode" > "$R/v2_${mode}.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$R/v2_${mode}.exit"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
