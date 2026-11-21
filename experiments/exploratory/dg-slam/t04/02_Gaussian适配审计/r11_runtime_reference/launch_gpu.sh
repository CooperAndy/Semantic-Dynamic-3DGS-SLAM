#!/usr/bin/env bash
# Prepared only. Explicit invocation AFTER GPU allocation is required.
set -euo pipefail
R=/root/autodl-tmp/dg_slam_chy_20261011
export PATH="$R/01_source_env/venv_dg/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 WANDB_MODE=disabled
export DG_REPORT_DIR="$R/05_reports/3080ti_prepare"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
python - <<'PY'
from pathlib import Path
import json
import os
import numpy as np
import torch
assert torch.__version__ == '2.8.0+cu128' and np.__version__ == '1.26.4', 'Runtime changed: repeat CPU audit / rebuild'
for stage in ['inputs','native','models']:
    r=json.loads((Path(os.environ['DG_REPORT_DIR'])/f'cpu_check_{stage}.json').read_text())
    assert r['status']=='PASS', f'CPU check not passed: {stage}'
    if stage=='native': assert r['target_arch']=='8.6', 'Native extensions were not accepted for sm_86'
sys_path = '/root/autodl-tmp/dg_slam_chy_20261011/scripts'
__import__('sys').path.insert(0, sys_path)
from tum_window import ROOT, MODEL, ONEFORMER_SHA256, sha256
assert sha256(ROOT/'01_source_env/weights/droid.pth')=='46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526'
assert sha256(MODEL/'pytorch_model.bin')==ONEFORMER_SHA256
assert torch.cuda.is_available(), 'No GPU allocated; stopped before mask generation or SLAM'
assert torch.cuda.get_device_capability()==(8,6), 'GPU architecture changed: rebuild and verify first'
PY
mode=${1:?Usage: bash launch_gpu.sh smoke|masks30|masks200|slam30|slam200}
case "$mode" in
  smoke)
    timeout --signal=TERM --kill-after=30s 20m bash -c 'set -e; R=/root/autodl-tmp/dg_slam_chy_20261011; python "$R/scripts/gpu_sanity.py"; python "$R/scripts/generate_masks.py" --frames 30; python "$R/scripts/run_window.py" --frames 30 --run-name "R0_30_$(date +%Y%m%d_%H%M%S)"' 2>&1 | tee "$R/03_gpu_smoke/smoke_$(date +%Y%m%d_%H%M%S).log"
    ;;
  masks30) python "$R/scripts/generate_masks.py" --frames 30 ;;
  masks200) python "$R/scripts/generate_masks.py" --frames 200 ;;
  slam30) python "$R/scripts/run_window.py" --frames 30 --run-name "R0_30_$(date +%Y%m%d_%H%M%S)" ;;
  slam200) python "$R/scripts/run_window.py" --frames 200 --run-name "R0_200_$(date +%Y%m%d_%H%M%S)" ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
