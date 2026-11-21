#!/usr/bin/env bash
set -euo pipefail

SPLATAM_ROOT=/root/autodl-tmp/SplaTAM_e6b
EVIDENCE_ROOT=/root/autodl-tmp/slam_project/stage1r_confidence_mapping
STATUS="$EVIDENCE_ROOT/results/s1_threshold_audit_20260916/gpu_ablation_status.txt"
CONFIG="$SPLATAM_ROOT/configs/kitti/splatam_seq00_s1_confhard_holdout_200.py"

cd "$SPLATAM_ROOT"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED_NO_GPU $(date -Is)" | tee -a "$STATUS"
  exit 3
fi

for threshold_tag in t060 t070 t090; do
  export S1_THRESHOLD_TAG="$threshold_tag"
  run_name="seq00_s1_confhard_${threshold_tag}_holdout_200_0"
  output_dir="$SPLATAM_ROOT/experiments/KITTI/$run_name"
  log="$EVIDENCE_ROOT/logs/s1_${threshold_tag}_holdout_200.log"
  if [[ -e "$output_dir" || -e "$log" ]]; then
    echo "REFUSE_EXISTING $threshold_tag output=$output_dir log=$log" | tee -a "$STATUS"
    exit 4
  fi
  echo "RUNNING $threshold_tag $(date -Is)" | tee -a "$STATUS"
  /root/miniconda3/bin/python scripts/splatam_holdout_compare.py "$CONFIG" > "$log" 2>&1
  echo "COMPLETED $threshold_tag $(date -Is)" | tee -a "$STATUS"
  grep -E "Average Mapping/Frame Time|Average PSNR|Average Depth L1|Average MS-SSIM|Average LPIPS|11R Total|11R Peak" "$log" | tee -a "$STATUS"
done

echo "ALL_REQUESTED_GPU_RUNS_COMPLETE $(date -Is)" | tee -a "$STATUS"
