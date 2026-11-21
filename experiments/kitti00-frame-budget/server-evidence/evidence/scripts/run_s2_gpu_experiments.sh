#!/usr/bin/env bash
set -euo pipefail

SPLATAM_ROOT=/root/autodl-tmp/SplaTAM_e6b
EVIDENCE_ROOT=/root/autodl-tmp/slam_project/stage1r_confidence_mapping
RESULT_ROOT="$EVIDENCE_ROOT/results/s2_frame_efficiency_20260919"
LOG_ROOT="$EVIDENCE_ROOT/logs/s2_frame_efficiency_20260919"
PYTHON=/root/miniconda3/bin/python
CONFIG="$SPLATAM_ROOT/configs/kitti/splatam_seq00_s2_frame_efficiency.py"
EVALUATOR="$SPLATAM_ROOT/scripts/evaluate_s2_common_support.py"
DATA_CONFIG="$SPLATAM_ROOT/configs/data/kitti_seq00_s2_common_raw_raft.yaml"
STATUS="$RESULT_ROOT/s2_gpu_status.txt"

mkdir -p "$RESULT_ROOT/common_support" "$LOG_ROOT"
nvidia-smi >/dev/null

declare -A RUN_NAMES=(
  [m025]=seq00_s2_confhard_t060_m025_holdout10_0
  [m050]=seq00_s2_confhard_t060_m050_holdout10_0
  [m100]=seq00_s2_confhard_t060_m100_holdout10_0
  [noheldout200]=seq00_s2_confhard_t060_noheldout_200_0
)

run_mapping() {
  local mode="$1"
  local run_name="${RUN_NAMES[$mode]}"
  local params="$SPLATAM_ROOT/experiments/KITTI/$run_name/params.npz"
  local log="$LOG_ROOT/${mode}.log"
  if [[ -f "$params" ]]; then
    printf '%s REUSED_EXISTING %s\n' "$mode" "$(date -Is)" | tee -a "$STATUS"
    return
  fi
  if [[ -e "$SPLATAM_ROOT/experiments/KITTI/$run_name" ]]; then
    echo "Refusing pre-existing incomplete run directory: $run_name" >&2
    exit 12
  fi
  printf '%s STARTED %s\n' "$mode" "$(date -Is)" | tee -a "$STATUS"
  (
    cd "$SPLATAM_ROOT"
    PYTHONDONTWRITEBYTECODE=1 S2_MODE="$mode" CUDA_VISIBLE_DEVICES=0 \
      "$PYTHON" scripts/splatam_s2_frame_efficiency.py "$CONFIG"
  ) 2>&1 | tee "$log"
  test -f "$params"
  printf '%s COMPLETED %s\n' "$mode" "$(date -Is)" | tee -a "$STATUS"
}

run_common_eval() {
  local mode="$1"
  local params="$2"
  local role="$3"
  local output="$RESULT_ROOT/common_support/$mode"
  if [[ -f "$output/common_support_summary.json" ]]; then
    printf '%s COMMON_EVAL_REUSED %s\n' "$mode" "$(date -Is)" | tee -a "$STATUS"
    return
  fi
  (
    cd "$SPLATAM_ROOT"
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$EVALUATOR" \
      --params "$params" \
      --data-config "$DATA_CONFIG" \
      --threshold 0.60 \
      --frame-role "$role" \
      --output-dir "$output"
  )
  printf '%s COMMON_EVAL_COMPLETED %s\n' "$mode" "$(date -Is)" | tee -a "$STATUS"
}

for mode in m025 m050 m100 noheldout200; do
  run_mapping "$mode"
done

run_common_eval m025 "$SPLATAM_ROOT/experiments/KITTI/${RUN_NAMES[m025]}/params.npz" heldout
run_common_eval m050 "$SPLATAM_ROOT/experiments/KITTI/${RUN_NAMES[m050]}/params.npz" heldout
run_common_eval m100 "$SPLATAM_ROOT/experiments/KITTI/${RUN_NAMES[m100]}/params.npz" heldout
run_common_eval m190 "$SPLATAM_ROOT/experiments/KITTI/seq00_s1_confhard_t060_holdout_200_0/params.npz" heldout
run_common_eval noheldout200 "$SPLATAM_ROOT/experiments/KITTI/${RUN_NAMES[noheldout200]}/params.npz" in_sample

printf 'ALL_S2_GPU_AND_COMMON_EVAL_COMPLETE %s\n' "$(date -Is)" | tee -a "$STATUS"
