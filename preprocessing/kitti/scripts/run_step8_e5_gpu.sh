#!/usr/bin/env bash
set -euo pipefail

SPLATAM_ROOT="/root/autodl-tmp/SplaTAM"
STAGE1_ROOT="/root/autodl-tmp/slam_project/stage1_driving_adaptation"
DATA_ROOT="/root/autodl-tmp/slam_project/data/kitti/dataset/sequences/00"
MASK_DIR="$DATA_ROOT/masks_static_valid"
MASKED_DEPTH_DIR="$DATA_ROOT/depth_raft_skyfar_mask_200"
RAW_DEPTH_DIR="$DATA_ROOT/depth_raft_200"
MODE="${1:-preflight}"

fail() {
  echo "[STEP8-E5][ERROR] $*" >&2
  exit 1
}

[[ -d "$SPLATAM_ROOT" ]] || fail "Missing SplaTAM: $SPLATAM_ROOT"
[[ -f "$STAGE1_ROOT/scripts/verify_step8_e5_package.py" ]] || fail "Missing verifier"

python "$STAGE1_ROOT/scripts/verify_step8_e5_package.py" \
  --mask-dir "$MASK_DIR" \
  --masked-depth-dir "$MASKED_DEPTH_DIR" \
  --raw-depth-dir "$RAW_DEPTH_DIR" \
  --num-frames 200

cd "$SPLATAM_ROOT"
python -c 'import ast, pathlib, sys; [ast.parse(pathlib.Path(p).read_text()) for p in sys.argv[1:]]' \
  configs/kitti/splatam_seq00_orbpose_raft_skyfar_smoke.py \
  configs/kitti/splatam_seq00_orbpose_raft_skyfar_200.py
python -c 'import pathlib, yaml; p=pathlib.Path("configs/data/kitti_seq00_orb_raft_skyfar200.yaml"); cfg=yaml.safe_load(p.read_text()); assert cfg["dataset_name"] == "kitti"; assert pathlib.Path(cfg["pose_path"]).is_file(); assert pathlib.Path(cfg["depth_folder"]).is_dir(); print(cfg)' 

echo "[STEP8-E5] CPU preflight: PASS"
df -h /root/autodl-tmp

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[STEP8-E5] GPU: NOT AVAILABLE"
  [[ "$MODE" == "preflight" ]] && exit 0
  fail "Start the RTX 5090 instance, then rerun this command"
fi
echo "[STEP8-E5] GPU: READY"

case "$MODE" in
  preflight)
    exit 0
    ;;
  smoke)
    config="configs/kitti/splatam_seq00_orbpose_raft_skyfar_smoke.py"
    run_name="seq00_raft_skyfar_orbpose_smoke_0"
    log_name="run_kitti_seq00_raft_skyfar_orbpose_smoke.log"
    ;;
  e5|200)
    config="configs/kitti/splatam_seq00_orbpose_raft_skyfar_200.py"
    run_name="seq00_raft_skyfar_orbpose_200_0"
    log_name="run_kitti_seq00_raft_skyfar_orbpose_200.log"
    ;;
  *)
    fail "Usage: $0 [preflight|smoke|e5]"
    ;;
esac

output_dir="experiments/KITTI/$run_name"
[[ ! -e "$output_dir" ]] || fail "Output already exists: $output_dir"
echo "[STEP8-E5] starting $MODE with $config"
python scripts/splatam.py "$config" 2>&1 | tee "$log_name"
