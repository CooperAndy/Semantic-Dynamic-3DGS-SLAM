#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
cd "$r/protocol"
/root/miniconda3/bin/python build_train_only_confidence.py --protocol-root "$r/protocol" --mode full > "$r/protocol/confidence_train.log" 2>&1
/root/miniconda3/bin/python build_depth_confidence_kitti.py --left-image-dir "$r/sequences/21_1200_1399/image_2" --right-image-dir "$r/sequences/21_1200_1399/image_3" --depth-left-dir "$r/raft_left" --depth-right-dir "$r/raft_right" --pose-path "$r/tracking/poses_orb_p2_c2w_rebased.txt" --calib "$r/sequences/21_1200_1399/calib.txt" --output-root "$r/confidence_all200" --static-valid-mask-dir "$r/static_valid_masks" --num-frames 200 --seed-threshold 0.60 --loss-threshold 0.20 --preview-indices 0 99 199 > "$r/confidence_all200.log" 2>&1
