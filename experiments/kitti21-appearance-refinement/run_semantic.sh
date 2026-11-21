#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
while tmux has-session -t seq21_raft_right 2>/dev/null; do sleep 2; done
test -f "$r/raft_right/generation_summary.json"
/root/venvs/semantic_seg/bin/python /root/autodl-tmp/slam_project/stage2_semantic/scripts/generate_mask2former_semantics_kitti.py --image-dir "$r/sequences/21_1200_1399/image_2" --output-dir "$r/semantic" --model-id /root/autodl-tmp/slam_project/stage2_semantic/models/mask2former-swin-small-cityscapes-semantic-91281d04 --model-source-id facebook/mask2former-swin-small-cityscapes-semantic --revision 91281d0492f2175fc409c66a231bc2ff9c4ebaf0 --local-files-only --start 0 --num-frames 200 --device cuda --preview-every 100 --seed 42 > "$r/semantic.log" 2>&1
