#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
/root/venvs/raft_stereo/bin/python /root/autodl-tmp/slam_project/stage1_driving_adaptation/scripts/generate_raft_depth_kitti.py --raft-root /root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo --checkpoint /root/raft_stereo_models/raftstereo-middlebury.pth --sequence-dir "$r/sequences/21_1200_1399" --output-dir "$r/raft_left" --num-frames 200 --max-depth 40 --preview-indices 0 99 199 > "$r/raft_left.log" 2>&1
