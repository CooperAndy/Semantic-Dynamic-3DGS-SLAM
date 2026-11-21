#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
/root/venvs/raft_stereo/bin/python "$r/generate_raft_right.py" --raft-root /root/autodl-tmp/slam_project/stage1_driving_adaptation/third_party/RAFT-Stereo --checkpoint /root/raft_stereo_models/raftstereo-middlebury.pth --sequence-dir "$r/sequences/21_1200_1399" --output-dir "$r/raft_right" --num-frames 200 --max-depth 40 --preview-indices 0 99 199 > "$r/raft_right.log" 2>&1
