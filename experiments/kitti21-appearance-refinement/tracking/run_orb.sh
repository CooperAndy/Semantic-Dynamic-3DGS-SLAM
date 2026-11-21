#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/seq21_chy_20260926
bin=$(find /root/autodl-tmp/slam_project/work_chy_20260924 -type f -name stereo_kitti | head -1)
base=$(dirname "$(dirname "$bin")")
export LD_LIBRARY_PATH="$(dirname "$bin"):$base/pangolin_install/lib:$base/ORB_SLAM3_4452a3c4/Thirdparty/DBoW2/lib:${LD_LIBRARY_PATH:-}"
cd "$root/tracking"
"$bin" "$base/ORB_SLAM3_4452a3c4/Vocabulary/ORBvoc.txt" "$root/tracking/seq21_color.yaml" "$root/sequences/21_1200_1399" > orb_run.log 2>&1
