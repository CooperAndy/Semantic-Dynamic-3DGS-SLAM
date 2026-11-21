#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/t04_chy_kitti00_200_20261012/orb_matched_v2"
export LD_LIBRARY_PATH="/root/autodl-tmp/t04_chy_kitti00_200_20261012/orb_matched_v2/DBoW2_x86_64/lib:/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/p4_prebuilt:/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/pangolin_install/lib:/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/ORB_SLAM3_4452a3c4/Thirdparty/DBoW2/lib:/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/ORB_SLAM3_4452a3c4/Thirdparty/g2o/lib:"
export OMP_NUM_THREADS=4
set +e
timeout 10m ./stereo_kitti_headless "/root/autodl-tmp/slam_project/work_chy_20260924/stage2/06_独立复核与TestOneShot/vehicle_only_v2/ORB_SLAM3_4452a3c4/Vocabulary/ORBvoc.txt" settings.yaml "/root/autodl-tmp/t04_chy_kitti00_200_20261012/matched_stereo" > tracking.log 2>&1
status=$?
printf "%s\n" "$status" > tracking.exit
exit "$status"
