#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/t04_chy_kitti00_200_20261012
# The Gaussian launcher holds this lock throughout BOTH runs. Wait, never overlap.
exec 9>"$R/v2_job.lock"
flock -x -w 2100 9
test "$(cat "$R/v2_off.exit")" = 0
test "$(cat "$R/v2_on.exit")" = 0
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$R/orb_matched_v2"
test ! -e tracking.log
flock -n job.lock bash run.sh
