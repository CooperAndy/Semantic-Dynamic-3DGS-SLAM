#!/usr/bin/env bash
set -uo pipefail
ROOT=/root/autodl-tmp/s1_t050_chy_20260929
SPLAT=/root/autodl-tmp/SplaTAM_e6b
cd "$SPLAT" || exit 2
export S1_THRESHOLD_TAG=t050
printf 'START %s\n' "$(date -Is)" > "$ROOT/mapping_status.txt"
/root/miniconda3/bin/python scripts/splatam_holdout_compare.py "$ROOT/configs/splatam_seq00_s1_t050_holdout_200.py" > "$ROOT/logs/mapping.log" 2>&1
rc=$?
printf 'EXIT_CODE %s END %s\n' "$rc" "$(date -Is)" >> "$ROOT/mapping_status.txt"
exit "$rc"
