#!/bin/bash
set -euo pipefail
r=/root/autodl-tmp/stage2_round12_20261031_v1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg P5_AUDIT_ROOT="$r"
cd "$r"
find source p5_snapshot -type f -print0 | sort -z | xargs -0 sha256sum > source_sha256.txt
for policy in D0 D3; do
 export P5_POLICY="$policy" P5_AUDIT_OUT="$r/$policy"
 echo "START $policy $(date -u +%FT%TZ)" | tee -a queue.log
 if timeout 3500s /root/venvs/semantic_seg/bin/python -u "$r/source/run40.py" >"$r/$policy/run.log" 2>&1; then
  echo "SUCCESS $policy $(date -u +%FT%TZ)" | tee -a queue.log
 else
  rc=$?; echo "FAIL $policy status=$rc $(date -u +%FT%TZ)" | tee -a queue.log; exit "$rc"
 fi
done
