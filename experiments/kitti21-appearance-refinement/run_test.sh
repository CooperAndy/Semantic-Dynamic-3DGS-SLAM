#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
while tmux has-session -t seq21_f1_dev 2>/dev/null; do sleep 3; done
test "$(cat "$r/dev_status.txt")" = DEV_EVAL_PASS
/root/miniconda3/bin/python "$r/freeze_and_test.py" > "$r/test_freeze.log" 2>&1
cd /root/autodl-tmp/SplaTAM
for name in F0 F1; do
 if test "$name" = F0; then map="$r/runs/train160_orb_raft_t060/params.npz"; else map="$r/runs/F1_train160/params.npz"; fi
 /root/miniconda3/bin/python "$r/protocol/evaluate_seq21_batch.py" --protocol-root "$r/protocol" --map "$map" --out "$r/eval/Test_$name" --split Test --frozen-decision "$r/protocol/frozen_test_decision.json" --test-batch-name "$name" --run-id "seq21_${name}_train160" > "$r/eval_Test_${name}.log" 2>&1
done
test -s "$r/protocol/test_consumed.json"
echo TEST_BATCH_PASS > "$r/test_status.txt"
