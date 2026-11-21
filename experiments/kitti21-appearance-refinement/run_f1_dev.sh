#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
while tmux has-session -t seq21_train160 2>/dev/null; do sleep 3; done
f0="$r/runs/train160_orb_raft_t060/params.npz"
test -s "$f0"
readarray -t ids < <(python - <<'PY'
import csv
p='/root/autodl-tmp/seq21_chy_20260926/protocol/split.csv'
for row in csv.DictReader(open(p)):
 if row['split']=='Train': print(row['original_frame_id'])
PY
)
test "${#ids[@]}" -eq 160
sha=$(sha256sum "$f0" | awk '{print $1}')
cd /root/autodl-tmp/SplaTAM
/root/miniconda3/bin/python "$r/train_f1_seq21.py" --map "$f0" --expected-map-sha256 "$sha" --out "$r/runs/F1_train160" --train-frame-ids "${ids[@]}" --steps 640 --color-lr 0.001 --opacity-lr 0.005 --protocol-root "$r/protocol" --run-kind f1_train160 > "$r/F1_train160.log" 2>&1
/root/miniconda3/bin/python "$r/protocol/evaluate_seq21_batch.py" --protocol-root "$r/protocol" --map "$f0" --out "$r/eval/Dev_F0" --split Dev --run-id seq21_F0_train160 > "$r/eval_Dev_F0.log" 2>&1
/root/miniconda3/bin/python "$r/protocol/evaluate_seq21_batch.py" --protocol-root "$r/protocol" --map "$r/runs/F1_train160/params.npz" --out "$r/eval/Dev_F1" --split Dev --run-id seq21_F1_train160 > "$r/eval_Dev_F1.log" 2>&1
echo DEV_EVAL_PASS > "$r/dev_status.txt"
