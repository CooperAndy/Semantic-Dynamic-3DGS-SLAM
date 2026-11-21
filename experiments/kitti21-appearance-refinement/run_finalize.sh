#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
while tmux has-session -t seq21_deployment 2>/dev/null; do sleep 4; done
test "$(cat "$r/deployment_status.txt")" = DEPLOYMENT_PASS
cd /root/autodl-tmp/SplaTAM
for name in F200 F1; do
 if test "$name" = F200; then map="$r/runs/all200_orb_raft_t060/params.npz"; else map="$r/runs/F1_all200/params.npz"; fi
 /root/miniconda3/bin/python "$r/evaluate_seen_seq21.py" --protocol-root "$r/deployment" --map "$map" --out "$r/deployment/seen_${name}" --frame-ids 1200 1249 1299 1349 1399 --run-id "seq21_${name}_all200" > "$r/deployment/seen_${name}.log" 2>&1
done
f1="$r/runs/F1_all200/params.npz"
sha=$(sha256sum "$f1" | awk '{print $1}')
count=$(python - "$f1" <<'PY'
import numpy as np,sys
with np.load(sys.argv[1],allow_pickle=False) as x:print(len(x['means3D']))
PY
)
/root/miniconda3/bin/python "$r/fuse_semantic_gaussians_seq21.py" --params "$f1" --semantic-dir "$r/semantic" --rgb-dir "$r/sequences/21_1200_1399/image_2" --output-dir "$r/semantic_fusion_F1_all200" --splatam-root /root/autodl-tmp/SplaTAM --expected-gaussians "$count" --expected-params-sha256 "$sha" --deep-check --dry-run --dry-run-report "$r/deployment/semantic_dryrun_report.json" > "$r/deployment/semantic_dryrun.log" 2>&1
/root/miniconda3/bin/python "$r/fuse_semantic_gaussians_seq21.py" --params "$f1" --semantic-dir "$r/semantic" --rgb-dir "$r/sequences/21_1200_1399/image_2" --output-dir "$r/semantic_fusion_F1_all200" --splatam-root /root/autodl-tmp/SplaTAM --expected-gaussians "$count" --expected-params-sha256 "$sha" > "$r/deployment/semantic_fusion.log" 2>&1
python "$r/export_seq21_assets.py" --params "$r/runs/all200_orb_raft_t060/params.npz" --out "$r/viewer_assets/F200" --name KITTI21_F200 > "$r/deployment/export_F200.log" 2>&1
python "$r/export_seq21_assets.py" --params "$f1" --out "$r/viewer_assets/F1" --name KITTI21_F1 > "$r/deployment/export_F1.log" 2>&1
echo FINALIZE_PASS > "$r/finalize_status.txt"
