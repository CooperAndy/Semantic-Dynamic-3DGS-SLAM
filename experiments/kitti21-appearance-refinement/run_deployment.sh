#!/usr/bin/env bash
set -euo pipefail
r=/root/autodl-tmp/seq21_chy_20260926
while tmux has-session -t seq21_test 2>/dev/null; do sleep 3; done
test "$(cat "$r/test_status.txt")" = TEST_BATCH_PASS
python - <<'PY'
from pathlib import Path
import csv,json,hashlib
r=Path('/root/autodl-tmp/seq21_chy_20260926');d=r/'deployment';d.mkdir(exist_ok=False)
with (d/'split.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['local_index','original_frame_id','split']);w.writeheader();w.writerows([{'local_index':i,'original_frame_id':1200+i,'split':'Train'} for i in range(200)])
sha=hashlib.sha256((d/'split.csv').read_bytes()).hexdigest()
(d/'roi_mask_ledger.csv').write_bytes((r/'protocol/roi_mask_ledger.csv').read_bytes())
roi_sha=hashlib.sha256((d/'roi_mask_ledger.csv').read_bytes()).hexdigest()
(d/'protocol_manifest.json').write_text(json.dumps({'status':'ALL_200_SEEN_DEPLOYMENT_NO_HELDOUT','split_sha256':sha,'roi_mask_ledger_sha256':roi_sha,'frames':[1200,1399],'pose_source':'estimated ORB-SLAM3 P2 c2w','after_test_consumed_sha256':hashlib.sha256((r/'protocol/test_consumed.json').read_bytes()).hexdigest()},indent=2)+'\n')
print('DEPLOYMENT_FROZEN',sha)
PY
cd /root/autodl-tmp/SplaTAM
/root/miniconda3/bin/python scripts/splatam.py "$r/splatam_all200_config.py" > "$r/F200_all200.log" 2>&1
f200="$r/runs/all200_orb_raft_t060/params.npz"; test -s "$f200"
sha=$(sha256sum "$f200" | awk '{print $1}')
readarray -t ids < <(seq 1200 1399)
/root/miniconda3/bin/python "$r/train_f1_seq21.py" --map "$f200" --expected-map-sha256 "$sha" --out "$r/runs/F1_all200" --train-frame-ids "${ids[@]}" --steps 800 --color-lr 0.001 --opacity-lr 0.005 --protocol-root "$r/deployment" --run-kind f1_all200 > "$r/F1_all200.log" 2>&1
python - <<'PY'
from pathlib import Path
import numpy as np,hashlib,json
r=Path('/root/autodl-tmp/seq21_chy_20260926');a=r/'runs/all200_orb_raft_t060/params.npz';b=r/'runs/F1_all200/params.npz'
with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
 assert set(x.files)==set(y.files)
 fixed={k:bool(np.array_equal(x[k],y[k])) for k in x.files if k not in {'rgb_colors','logit_opacities'}}
 assert all(fixed.values())
 count=int(x['means3D'].shape[0]);assert count==int(y['means3D'].shape[0]) and x['cam_trans'].shape[-1]==200
(r/'deployment/field_verification.json').write_text(json.dumps({'F200_gaussians':count,'F1_gaussians':count,'all_fixed_equal':True,'fixed_fields':fixed},indent=2)+'\n')
print('DEPLOYMENT_FIELD_PASS',count)
PY
echo DEPLOYMENT_PASS > "$r/deployment_status.txt"
