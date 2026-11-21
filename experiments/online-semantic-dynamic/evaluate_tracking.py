"""Post-seal shadow RPE; no composed-trajectory or closed-loop claim."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiment import REMOTE

R=REMOTE/'E1_tracking';f=R/'predictions.json'
receipt=json.loads((R/'prediction_receipt.json').read_text())
assert receipt['status']=='PASS_E1_FROZEN_TRACKING_REPLAY'
assert hashlib.sha256(f.read_bytes()).hexdigest()==receipt['prediction_sha256']
reference=Path('/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/evaluation_only/B03_MASS_S0_metrics.npz')
with np.load(reference,allow_pickle=False) as z:G=z['ground_truth_P2_c2w'].copy()
P=reference.parents[1]/'runs/B03_MASS_S0/backend_events.jsonl'
parents=[json.loads(line) for line in P.read_text().splitlines()]
rows=json.loads(f.read_text());errors={k:[] for k in ['PREDICTION','NONE','SEMANTIC','RANDOM0','RANDOM1','RANDOM2']};details=[]
for row in rows:
    i=row['frame'];previous=np.asarray(parents[i-1]['selected_w2c'],np.float64)
    truth=np.linalg.inv(G[i-1])@G[i]
    def error(W):
        relative=previous@np.linalg.inv(W);E=np.linalg.inv(truth)@relative
        return float(np.linalg.norm(E[:3,3]))
    predicted=np.asarray(row['prediction_w2c'],np.float64);e0=error(predicted);errors['PREDICTION'].append(e0)
    item=dict(frame=i,prediction_error_m=e0,arms={})
    for name,arm in row['arms'].items():
        candidate=np.asarray(arm['candidate_w2c'],np.float64) if arm['ok'] and 'candidate_w2c' in arm else predicted
        e=error(candidate);errors[name].append(e);item['arms'][name]=dict(error_m=e,geometry_ok=arm['ok'])
    details.append(item)
summary={k:dict(frames=len(v),SSE_m2=float(np.square(v).sum()),RMSE_m=float(np.sqrt(np.square(v).mean()))) for k,v in errors.items()}
result=dict(status='COMPLETE_E1_SHADOW_GEOMETRY_EVALUATION',summary=summary,rows=details,
            prediction_sha256=receipt['prediction_sha256'],GT_file_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            scope='All199 steps with same frozen previous pose; valid geometry substituted, failed geometry falls back. Not MASS acceptance or closed-loop RPE.',quality_adoption=False)
(R/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary))
