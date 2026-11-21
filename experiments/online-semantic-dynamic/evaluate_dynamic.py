import json
from pathlib import Path
import numpy as np
from experiment import REMOTE

R=REMOTE/'evaluation_only';diagnostic=R/'dynamic_diagnostic/results.json'
result=json.loads(diagnostic.read_text());runtime=json.loads((REMOTE/'inputs200/runtime.json').read_text())
truth_file=Path('/root/autodl-tmp/stage3_g30_recovery_run_v2_20261101_p44181/evaluation_only/B03_MASS_S0_metrics.npz')
with np.load(truth_file,allow_pickle=False) as z:G=z['ground_truth_P2_c2w']
events=[json.loads(x) for x in (truth_file.parents[1]/'runs/B03_MASS_S0/backend_events.jsonl').read_text().splitlines()]
arms={n:[] for n in ['NONE','PREDICTED_DYNAMIC','OFFICIAL_SPARSE_MOTION']};deletions={n:0 for n in arms}
supported_ref=0;supported_current=0;correspondences=0
for row in result['rows']:
    if 'arms' not in row:continue
    i=row['frame'];Tgt=np.linalg.inv(G[i-1])@G[i];previous=np.asarray(events[i-1]['selected_w2c'])
    for name,p in row['arms'].items():
        W=np.asarray(p['candidate_w2c']) if p['ok'] and 'candidate_w2c' in p else np.asarray(events[i]['prediction_w2c'])
        error=np.linalg.inv(Tgt)@(previous@np.linalg.inv(W));arms[name].append(float(np.linalg.norm(error[:3,3])));deletions[name]+=p['removed']
    supported_ref+=row['reference_GT_supported'];supported_current+=row['current_GT_supported'];correspondences+=row['correspondences']
counts=dict(TP=0,FP=0,FN=0,TN=0)
for i in range(200):
    with np.load(R/'official_gt_v2'/f'{3000+i:06}.npz') as z:known=z['motion_known'];moving=z['moving']
    with np.load(REMOTE/'inputs200/observations'/f'{i:06}.npz') as z:pred=z['state']==2
    counts['TP']+=int((known&moving&pred).sum());counts['FP']+=int((known&~moving&pred).sum())
    counts['FN']+=int((known&moving&~pred).sum());counts['TN']+=int((known&~moving&~pred).sum())
tp,fp,fn=counts['TP'],counts['FP'],counts['FN']
out=dict(status='COMPLETE_NO_MANUAL_DYNAMIC_EVALUATION',motion_counts=counts,
         precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
         F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
         shadow={k:dict(steps=len(v),RMSE_m=float(np.sqrt(np.square(v).mean())) if v else None,removed_correspondences=deletions[k]) for k,v in arms.items()},
         correspondences=correspondences,reference_GT_supported=supported_ref,current_GT_supported=supported_current,
         scope='Official projected sparse label support; twenty fixed target steps, not full-frame oracle; no GT in deployed predictor',quality_adoption=False)
(R/'dynamic_evaluation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
