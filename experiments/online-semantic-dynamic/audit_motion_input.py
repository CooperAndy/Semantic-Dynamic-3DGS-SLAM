"""Input-interface diagnostic only: dense raw stereo vs birth-filtered depth."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiment import REMOTE

R=REMOTE/'inputs200';m=json.loads((R/'runtime.json').read_text())
registration=json.loads(Path('/root/autodl-tmp/stage3_g19_past_conf_20261029_p47149/registration.json').read_text())
rows=[]
for row in m['rows']:
    i=row['frame_id'];raw_record=registration['rows'][i]['inputs']['depth_left'];raw=Path(raw_record['path'])
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==raw_record['sha256']
    a=np.load(raw,allow_pickle=False);b=np.load(row['depth'],allow_pickle=False)
    det=json.loads((R/'instances'/f'{i:06}.json').read_text())
    with np.load(R/'instances'/f'{i:06}_masks.npz',allow_pickle=False) as z:masks=z['masks']
    for instance in det['instances']:
        mask=masks[instance['mask_index']].astype(bool);n=int(mask.sum())
        if not n:continue
        rows.append(dict(frame=i,track_id=instance['track_id'],pixels=n,
                         raw_depth_valid_fraction=float((np.isfinite(a[mask])&(a[mask]>0)).mean()),
                         birth_depth_valid_fraction=float((np.isfinite(b[mask])&(b[mask]>0)).mean())))
threshold=.35
result=dict(status='COMPLETE_P1_DEPTH_INTERFACE_DIAGNOSTIC',instances=len(rows),
            birth_depth_cannot_reach_min_valid_ratio=sum(x['birth_depth_valid_fraction']<threshold for x in rows),
            raw_depth_cannot_reach_min_valid_ratio=sum(x['raw_depth_valid_fraction']<threshold for x in rows),
            tsdd_min_valid_ratio=threshold,scope='Upper bound before FB and occlusion; not motion classification accuracy; no GT used',rows=rows)
(REMOTE/'motion_depth_interface_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
