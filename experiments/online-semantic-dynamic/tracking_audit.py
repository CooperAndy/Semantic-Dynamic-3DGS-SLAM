"""Full-ledger tracking diagnostics, including every prediction fallback."""
import collections,json
from pathlib import Path

def audit_tracking(folder,frames=1000):
    folder=Path(folder)
    rows=[json.loads(line) for line in (folder/'backend_events.jsonl').read_text().splitlines()]
    assert [r['frame'] for r in rows]==list(range(frames))
    steps=rows[1:];reasons=collections.Counter(r['geometry_reason'] for r in steps if not r['geometry_ok'])
    matches=[];inliers=[];cells=[];semantic_removed=0
    for row in steps:
        p=folder/row['proposal_path'];proposal=json.loads(p.read_text())
        matches.append(proposal.get('matches',0));inliers.append(len(proposal.get('inliers',[])))
        cells.append(proposal.get('occupied_cells',0))
        semantic_removed+=proposal.get('semantic_filter',{}).get('removed',0)
    result=dict(frames=frames,tracking_steps=len(steps),accepted_steps=sum(r['accepted'] for r in steps),
                geometry_qualified_steps=sum(r['geometry_ok'] for r in steps),
                geometry_failure_steps=sum(not r['geometry_ok'] for r in steps),
                prediction_fallback_steps=sum(not r['accepted'] for r in steps),
                geometry_failure_reasons=dict(reasons),matches=matches,inliers=inliers,
                occupied_4x4_cells=cells,semantic_removed_correspondences=semantic_removed,
                definition='Geometry failures are failed proposal qualification; fallback is a valid predicted pose, not an absent trajectory frame')
    totals=collections.Counter()
    ledger=folder/'semantic_events.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            event=json.loads(line)
            for kind,field,key in [('D_birth_mask','removed','D_birth_candidates_removed'),('D_loss_mask','depth_removed','D_depth_loss_pixel_visits_removed'),('D_loss_mask','RGB_removed','D_RGB_loss_pixel_visits_removed'),('D_tracking_input','reference_valid_depth_removed','D_tracking_reference_pixel_visits_removed')]:
                if event['kind']==kind:totals[key]+=event[field]
    result['dynamic_consumption']=dict(totals)
    result['dynamic_mapping_status']='INFORMATIVE' if sum(totals[k] for k in ['D_birth_candidates_removed','D_depth_loss_pixel_visits_removed','D_RGB_loss_pixel_visits_removed'])>0 else 'NOT_INFORMATIVE'
    (folder/'tracking_diagnostics.json').write_text(json.dumps(result,indent=2))
    return result
