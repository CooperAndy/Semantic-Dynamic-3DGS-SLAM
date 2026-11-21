"""CPU replay only deterministic perturbed IDs to measure recovery support counts."""
import json
from pathlib import Path
import numpy as np
from semantic_cpu import factor_weights
R=Path(__file__).resolve().parents[1];O=R/'05_条件轻量候选与历史诊断/dev_diagnostics_20261007_v2';GROUPS=['S0','S1','S-G','S2','S2-H'];out=[]
man=json.loads((R/'inputs/vkitti2_gt_dev_v2/manifest.json').read_text());trace=json.loads((O/'history_recovery_trace.json').read_text())
for seq in man['sequences']:
 name=seq['name'];d=R/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{name}_full'
 with np.load(R/'inputs/vkitti2_gt_dev_v2'/name/'base_map.npz') as b:ids=b['source_gaussian_index'];selected=np.flatnonzero(ids%100==0);n=len(selected);lookup=np.full(len(ids),-1,np.int32);lookup[selected]=np.arange(n)
 plain={g:np.zeros((n,19),np.float32) for g in GROUPS};bad={g:np.zeros((n,19),np.float32) for g in GROUPS};support={g:np.zeros(n,np.uint32) for g in GROUPS};initial={};first={g:np.zeros(n,np.uint32) for g in GROUPS};firstsupport={g:np.zeros(n,np.uint32) for g in GROUPS}
 for pos,f in enumerate(seq['fusion']):
  with np.load(d/'observations'/f'{f:06d}_prediction.npz') as c:i=c['gaussian_index'];take=lookup[i]>=0;k=lookup[i[take]];dist=c['distribution'][take];sem=c['semantic_reliability'][take];geo=c['geometry_reliability'][take]
  for g in GROUPS:
   w=factor_weights(g,np.ones(len(k),bool),sem,geo,.5);plain[g][k]+=(dist*w[:,None]).astype(np.float32);bad[g][k]+=((np.roll(dist,1,axis=1) if pos<3 else dist)*w[:,None]).astype(np.float32);support[g][k]+=(w>0).astype(np.uint32);equal=plain[g].argmax(1)==bad[g].argmax(1);known=plain[g].sum(1)>0
   if pos==2:initial[g]=~equal&known
   if pos>2:
    newly=initial[g]&equal&known&(first[g]==0);first[g][newly]=pos+1;firstsupport[g][newly]=support[g][newly]
 for g in GROUPS:
  equal=plain[g].argmax(1)==bad[g].argmax(1);targets=initial[g];row=next(r for r in trace if r['sequence']==name and r['group']==g and r['distinct_frames_processed']==21);assert int((targets&equal).sum())==row['initial_targets_currently_match_unperturbed_prefix']
  np.savez_compressed(O/name/f'{g}_recovery_detail.npz',Gaussian_id=ids[selected][targets],first_matching_prefix_distinct_frames=first[g][targets],effective_source_support_at_first_match=firstsupport[g][targets],final_effective_source_support=support[g][targets],final_matches_unperturbed_label=equal[targets])
  recovered=targets&equal;unresolved=targets&~equal;out.append({'sequence':name,'group':g,'initial_changed_labels':int(targets.sum()),'final_match_count':int(recovered.sum()),'final_recovery_fraction':float(recovered.sum()/max(targets.sum(),1)),'first_match_median_distinct_processed_frames':float(np.median(first[g][targets&(first[g]>0)])) if (targets&(first[g]>0)).any() else None,'first_match_median_effective_source_support':float(np.median(firstsupport[g][targets&(first[g]>0)])) if (targets&(first[g]>0)).any() else None,'unresolved_with_at_most_3_effective_source_observations':int((unresolved&(support[g]<=3)).sum()),'scope':'agreement with unperturbed same-method evolving label, not GT correctness; first match may later flip again'})
(O/'recovery_support_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out),flush=True)
