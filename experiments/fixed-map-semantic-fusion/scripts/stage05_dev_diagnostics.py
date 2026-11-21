"""DEV-only condition audit and shared early-evidence stress; no final-Test access."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from semantic_cpu import Evidence,factor_weights
R=Path(__file__).resolve().parents[1];D=R/'05_条件轻量候选与历史诊断';O=D/'dev_diagnostics_20261007_v2'
GROUPS=['S0','S1','S-G','S2','S2-H']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not O.exists();O.mkdir();manpath=R/'inputs/vkitti2_gt_dev_v2/manifest.json';man=json.loads(manpath.read_text())
 protocol={'status':'FROZEN_BEFORE_05_DIAGNOSTICS','windows':[s['name'] for s in man['sequences']],'exposure':'HISTORICALLY_EXPOSED_DEV; no KITTI360 final-entry access','history':'OFF all groups','stress':'first3 allowed source frames; active stableGaussianID modulo100==0; rotate19-class associated frame distribution by +1; restore original subsequent observations','stress_scope':'post-association Gaussian/frame evidence perturbation; not raw 2D pixel GT or a full feedback state machine','V_group_rule':'floor(local_frame/5), fixed time blocks; offline allowed fusion frames only','V_cap':1.,'V_trigger':'>=100 Gaussians in either DEV window with strong fusion-GT association consensus>=.8 and support>=3, wrong global S2 winner, >=3 wrong same-class observations in a time block, and >=2 correct observations in other time blocks','V_GT_role':'fusion-frame O-Sem associated distributions only diagnose, never V weights/grouping; not independent3DGT','Z':'separate native fusion-view depth diagnostic required, not triggered solely by V findings','candidate_combination':'FORBIDDEN_THIS_ROUND','GPU_rendering':'only condition-triggered candidate and stress states, fixed native renderer','final_acceptance':'no new isolated entry; DEV findings cannot replace prior final results'}
 dump(O/'protocol.json',protocol);locks=[];trigger=[];stress=[]
 for seq in man['sequences']:
  name=seq['name'];old=R/'02_基础因子DEV与Oracle诊断/gt_dev_v2'/f'{name}_full';base=R/'inputs/vkitti2_gt_dev_v2'/name/'base_map.npz'
  with np.load(base) as b:ids=b['source_gaussian_index'];n=len(ids)
  folder=O/name;folder.mkdir();(folder/'states').mkdir();(folder/'renders').mkdir();frames=seq['fusion'];blocks=sorted({f//5 for f in frames});bi={v:i for i,v in enumerate(blocks)};ng=len(blocks)
  oracle=np.zeros((n,19),np.float32);os=np.zeros(n,np.uint32)
  for f in frames:
   p=old/'observations'/f'{f:06d}_oracle.npz';locks.append({'path':p.relative_to(R).as_posix(),'sha256':sha(p)})
   with np.load(p) as c:i=c['gaussian_index'];oracle[i]+=c['distribution'];os[i]+=1
  ref=oracle.argmax(1);strong=(os>=3)&(oracle.max(1)/np.maximum(os,1)>=.8)
  del oracle
  correct=np.zeros((ng,n),np.uint8);wrong_counts=np.zeros((ng,n,19),np.uint8);wrong_evidence=np.zeros((ng,n,19),np.float32)
  plain={g:Evidence(ids,19) for g in GROUPS};bad={g:Evidence(ids,19) for g in GROUPS};touched=np.zeros(n,bool);initial={};rows=[]
  for pos,f in enumerate(frames):
   p=old/'observations'/f'{f:06d}_prediction.npz';locks.append({'path':p.relative_to(R).as_posix(),'sha256':sha(p)})
   with np.load(p) as c:i=c['gaussian_index'];dist=c['distribution'].copy();sem=c['semantic_reliability'];geo=c['geometry_reliability'];version=str(c['input_version'])
   b=bi[f//5];pred=dist.argmax(1);yes=pred==ref[i];correct[b,i[yes]]+=1;wrong_counts[b,i[~yes],pred[~yes]]+=1;wrong_evidence[b,i]+=dist*(sem*geo)[:,None]
   pd=dist.copy();pick=ids[i]%100==0 if pos<3 else np.zeros(len(i),bool);pd[pick]=np.roll(pd[pick],1,axis=1);touched[i[pick]]=True
   full=np.zeros((n,19),np.float32);full[i]=dist;pert=np.zeros_like(full);pert[i]=pd;source={'dataset':'VKITTI2','sequence':seq['scene']+'/clone','camera':'Camera_0','original_frame':seq['records'][f]['original_frame']}
   for g in GROUPS:
    w=np.zeros(n,np.float64);w[i]=factor_weights(g,np.ones(len(i),bool),sem,geo,.5)
    assert plain[g].update_frame(source,version,full,w);assert bad[g].update_frame(source,version+':stress_v1',pert,w)
    assert not bad[g].update_frame(source,version+':stress_v1',pert,w)
    a=plain[g].evidence.argmax(1);z=bad[g].evidence.argmax(1);known=plain[g].evidence.sum(1)>0;changed=touched&known&(a!=z)
    if pos==2:initial[g]=changed.copy()
    if pos>=2:rows.append({'sequence':name,'group':g,'distinct_frames_processed':pos+1,'initially_changed_labels':int(initial[g].sum()),'initial_targets_currently_match_unperturbed_prefix':int((initial[g]&known&(a==z)).sum()),'touched_current_label_difference':int(changed.sum()),'normal_untouched_labels_changed':int((~touched&known&(a!=z)).sum())})
   print(json.dumps({'phase':'05_CPU_history','sequence':name,'frames':pos+1,'total':len(frames)}),flush=True)
  for g in GROUPS:
   oldstate=old/'states'/f'{g}_final.npz';locks.append({'path':oldstate.relative_to(R).as_posix(),'sha256':sha(oldstate)});stored=Evidence.load(oldstate,ids)
   assert np.array_equal(stored.evidence,plain[g].evidence) and np.array_equal(stored.support,plain[g].support) and np.array_equal(stored.total_weight,plain[g].total_weight)
   assert np.array_equal(bad[g].evidence[~touched],plain[g].evidence[~touched]);assert np.array_equal(bad[g].support,plain[g].support) and np.array_equal(bad[g].total_weight,plain[g].total_weight)
   bad[g].save(folder/'states'/f'{g}-EarlyNoise_final.npz')
  winner=plain['S2'].evidence.argmax(1);eligible=strong&(plain['S2'].evidence.sum(1)>0)&(winner!=ref);witness=np.zeros(n,bool)
  for b in range(ng):
   block_winner=wrong_evidence[b].argmax(1);other_correct=correct.sum(0,dtype=np.uint16)-correct[b]
   witness|=eligible&(wrong_counts[b,np.arange(n),winner]>=3)&(block_winner==winner)&(other_correct>=2)
  triggered=int(witness.sum())>=100
  trigger.append({'sequence':name,'reference':'DEV fusion GT native associated consensus, not 3D GT','strong_reference_Gaussians':int(strong.sum()),'wrong_S2_winner_among_strong':int(eligible.sum()),'repeated_wrong_with_other_correct_witnesses':int(witness.sum()),'S2_V_triggered':triggered})
  if triggered:
   # Cap block total weight per Gaussian while preserving within-block relative weights.
   ve=np.zeros((n,19),np.float32);vs=np.zeros(n,np.uint32);vw=np.zeros(n,np.float64)
   for b in blocks:
    e=np.zeros((n,19),np.float32);w=np.zeros(n,np.float64);count=np.zeros(n,np.uint32)
    for f in [f for f in frames if f//5==b]:
     with np.load(old/'observations'/f'{f:06d}_prediction.npz') as c:i=c['gaussian_index'];weight=(c['semantic_reliability'].astype(float)*c['geometry_reliability'].astype(float));e[i]+=(c['distribution']*weight[:,None]).astype(np.float32);w[i]+=weight;count[i]+=(weight>0)
    scale=np.minimum(1.,1/np.maximum(w,1e-30));ve+=(e*scale[:,None]).astype(np.float32);vw+=w*scale;vs+=count
   s=Evidence(ids,19);s.evidence=ve;s.total_weight=vw;s.support=vs;s.processed=plain['S2'].processed.copy();s.save(folder/'states/S2-V_final.npz')
   dump(folder/'V_cap_diagnostic.json',{'capped_blocks':ng,'distinct_frame_support_not_group_count':True,'same_frame_set':frames,'selected_cap':1.,'sum_weight_before':float(plain['S2'].total_weight.sum()),'sum_weight_after':float(vw.sum()),'GT_not_used_in_weights':True,'offline':True})
  stress.extend(rows);dump(folder/'base_lock.json',{'path':base.relative_to(R).as_posix(),'sha256':sha(base),'Gaussian_count':n,'fusion':frames,'evaluation':seq['evaluation']})
  del plain,bad,wrong_evidence
 dump(O/'input_locks.json',locks);dump(O/'V_trigger_diagnostic.json',trigger);dump(O/'history_recovery_trace.json',stress)
 dump(O/'CPU_result.json',{'status':'COMPLETE_REAL_DEV_CPU_DIAGNOSTICS_AWAIT_NATIVE_RENDER','S2_V_triggered':any(t['S2_V_triggered'] for t in trigger),'V_triggers':trigger,'source_frame_replays_noop':True,'normal_untouched_evidence_bitwise_preserved':True,'all_five_original_states_CPU_reproduced':True,'Test_accessed':False})
if __name__=='__main__':main()
