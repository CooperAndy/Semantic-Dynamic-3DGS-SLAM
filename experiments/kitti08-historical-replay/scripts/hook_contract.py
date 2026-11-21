from common import *
import sys,types,numpy as np,torch
sys.path.insert(0,str(R/'scripts'));from factor_hooks import install
checks=[]
for seed in range(3):
 for t in [198,200,998]:
  for it in [7,15,35,55]:
   fixture=R/'cpu_contracts'/f'S{seed}_{t}_{it}';(fixture/'runs').mkdir(parents=True)
   expected={}
   for budget in [0,3]:
    v=f'B{budget:02d}C0_S{seed}';out=fixture/'runs'/v;out.mkdir()
    params=dict(rgb_colors=torch.nn.Parameter(torch.tensor([[-.2,.5,1.2]])),means3D=torch.nn.Parameter(torch.tensor([[2.,-2.,3.]])))
    def orig_loss(p,d,variables,idx,*a,**k):
     assert d['id']==idx and bool((d['im']==idx).all());return p['rgb_colors'].sum(),variables,{}
    def orig_opt(p,lrs,tracking):return torch.optim.Adam([p['rgb_colors']],lr=.01)
    core=types.SimpleNamespace(get_loss=orig_loss,initialize_optimizer=orig_opt)
    cfg=dict(seed=seed,factorial=definition(v));obs,finish=install(core,cfg,out);opt=core.initialize_optimizer(params,{},False)
    time_idx=t;iter=it;curr_data=dict(im=torch.full((3,2,2),float(t)),depth=torch.ones(1,2,2),depth_confidence=torch.ones(1,2,2))
    keyframe_list=[dict(id=q,color=torch.full((3,2,2),float(q)),depth=torch.ones(1,2,2),depth_confidence=torch.ones(1,2,2)) for q in [0]+[q for q in range(1,t) if (q+1)%5==0 and q%20!=19]]
    selected_time_idx=[q['id'] for q in keyframe_list][-23:]+[t];gt_w2c_all_frames=[torch.eye(4)]*(t+1)
    d=dict(curr_data,id=t,iter_gt_w2c_list=gt_w2c_all_frames)
    value,_,_=core.get_loss(params,d,{},t,{},False,.5,True,False,mapping=True);value.backward();opt.step();finish()
    row=json.loads((out/'mapping_updates.jsonl').read_text());base=selected_time_idx[np.random.default_rng(np.random.SeedSequence([seed,t,it,0])).integers(len(selected_time_idx))]
    old=[x['id'] for x in keyframe_list if x['id']<=t-200];slot=bool(budget and old and it in [15,35,55]);target=old[np.random.default_rng(np.random.SeedSequence([seed,t,it,1])).integers(len(old))] if slot else base
    assert row['base_target']==base and row['target']==target and row['history_slot']==slot
    assert obs['completed_steps']==1 and obs['box_projection_steps']==0 and params['rgb_colors'].min()<0 and params['rgb_colors'].max()>1
    assert torch.equal(params['means3D'],torch.tensor([[2.,-2.,3.]]))
    checks.append(dict(seed=seed,t=t,it=it,budget=budget,base=base,target=target,slot=slot))
assert len(checks)==72 and not torch.cuda.is_initialized()
dump(R/'receipts/hook_cpu_contract.json',dict(status='PASS72_DONOR_COUNTER_HISTORY_CPU_FIXTURES',at=now(),checks=checks,CUDA_initialized=False))
print('CPU_CONTRACT_PASS72',flush=True)
