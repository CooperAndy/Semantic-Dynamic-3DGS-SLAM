"""Attach IDs to variables and checkpoint copies without changing optimization."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def install(core,out):
    out=Path(out);(out/'pruning_ids').mkdir()
    state={'variables':None,'next_id':0,'frame':0,'prune_event':0}
    init=core.initialize_first_timestep;add=core.add_new_gaussians
    def init_hook(*args,**kwargs):
        result=init(*args,**kwargs);p,v=result[:2];n=len(p['means3D'])
        assert state['variables'] is None
        v['stable_id']=torch.arange(n,device=p['means3D'].device,dtype=torch.int64)
        state['variables']=v;state['next_id']=n
        return result
    def add_hook(*args,**kwargs):
        p,v=args[:2];before=len(p['means3D']);state['frame']=int(args[4])
        assert len(v['stable_id'])==before
        p,v=add(*args,**kwargs);born=len(p['means3D'])-before
        v['stable_id']=torch.cat((v['stable_id'],torch.arange(state['next_id'],state['next_id']+born,device=p['means3D'].device,dtype=torch.int64)))
        state['next_id']+=born;state['variables']=v
        assert len(v['stable_id'])==len(p['means3D'])
        return p,v
    import utils.slam_external as external
    remove=external.remove_points
    def remove_hook(mask,p,v,opt):
        ids=v['stable_id'];assert len(ids)==len(mask)==len(p['means3D'])
        dropped=ids[mask].detach().cpu().numpy();keep=ids[~mask].clone()
        event={'event':state['prune_event'],'frame':state['frame'],'iteration':int(sys._getframe(1).f_locals.get('iter',-1)),'before':len(ids),'removed':len(dropped)}
        if len(dropped):
            fn=f"pruning_ids/{state['prune_event']:06d}.npz";np.savez_compressed(out/fn,deleted_ids=dropped)
            event['file']=fn;event['sha256']=hashlib.sha256((out/fn).read_bytes()).hexdigest()
        p,v=remove(mask,p,v,opt);v['stable_id']=keep;state['variables']=v
        assert len(keep)==len(p['means3D']) and torch.all(keep[1:]>keep[:-1])
        with (out/'pruning.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        state['prune_event']+=1
        return p,v
    def snapshot(p):
        v=state['variables'];assert len(v['stable_id'])==len(p['means3D'])
        q=dict(p);q['stable_id']=v['stable_id'];q['timestep']=v['timestep'];return q
    save=core.save_params;ckpt=core.save_params_ckpt
    def save_hook(p,*a,**k):return save(snapshot(p),*a,**k)
    def ckpt_hook(p,*a,**k):return ckpt(snapshot(p),*a,**k)
    core.initialize_first_timestep=init_hook;core.add_new_gaussians=add_hook
    external.remove_points=remove_hook;core.save_params=save_hook;core.save_params_ckpt=ckpt_hook
    return state
