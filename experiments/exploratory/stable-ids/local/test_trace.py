"""CPU check: additions, deletions and snapshots preserve exact IDs."""
import json,sys,types
from pathlib import Path
import torch
import stable_trace

def main():
 out=Path(__file__).parent/'trace_test_output';out.mkdir(exist_ok=True)
 if (out/'pruning_ids').exists():raise RuntimeError('Preserve previous test output; use a fresh path')
 utils=types.ModuleType('utils');utils.__path__=[];external=types.ModuleType('utils.slam_external')
 def remove(mask,p,v,opt):
  for k in p:p[k]=p[k][~mask]
  v['timestep']=v['timestep'][~mask]
  return p,v
 external.remove_points=remove;utils.slam_external=external;sys.modules['utils']=utils;sys.modules['utils.slam_external']=external
 core=types.SimpleNamespace();saved=[]
 core.initialize_first_timestep=lambda:({'means3D':torch.arange(9).reshape(3,3).float()},{'timestep':torch.zeros(3)})
 def add(p,v,data,sil,index,*a,**kw):
  p['means3D']=torch.cat((p['means3D'],torch.ones(2,3)));v['timestep']=torch.cat((v['timestep'],torch.full((2,),float(index))));return p,v
 core.add_new_gaussians=add
 core.save_params=lambda p,*a,**k:saved.append({n:x.clone() for n,x in p.items()})
 core.save_params_ckpt=core.save_params
 state=stable_trace.install(core,out);p,v=core.initialize_first_timestep()
 assert v['stable_id'].tolist()==[0,1,2]
 p,v=core.add_new_gaussians(p,v,None,.5,1,None,None)
 assert v['stable_id'].tolist()==[0,1,2,3,4]
 p,v=external.remove_points(torch.tensor([False,True,False,True,False]),p,v,None)
 assert v['stable_id'].tolist()==[0,2,4]
 p,v=core.add_new_gaussians(p,v,None,.5,2,None,None)
 assert v['stable_id'].tolist()==[0,2,4,5,6]
 core.save_params_ckpt(p,out,99)
 assert 'stable_id' not in p and saved[0]['stable_id'].tolist()==[0,2,4,5,6]
 assert saved[0]['timestep'].tolist()==[0,0,1,2,2]
 events=[json.loads(x) for x in (out/'pruning.jsonl').read_text().splitlines()]
 assert events[0]['removed']==2 and state['next_id']==7
 print('PASS: deleted IDs removed, new IDs not reused, snapshot alignment exact; params unchanged')
if __name__=='__main__':main()
