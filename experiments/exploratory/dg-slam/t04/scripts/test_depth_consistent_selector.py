"""CPU projection, parameter synchronization and fallback checks."""
from types import SimpleNamespace
from pathlib import Path
import torch
from depth_consistent_selector import select_gaussians,install_selector,InsufficientTrackingSupport

D=torch.full((3,5),5.0);K=(2.,2.,2.,1.)
camera=torch.tensor([[0.,0.,5.],[2.5,0.,5.],[0.,0.,5.7],[0.,0.,-5.],[100.,0.,5.],[float('nan'),0.,5.]])
expected=[True,True,False,False,False,False]
mask,stats=select_gaussians(camera,torch.eye(4),D,K);assert mask.tolist()==expected
# Translation and nontrivial rotation; this catches wrong pose direction/axis.
pose=torch.eye(4);pose[:3,:3]=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]);pose[:3,3]=torch.tensor([10.,2.,-4.])
world=camera@pose[:3,:3].T+pose[:3,3]
assert select_gaussians(world,pose,D,K)[0].tolist()==expected
bad=D.clone();bad[1,2]=float('nan');assert select_gaussians(camera,torch.eye(4),bad,K)[0].tolist()==[False,True,False,False,False,False]
assert select_gaussians(torch.empty(0,3),torch.eye(4),D,K)[0].numel()==0

# Hooks select all six synchronized tensors only during tracking render.
for force_support_failure in [False,True]:
    tensors=[camera.clone()]+[torch.arange(6).reshape(6,1).float()+10*i for i in range(1,6)]
    seen=[]
    def render(*args):seen.append([x.clone() for x in args]);return 'rendered'
    module=SimpleNamespace(render=render)
    mapper=SimpleNamespace(cfg={},H=3,W=5,fx=2.,fy=2.,cx=2.,cy=1.,
        gaussians=SimpleNamespace(get_xyz=lambda:camera),_t04_source_id=7,
        estimate_c2w_list=torch.zeros(4,4,4),idx=torch.zeros(1,dtype=torch.long))
    def optimize(*args):
        module.render(*tensors)
        if force_support_failure:raise InsufficientTrackingSupport('synthetic_support_failure')
        return (None,None,None,None,None,torch.ones(3,5,dtype=torch.bool))
    mapper.optimize_cam_in_batch=optimize
    def tracking(idx,color,depth,coarse,seg):
        mapper.optimize_cam_in_batch()
        mapper.estimate_c2w_list[idx]=coarse
    mapper.tracking=tracking
    out=Path(__file__).resolve().parents[1]/'02_Gaussian适配审计/GS_SLAM借鉴_20261012/stage_b_patch/cpu_checks'/str(force_support_failure)
    out.mkdir(parents=True,exist_ok=True)
    if True:
        install_selector(mapper,module,{7:D},out,min_gaussians=1,min_pixels=1)
        mapper.tracking(2,None,D,torch.eye(4),None)
        assert torch.equal(mapper.estimate_c2w_list[2],torch.eye(4))
        assert all(torch.equal(got,t[mask]) for got,t in zip(seen[0],tensors))
        assert torch.allclose(camera,tensors[0],equal_nan=True)
        module.render(*tensors)
        assert seen[-1][0].shape==(6,3), 'mapping/full render must retain all points'
        assert mapper._t04_render_selection_active is False
print('CPU_SELECTOR_PASS: projection, measured z, invalid observations, synchronized tensors, fallback, full-map preservation')
