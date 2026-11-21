"""Prepared GPU forward/backward smoke; NOT executed during CPU preparation."""
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tum_window import ROOT

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(1219)
    source=ROOT/'01_source_env/DG-SLAM'
    sys.path[:0]=[str(source),str(source/'dg_slam')]
    from lietorch import SE3
    from torch_scatter import scatter_sum
    from dg_slam.modules.corr import CorrBlock,AltCorrBlock
    import faiss
    import faiss.contrib.torch_utils
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    x=torch.tensor([[.1,.2,.3,.0,.0,.1]],device='cuda',requires_grad=True)
    actual=SE3.exp(x).log()
    assert torch.allclose(actual,x,atol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    report['lie_gpu_forward_backward']='PASS'
    a=scatter_sum(torch.tensor([1.,2.,3.],device='cuda'),torch.tensor([0,1,0],device='cuda'))
    assert torch.equal(a.cpu(),torch.tensor([4.,2.]))
    report['scatter_gpu']='PASS'
    resources=faiss.StandardGpuResources()
    index=faiss.index_cpu_to_gpu(resources,0,faiss.IndexFlatL2(3))
    points=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],device='cuda')
    index.add(points)
    distances,indices=index.search(points,1)
    assert torch.equal(indices.cpu().flatten(),torch.arange(3)) and torch.allclose(distances,torch.zeros_like(distances))
    report['faiss_gpu_torch_search']='PASS'
    ivf=faiss.index_cpu_to_gpu(resources,0,faiss.IndexIVFFlat(faiss.IndexFlatL2(3),3,4,faiss.METRIC_L2))
    training=torch.rand(512,3,device='cuda')
    ivf.train(training)
    ivf.add(training)
    ivf.nprobe=4
    d,i=ivf.search(training[:16],1)
    assert torch.equal(i.cpu().flatten(),torch.arange(16)) and torch.allclose(d,torch.zeros_like(d),atol=1e-5)
    report['faiss_gpu_IVF_train_add_search']='PASS'
    f1=torch.rand(1,1,32,16,16,device='cuda',requires_grad=True)
    f2=torch.rand_like(f1,requires_grad=True)
    yy,xx=torch.meshgrid(torch.arange(16,device='cuda'),torch.arange(16,device='cuda'),indexing='ij')
    coords=torch.stack([xx,yy],-1).float()[None,None]
    corr=CorrBlock(f1,f2)(coords)
    corr.mean().backward()
    assert torch.isfinite(corr).all() and torch.isfinite(f1.grad).all() and torch.isfinite(f2.grad).all()
    report['droid_correlation_forward_backward']='PASS'
    afmaps=torch.cat([f1.detach(),f2.detach()],dim=1).requires_grad_()
    alt=AltCorrBlock(afmaps)(coords,torch.tensor([0],device='cuda'),torch.tensor([1],device='cuda'))
    assert torch.allclose(alt,corr.detach(),atol=2e-5,rtol=2e-4)
    alt.mean().backward()
    assert torch.isfinite(afmaps.grad).all()
    assert torch.allclose(afmaps.grad[:,0:1],f1.grad,atol=2e-5,rtol=2e-4)
    assert torch.allclose(afmaps.grad[:,1:2],f2.grad,atol=2e-5,rtol=2e-4)
    report['droid_legacy_altcorr_forward_backward']='PASS'
    report['altcorr_vs_volume_max_error']=float((alt-corr.detach()).abs().max())
    # Exercise the author-compatible BA signature on a known two-camera problem.
    import droid_backends
    import dg_slam.geom.projective_ops as pops
    poses=torch.tensor([[0.,0.,0.,0.,0.,0.,1.],[.08,0.,0.,0.,0.,0.,1.]],device='cuda')
    truth=poses.clone();truth[1,0]=.1
    disps=torch.ones(2,8,8,device='cuda')
    intrinsics=torch.tensor([8.,8.,4.,4.],device='cuda')
    ii=torch.tensor([0,1],device='cuda');jj=torch.tensor([1,0],device='cuda')
    def project(p):
        return pops.projective_transform(SE3(p[None]),disps[None],intrinsics.repeat(2,1)[None],ii,jj)[0]
    target=project(truth)
    before=float((project(poses)-target).square().mean())
    droid_backends.ba(poses,disps,intrinsics,disps.clone(),target[0].permute(0,3,1,2).contiguous(),torch.ones(2,2,8,8,device='cuda'),torch.full((2,8,8),1e-3,device='cuda'),ii,jj,1,2,2,1e-4,.1,True)
    after=float((project(poses)-target).square().mean())
    assert torch.isfinite(poses).all() and torch.isfinite(disps).all() and after<before
    distance=droid_backends.frame_distance(poses,disps,intrinsics,ii,ii,.3)
    assert torch.isfinite(distance).all() and distance.abs().max()<1e-6
    report.update(droid_BA_and_frame_distance='PASS',BA_reprojection_mse_before=before,BA_reprojection_mse_after=after)
    from diff_gaussian_rasterization_depth_pose import GaussianRasterizationSettings,GaussianRasterizer
    from dg_slam.gaussian.graphics_utils import getProjectionMatrix
    xyz=torch.tensor([[-.2,.0,2.],[.15,.1,2.3],[.0,-.2,2.7]],device='cuda',requires_grad=True)
    colors=torch.tensor([[.9,.2,.1],[.2,.9,.1],[.1,.3,.9]],device='cuda',requires_grad=True)
    opacity=torch.full((3,1),.7,device='cuda',requires_grad=True)
    scales=torch.full((3,3),.07,device='cuda',requires_grad=True)
    rotations=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda',requires_grad=True)
    means2d=torch.zeros_like(xyz,requires_grad=True)
    view=torch.eye(4,device='cuda',requires_grad=True)
    projection=view @ getProjectionMatrix(.01,100.,1.,1.).T.cuda()
    settings=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=float(np.tan(.5)),tanfovy=float(np.tan(.5)),bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=view,projmatrix=projection,sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
    image,radii,depth,acc=GaussianRasterizer(settings)(means3D=xyz,means2D=means2d,opacities=opacity,colors_precomp=colors,scales=scales,rotations=rotations)
    assert (radii>0).any() and torch.isfinite(image).all() and torch.isfinite(depth).all()
    loss=(image*torch.linspace(0.,1.,32,device='cuda')[None,None,:]).mean()+depth.mean()*.01
    loss.backward()
    for name,tensor in [('xyz',xyz),('opacity',opacity),('viewmatrix',view)]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(),name
        report[name+'_gradient_norm']=float(tensor.grad.norm())
    assert report['viewmatrix_gradient_norm']>0
    report['dg_rasterizer_forward_backward_pose_gradient']='PASS'
    torch.cuda.synchronize()
    report['status']='PASS'
    destination=ROOT/'03_gpu_smoke'/f'gpu_sanity_{__import__("time").strftime("%Y%m%d_%H%M%S")}.json'
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
