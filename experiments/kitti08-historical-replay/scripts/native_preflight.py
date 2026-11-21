from pathlib import Path
import sys,json,hashlib,subprocess,time
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'source'))
import torch,numpy as np
import diff_gaussian_rasterization as rast
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation
assert torch.cuda.get_device_capability(0)==(12,0)
assert torch.__version__=='2.8.0+cu128' and np.__version__=='1.26.4'
so=Path(rast._C.__file__)
digest=hashlib.sha256(so.read_bytes()).hexdigest()
assert so.resolve().is_relative_to(Path('/root/miniconda3/lib/python3.12/site-packages'))
elf=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--list-elf',str(so)],capture_output=True,text=True,timeout=20)
assert elf.returncode==0 and 'sm_120' in elf.stdout
W,H=64,48
cam=setup_camera(W,H,np.array([[50.,0,32.],[0,50.,24.],[0,0,1.]]),np.eye(4))
base=torch.tensor([[-.3,-.1,3.],[.2,.3,3.4],[.6,-.2,4.]],device='cuda')
q=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True)
t=torch.tensor([[.03,.01,0.]],device='cuda',requires_grad=True)
means=(build_rotation(q)[0]@base.T).T+t
kw=dict(means3D=means,means2D=torch.zeros_like(means,requires_grad=True),shs=None,
        colors_precomp=torch.tensor([[1.,.2,.3],[.1,1.,.2],[.2,.3,1.]],device='cuda'),
        opacities=torch.full((3,1),.8,device='cuda'),scales=torch.full((3,3),.12,device='cuda'),
        rotations=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda'),cov3D_precomp=None)
started=time.monotonic()
rgb,radii,depth=rast.GaussianRasterizer(raster_settings=cam)(**kw)
assert rgb.shape==(3,H,W) and depth.shape==(1,H,W) and (radii>0).all()
loss=(rgb*torch.linspace(0,1,W,device='cuda')[None,None,:]).sum()+.01*depth.sum()
loss.backward();torch.cuda.synchronize()
assert torch.isfinite(rgb).all() and torch.isfinite(depth).all()
assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in [q,t])
r=dict(status='PASS_NATIVE_FORWARD_BACKWARD',python=sys.executable,GPU=torch.cuda.get_device_name(0),
       capability=list(torch.cuda.get_device_capability(0)),torch=torch.__version__,numpy=np.__version__,
       rasterizer=str(so),sha256=digest,elf=elf.stdout,q_gradient=q.grad.cpu().tolist(),t_gradient=t.grad.cpu().tolist(),
       seconds=time.monotonic()-started,scope='Synthetic3Gaussian hardware gate; no KITTI quality claim')
(R/'receipts/native_preflight.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))
