"""Differentiable SE(3) exponential; evaluate smooth coefficients in float64."""
import torch
def skew(v):
    x,y,z=v.unbind();o=x*0
    return torch.stack([o,-z,y,z,o,-x,-y,x,o]).reshape(3,3)
def exp_se3(xi):
    rho,w=xi[:3],xi[3:];W=skew(w);theta2=(w*w).sum()
    # The inactive trigonometric branch also remains finite at exactly zero.
    theta=torch.sqrt(torch.clamp(theta2,min=1e-12))
    A=torch.where(theta2<1e-8,1-theta2/6+theta2*theta2/120,torch.sin(theta)/theta)
    B=torch.where(theta2<1e-8,.5-theta2/24+theta2*theta2/720,(1-torch.cos(theta))/(theta*theta))
    C=torch.where(theta2<1e-8,1/6-theta2/120+theta2*theta2/5040,(theta-torch.sin(theta))/(theta*theta*theta))
    I=torch.eye(3,dtype=xi.dtype,device=xi.device);R=I+A*W+B*(W@W);V=I+B*W+C*(W@W)
    top=torch.cat([R,(V@rho)[:,None]],dim=1)
    return torch.cat([top,xi.new_tensor([[0.,0.,0.,1.]])],dim=0)
def se3_camera(u,anchor,P,scale):
    # Raw optimizer leaves/momenta float32 in all arms. Lie algebra arithmetic
    # is float64 in every SE3 arm; cast the same native renderer input float32.
    return (anchor.double()@exp_se3(scale*(P@u.double()))).float()
