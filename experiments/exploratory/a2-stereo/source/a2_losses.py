"""A2 exploratory stereo residual scales; confidence is not calibrated variance."""
import torch

def residual_scale(depth,confidence,fb,mode):
 if fb<=0 or not torch.isfinite(torch.as_tensor(fb)):raise ValueError('invalid f*b')
 if not torch.isfinite(depth).all() or (depth<=0).any():raise ValueError('invalid depth')
 if mode=='fixed':return torch.full_like(depth,.3)
 scale=(depth.square()/fb).clamp(.1,3.)
 if mode=='distance':return scale
 if mode!='confidence':raise ValueError('unknown scale mode')
 if confidence is None or not torch.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('invalid confidence')
 return (scale*(1+2*(1-confidence))).clamp(.1,3.)

def robust_loss(residual,depth,confidence,fb,mode,gamma=1.):
 if not torch.isfinite(residual).all() or gamma<=0:raise ValueError('invalid residual or gamma')
 u=residual/residual_scale(depth,confidence,fb,mode);a=u.abs()
 rho=torch.where(a<=1,.5*u.square(),a-.5)
 return gamma*.3*rho.mean()
