"""Prepared A2 factorial losses, same masks and residuals as mapper. No LiDAR inputs."""
import torch
GAMMA=0.7633236491505211
GAMMA_FIXED=1.0010791602334679

def scales(depth,confidence,fb,arm):
 if arm not in ('C2','C3','C4','C5'):raise ValueError('invalid factorial arm')
 if not torch.isfinite(torch.as_tensor(fb)).all() or fb<=0:raise ValueError('invalid fb')
 if not torch.isfinite(depth).all() or (depth<=0).any():raise ValueError('invalid depth')
 scale=(depth.square()/fb).clamp(.1,3.)
 if arm in ('C3','C5'):
  if confidence is None or confidence.shape!=depth.shape or not torch.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('invalid confidence')
  scale=(scale*(1+2*(1-confidence))).clamp(.1,3.)
 if arm in ('C4','C5'):scale=scale.clamp(max=1.5)
 return scale

def loss(residual,depth,confidence,fb,arm):
 if residual.shape!=depth.shape or residual.numel()==0 or not torch.isfinite(residual).all():raise ValueError('invalid residual')
 if arm=='C0':return residual.abs().mean()
 if arm=='C1':s=torch.full_like(depth,.3);gamma=GAMMA_FIXED
 else:s=scales(depth,confidence,fb,arm);gamma=GAMMA
 u=residual/s;a=u.abs();return gamma*.3*torch.where(a<=1,.5*u.square(),a-.5).mean()

