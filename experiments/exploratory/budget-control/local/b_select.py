"""Experimental fixed-budget ranking on native G07 candidate pixels."""
import time
import numpy as np
import torch

def select(candidate,observed,rendered,silhouette,rgb,K,cap,method,seed,index):
    t0=time.monotonic(); H,W=observed.shape
    ids=np.flatnonzero(candidate.detach().cpu().numpy().ravel()); n=len(ids)
    if n<cap: raise RuntimeError(f'insufficient_candidates {index}: {n} < {cap}')
    if method=='N0':
        ranks=np.random.default_rng(np.random.SeedSequence([seed,index])).random(H*W,dtype=np.float32)
        chosen=ids[np.argpartition(ranks[ids],cap-1)[:cap]]
        return np.sort(chosen),dict(selector=method,candidates=n,quota=cap,seconds=time.monotonic()-t0)
    image=rgb.detach();gx=torch.zeros_like(image);gy=torch.zeros_like(image)
    gx[:,:,1:-1]=(image[:,:,2:]-image[:,:,:-2])/2;gy[:,1:-1,:]=(image[:,2:,:]-image[:,:-2,:])/2
    z=(rendered/silhouette.clamp_min(1e-6)).detach(); valid=(silhouette>.99)&torch.isfinite(z)&(z>.5)&(z<80)
    grid=torch.zeros_like(valid);grid[::4,::4]=True; now=(valid&grid).flatten().nonzero().flatten()
    if len(now)<500:raise RuntimeError(f'insufficient_existing_support frame={index} n={len(now)}')
    c=torch.as_tensor(ids,device=observed.device,dtype=torch.long);ob=observed.flatten()[c]
    if not torch.isfinite(ob).all() or not (ob>0).all():raise RuntimeError('invalid candidate depth')
    L=z.flatten()[now].median().item(); L=max(L,.5)
    S=torch.tensor([1.,1.,1.,1/L,1/L,1/L],device=observed.device)
    fx,fy,cx,cy=[float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2])]
    def jac(ind,depth):
        y=torch.div(ind,W,rounding_mode='floor').float();x=(ind%W).float(); X=(x-cx)/fx;Y=(y-cy)/fy
        jx=torch.stack((fx/depth,torch.zeros_like(X),-fx*X/depth,-fx*X*Y,fx*(1+X*X),-fx*Y),1)
        jy=torch.stack((torch.zeros_like(X),fy/depth,-fy*Y/depth,-fy*(1+Y*Y),fy*X*Y,fy*X),1)
        return (gx.reshape(3,-1)[:,ind].T[:,:,None]*jx[:,None,:]+gy.reshape(3,-1)[:,ind].T[:,:,None]*jy[:,None,:])*S
    j0=jac(now,z.flatten()[now]);hm=torch.einsum('nci,ncj->ij',j0,j0)/len(now)
    ridge=hm.trace()*1e-3/6
    if not torch.isfinite(hm).all() or ridge<=0:raise RuntimeError('invalid information matrix')
    jc=jac(c,ob); inv=torch.linalg.inv(hm+ridge*torch.eye(6,device=hm.device))
    info=torch.einsum('nci,ij,ncj->n',jc,inv,jc)
    grad=(gx.abs().mean(0)+gy.abs().mean(0)).flatten()[c]/2
    scores=(info if method=='B' else grad).detach().cpu().numpy();depth=ob.detach().cpu().numpy()
    if not np.isfinite(scores).all():raise RuntimeError('nonfinite selection score')
    # Reserve 20% of budget by selecting in rounds from occupied 32px/depth bins.
    bins=np.digitize(depth,[10.,25.]); yy=ids//W;xx=ids%W
    groups=((yy//32)*((W+31)//32)+(xx//32))*3+bins
    order=np.lexsort((ids,-scores)); ordered_groups=groups[order]
    _,first=np.unique(ordered_groups,return_index=True); picked=np.zeros(n,dtype=bool)
    reserve=min(600,cap); first=np.sort(first)
    for i in first[:reserve]:picked[order[i]]=True
    # If fewer than 600 bins exist, fill the reserved quota by repeated rounds.
    while picked.sum()<reserve:
        left=order[~picked[order]];_,first2=np.unique(groups[left],return_index=True)
        take=left[np.sort(first2)[:reserve-int(picked.sum())]]
        if not len(take):break
        picked[take]=True
    rest=order[~picked[order]][:cap-int(picked.sum())];picked[rest]=True
    chosen=ids[picked];assert len(chosen)==cap and len(np.unique(chosen))==cap
    torch.cuda.synchronize()
    return np.sort(chosen),dict(selector=method,candidates=n,quota=cap,coverage_reserve=reserve,current_support=len(now),scale_m=L,occupied_bins=int(len(first)),chosen_bins=int(np.unique(groups[picked]).size),seconds=time.monotonic()-t0)
