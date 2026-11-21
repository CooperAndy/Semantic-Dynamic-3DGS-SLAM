"""Independent T/M/D flags and reproducible match-level attribution."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Flags:
    T: bool=False
    M: bool=False
    D: bool=False

    def __post_init__(self):
        if any(type(x) is not bool for x in (self.T,self.M,self.D)):
            raise TypeError('T/M/D must be independent booleans')


def sample(obs, pixels):
    pixels=np.asarray(pixels,np.float64)
    if pixels.ndim!=2 or pixels.shape[1]!=2 or not np.isfinite(pixels).all():
        raise ValueError('Expected finite Nx2 pixel coordinates')
    category,confidence=obs['category'],obs['confidence']
    if category.ndim!=2 or confidence.shape!=category.shape:
        raise ValueError('Semantic grid mismatch')
    h,w=category.shape;xy=np.floor(pixels+.5).astype(np.int64)
    inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    out=np.full(len(pixels),-1,np.int16);score=np.zeros(len(pixels),np.float32)
    out[inside]=category[xy[inside,1],xy[inside,0]]
    score[inside]=confidence[xy[inside,1],xy[inside,0]]
    return out,score


def consistent_matches(reference,current,xy,query,threshold=.8):
    a,ca=sample(reference,xy);b,cb=sample(current,query)
    trusted=(a>=0)&(a<19)&(b>=0)&(b<19)&(ca>=threshold)&(cb>=threshold)
    reject=trusted&(a!=b)
    pairs,counts=np.unique(np.column_stack([a[reject],b[reject]]),axis=0,return_counts=True)
    report=dict(before=len(a),after=int((~reject).sum()),removed=int(reject.sum()),
                trusted=int(trusted.sum()),category_pairs=[dict(reference=int(p[0]),current=int(p[1]),count=int(n)) for p,n in zip(pairs,counts)])
    return ~reject,report


def matched_random_keep(semantic_keep,query,shape,seed):
    """Delete the same count in each 4x4 TARGET cell, without using GT."""
    keep=np.asarray(semantic_keep,bool);q=np.asarray(query,np.float64)
    if q.shape!=(len(keep),2):raise ValueError('Mismatch')
    h,w=shape;cells=np.clip((q[:,1]/h*4).astype(int),0,3)*4+np.clip((q[:,0]/w*4).astype(int),0,3)
    rng=np.random.default_rng(int(seed));out=np.ones(len(keep),bool)
    for c in range(16):
        ids=np.flatnonzero(cells==c);n=int((~keep[ids]).sum())
        if n:out[rng.choice(ids,n,replace=False)]=False
    return out


def shuffled_labels(category,valid,seed):
    result=np.array(category,copy=True);valid=np.asarray(valid,bool)
    if result.shape!=valid.shape:raise ValueError('Mask shape mismatch')
    result[valid]=np.random.default_rng(int(seed)).permutation(result[valid])
    return result
