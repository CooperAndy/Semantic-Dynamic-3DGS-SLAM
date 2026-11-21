"""Dense CPU evidence in exact live Gaussian order, with immutable frame snapshot."""
import json
import numpy as np


class OnlineState:
    def __init__(self,classes=19):
        self.classes=int(classes);self.ids=np.empty(0,np.int64);self.next_id=0
        self.votes=np.zeros((0,self.classes),np.float32);self.frozen=self.votes.copy()
        self.support=np.empty(0,np.uint32);self.processed=set();self.frame=-1
        self.cache=None;self.generation=0

    def birth(self,count):
        count=int(count)
        if count<0:raise ValueError('Negative birth count')
        ids=np.arange(self.next_id,self.next_id+count,dtype=np.int64);self.next_id+=count
        self.ids=np.concatenate([self.ids,ids]);zero=np.zeros((count,self.classes),np.float32)
        self.votes=np.concatenate([self.votes,zero]);self.frozen=np.concatenate([self.frozen,zero])
        self.support=np.concatenate([self.support,np.zeros(count,np.uint32)])
        self.cache=None;self.generation+=1;return ids

    def prune(self,remove):
        remove=np.asarray(remove,bool).reshape(-1)
        if remove.shape!=self.ids.shape:raise ValueError('Prune identity mismatch')
        keep=~remove
        for name in ['ids','votes','frozen','support']:setattr(self,name,getattr(self,name)[keep])
        self.cache=None;self.generation+=1

    def begin(self,frame):
        frame=int(frame)
        if frame!=self.frame+1:raise ValueError('Nonconsecutive semantic frame')
        if any(int(k)>=frame for k in self.processed):raise ValueError('Future semantic evidence')
        self.frame=frame;self.frozen=self.votes.copy();self.cache=None

    def fuse(self,frame,distribution,weights):
        frame=int(frame);d=np.asarray(distribution,np.float32);w=np.asarray(weights,np.float32)
        if frame!=self.frame or d.shape!=self.votes.shape or w.shape!=self.ids.shape:
            raise ValueError('Semantic identity/frame mismatch')
        if not np.isfinite(d).all() or not np.isfinite(w).all() or (d<0).any() or (w<0).any() or (w>1).any():
            raise ValueError('Invalid semantic transaction')
        sums=d.sum(1);active=w>0
        if not np.allclose(sums[active],1,atol=1e-4):raise ValueError('Frame distribution must normalize')
        if str(frame) in self.processed:return False
        self.votes+=d*w[:,None];self.support+=active.astype(np.uint32);self.processed.add(str(frame))
        return True

    def features(self):
        total=self.frozen.sum(1,keepdims=True)
        return np.divide(self.frozen,total,out=np.zeros_like(self.frozen),where=total>0)

    def save(self,path):
        np.savez_compressed(path,ids=self.ids,votes=self.votes,frozen=self.frozen,support=self.support,
                            metadata=np.array(json.dumps(dict(classes=self.classes,next_id=self.next_id,frame=self.frame,
                                                              processed=sorted(self.processed),generation=self.generation))))

    @classmethod
    def load(cls,path,expected_ids=None):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));s=cls(meta['classes'])
            for k in ['ids','votes','frozen','support']:setattr(s,k,z[k].copy())
        if expected_ids is not None and not np.array_equal(s.ids,expected_ids):raise ValueError('Checkpoint Gaussian identity mismatch')
        if s.votes.shape!=(len(s.ids),s.classes) or len(np.unique(s.ids))!=len(s.ids):raise ValueError('Corrupt state')
        s.next_id=meta['next_id'];s.frame=meta['frame'];s.processed=set(meta['processed']);s.generation=meta['generation'];return s
