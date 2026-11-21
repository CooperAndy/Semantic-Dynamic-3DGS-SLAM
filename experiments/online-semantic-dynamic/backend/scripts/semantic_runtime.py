"""Online lifecycle and geometry-only M gradients; installed into private native runs."""
import json,time,functools
from pathlib import Path
import numpy as np
import torch
from online_state import OnlineState
from semantic_constraints import Flags,consistent_matches,shuffled_labels
from semantic_mapping import render_channels,initial_mask,semantic_gradients,fuse_native

ACTIVE=None


class Runtime:
    def __init__(self,core,flags,provider,out,holdout,labels=True,shuffle=False,shuffle_seed=0):
        self.core=core;self.flags=flags;self.provider=provider;self.out=Path(out)
        self.holdout=set(holdout);self.labels_enabled=bool(labels or flags.M)
        self.shuffle=bool(shuffle);self.shuffle_seed=int(shuffle_seed)
        self.state=OnlineState();self.pixel_mask=None;self.label_tensor=None;self.pending=None
        self.log=(self.out/'semantic_events.jsonl').open('x',buffering=1)
        self.mapping_calls=0;self.gradient_calls=0;self.last_semantic_loss=0.
        self.timing_seconds={}

    def event(self,kind,**kw):
        self.log.write(json.dumps(dict(kind=kind,frame=self.state.frame,**kw),allow_nan=False)+'\n')

    def allow(self,frame):
        if not self.flags.D:return None
        return self.provider.read(frame)['state']!=2

    def match(self,reference,current,xy,query):
        if not self.flags.T:return np.ones(len(xy),bool),dict(enabled=False,before=len(xy),after=len(xy),removed=0)
        keep,report=consistent_matches(self.provider.read(reference),self.provider.read(current),xy,query)
        report.update(enabled=True,reference_frame=reference,current_frame=current)
        self.event('T_matches',**report);return keep,report

    def features(self,params):
        if len(params['means3D'])!=len(self.state.ids):raise ValueError('Gaussian semantic order desynchronized')
        if self.state.cache is None:self.state.cache=torch.as_tensor(self.state.features(),device=params['means3D'].device)
        return self.state.cache

    def begin(self,params,data,frame):
        self.state.begin(frame);self.pixel_mask=None;self.label_tensor=None;self.pending=None
        if not self.flags.M or frame in self.holdout:return
        obs=self.provider.read(frame);device=params['means3D'].device
        self.label_tensor=torch.as_tensor(obs['category'].astype(np.int64),device=device)
        confidence=torch.as_tensor(obs['confidence'],device=device)
        allow=self.allow(frame)
        if allow is not None:allow=torch.as_tensor(allow,device=device)
        with torch.no_grad():
            features=self.features(params)
            if not bool((features.sum()>0).item()):self.pixel_mask=torch.zeros_like(self.label_tensor,dtype=torch.bool)
            else:
                mass=render_channels(self.core,params,data,frame,features,False)
                alpha=render_channels(self.core,params,data,frame,params['rgb_colors'].new_ones((len(features),3)),False)[0]
                self.pixel_mask=initial_mask(mass,alpha,self.label_tensor,confidence,data['depth'].squeeze(0),allow)
        if self.shuffle:
            lab=shuffled_labels(obs['category'],self.pixel_mask.cpu().numpy(),self.shuffle_seed+frame)
            self.label_tensor=torch.as_tensor(lab.astype(np.int64),device=device)
        self.event('M_frozen_support',pixels=int(self.pixel_mask.sum()),prior_frames=sorted(int(x) for x in self.state.processed),shuffle=self.shuffle)

    def prepare_loss(self,params,data,target):
        self.pending=None
        if not self.flags.M or target!=self.state.frame or self.pixel_mask is None:return
        if target in self.holdout:raise ValueError('Heldout frame entering semantic loss')
        self.mapping_calls+=1
        self.pending,self.last_semantic_loss=semantic_gradients(self.core,params,data,target,self.features(params),self.label_tensor,self.pixel_mask)
        if self.pending is not None:self.gradient_calls+=1
        self.event('M_loss',target=int(target),pixels=int(self.pixel_mask.sum()),loss=self.last_semantic_loss,
                   gradient_keys=[] if self.pending is None else [k for k,v in self.pending.items() if v is not None])

    def after_backward(self,params):
        if self.pending is None:return
        with torch.no_grad():
            for k,g in self.pending.items():
                if g is not None:
                    if params[k].grad is None:params[k].grad=g
                    else:params[k].grad.add_(g)
        self.pending=None

    def finish_frame(self,params,data,frame):
        if frame!=self.state.frame:raise ValueError('Semantic end frame mismatch')
        if frame in self.holdout or not self.labels_enabled:return
        obs=self.provider.read(frame);allow=self.allow(frame)
        if allow is not None:allow=torch.as_tensor(allow,device=params['means3D'].device)
        distribution,weights,error=fuse_native(self.core,params,data,frame,obs['category'],allow)
        assert self.state.fuse(frame,distribution,weights)
        assert not self.state.fuse(frame,distribution,weights)
        self.event('fusion',associated=int((weights>0).sum()),adjoint_mass_relative_error=error,duplicate_replay_noop=True)

    def install_lifecycle(self):
        import utils.slam_external as se
        core=self.core;initial=core.initialize_params;new=core.initialize_new_params
        def initialize(*a,**kw):
            if len(a[0])==0:raise RuntimeError('NO_STATIC_INITIAL_GAUSSIANS')
            result=initial(*a,**kw);self.state.birth(len(result[0]['means3D']));return result
        def born(*a,**kw):
            result=new(*a,**kw);self.state.birth(len(result['means3D']));return result
        remove=se.remove_points;cat=se.cat_params_to_optimizer
        def prune(mask,*a,**kw):
            copy=mask.detach().cpu().numpy().reshape(-1).copy();result=remove(mask,*a,**kw)
            self.state.prune(copy);self.event('prune',removed=int(copy.sum()),remaining=len(self.state.ids));return result
        def densify(new_params,*a,**kw):
            result=cat(new_params,*a,**kw);ids=self.state.birth(len(new_params['means3D']))
            self.event('densify_unknown',added=len(ids));return result
        core.initialize_params=initialize;core.initialize_new_params=born
        se.remove_points=prune;se.cat_params_to_optimizer=densify
        save=core.save_params;checkpoint=core.save_params_ckpt
        def final(params,path):
            save(params,path);self.state.save(Path(path)/'semantic_state_final.npz')
            np.save(Path(path)/'source_gaussian_index.npy',self.state.ids,allow_pickle=False)
        def ckpt(params,path,index):
            checkpoint(params,path,index);self.state.save(Path(path)/f'semantic_state_{index}.npz')
        core.save_params=final;core.save_params_ckpt=ckpt

    def close(self):
        self.log.close()
        receipt=dict(status='ONLINE_STATE_FINISHED',T=self.flags.T,M=self.flags.M,D=self.flags.D,
                     gaussian_count=len(self.state.ids),known_count=int((self.state.votes.sum(1)>0).sum()),
                     source_frames=sorted(int(x) for x in self.state.processed),mapping_semantic_calls=self.mapping_calls,
                     semantic_gradient_calls=self.gradient_calls,online_lifecycle=True,
                     timing_seconds=self.timing_seconds,timing_scope='Synchronized host elapsed; fusion includes native semantic renders, do not add render subtotal')
        (self.out/'semantic_receipt.json').write_text(json.dumps(receipt,indent=2))


def timed_stage(name):
    def decorate(method):
        @functools.wraps(method)
        def wrapped(self,*args,**kwargs):
            cuda=torch.cuda.is_initialized()
            if cuda:torch.cuda.synchronize()
            start=time.perf_counter()
            try:return method(self,*args,**kwargs)
            finally:
                if cuda:torch.cuda.synchronize()
                self.timing_seconds[name]=self.timing_seconds.get(name,0.)+time.perf_counter()-start
        return wrapped
    return decorate

Runtime.begin=timed_stage('semantic_support')(Runtime.begin)
Runtime.prepare_loss=timed_stage('semantic_loss_gradient')(Runtime.prepare_loss)
Runtime.finish_frame=timed_stage('semantic_fusion')(Runtime.finish_frame)

def install(core,flags,provider,out,holdout,**kwargs):
    global ACTIVE
    ACTIVE=Runtime(core,flags,provider,out,holdout,**kwargs);ACTIVE.install_lifecycle();return ACTIVE
