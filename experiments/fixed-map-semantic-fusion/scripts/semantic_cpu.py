"""CPU semantic evidence, alpha compositing reference, and scoring contracts.

This is not a replacement for the native Gaussian rasterizer. The compositing
function consumes depth-sorted per-pixel alpha contributions produced upstream.
"""
import json
from pathlib import Path
import numpy as np

UNKNOWN = -1
NO_MAP = -2

class Evidence:
    def __init__(self, ids, classes):
        self.ids=np.asarray(ids,dtype=np.int64)
        if len(np.unique(self.ids))!=len(self.ids): raise ValueError('duplicate Gaussian identity')
        self.evidence=np.zeros((len(self.ids),classes),dtype=np.float32)
        self.support=np.zeros(len(self.ids),dtype=np.uint32)
        self.total_weight=np.zeros(len(self.ids),dtype=np.float64)
        self.processed={}

    def update_frame(self, source_frame, version, distributions, weights):
        """One aggregated observation per Gaussian per distinct source frame.

        source_frame includes dataset/sequence/camera/frame. A replay of the
        same version is a no-op; a changed version must start another run.
        Validate the entire frame before committing; checkpoints are atomic.
        """
        key=json.dumps(source_frame,sort_keys=True,separators=(',',':'))
        if key in self.processed:
            if self.processed[key]!=version: raise ValueError('input version changed within run')
            return False
        d=np.asarray(distributions,dtype=np.float32);w=np.asarray(weights,dtype=np.float64)
        if d.shape!=self.evidence.shape or w.shape!=self.support.shape:raise ValueError('shape mismatch')
        if not np.isfinite(d).all() or not np.isfinite(w).all() or (d<0).any() or (w<0).any():raise ValueError('invalid evidence')
        active=w>0
        if not np.allclose(d[active].sum(1),1,atol=1e-5):raise ValueError('frame distributions must sum to one')
        self.evidence += (d*w[:,None]).astype(np.float32)
        self.support += active.astype(np.uint32)
        self.total_weight += w
        self.processed[key]=version
        return True

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,gaussian_id=self.ids,evidence=self.evidence,
            support_count=self.support,total_weight=self.total_weight,
            processed_json=np.array(json.dumps(self.processed,sort_keys=True)))
        temporary.replace(path)

    @classmethod
    def load(cls,path,expected_ids):
        with np.load(path,allow_pickle=False) as p:
            if not np.array_equal(p['gaussian_id'],expected_ids):raise ValueError('Gaussian identity mismatch')
            s=cls(p['gaussian_id'],p['evidence'].shape[1])
            s.evidence=p['evidence'].copy();s.support=p['support_count'].copy();s.total_weight=p['total_weight'].copy()
            s.processed=json.loads(str(p['processed_json']))
        return s

def aggregate_pixels(gaussian_indices,labels,alpha_contributions,n_gaussians,n_classes):
    """Candidate shared rule: normalize alpha-weighted hard votes within frame.

    Per-frame support budget = 1 for a Gaussian with positive valid support;
    pixel area alone cannot multiply the evidence budget. All groups share it.
    Invalid class IDs are excluded before association, not mapped to a class.
    """
    g=np.asarray(gaussian_indices);c=np.asarray(labels);a=np.asarray(alpha_contributions,dtype=float)
    if g.shape!=c.shape or a.shape!=g.shape:raise ValueError('shape mismatch')
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('invalid alpha')
    if ((g<0)|(g>=n_gaussians)).any():raise ValueError('invalid Gaussian index')
    votes=np.zeros((n_gaussians,n_classes),dtype=np.float64)
    valid=(c>=0)&(c<n_classes)&(a>0)
    np.add.at(votes,(g[valid],c[valid]),a[valid])
    totals=votes.sum(1);active=totals>0
    votes[active]/=totals[active,None]
    return votes.astype(np.float32),active.astype(float)

def factor_weights(method,base_valid,semantic,geometric,hard_threshold=0.5):
    b=np.asarray(base_valid,dtype=bool);s=np.asarray(semantic,float);g=np.asarray(geometric,float)
    if s.shape!=b.shape or g.shape!=b.shape:raise ValueError('shape mismatch')
    if not np.isfinite(s).all() or not np.isfinite(g).all() or ((s<0)|(s>1)|(g<0)|(g>1)).any():raise ValueError('invalid reliability')
    values={'S0':np.ones_like(s),'S1':s,'S-G':g,'S2':s*g,'S2-H':(s*g>=hard_threshold).astype(float)}
    if method not in values:raise ValueError('unknown method')
    return b.astype(float)*values[method]

def composite(alpha,class_distributions,unknown,alpha_min=0.05,unknown_fraction_max=0.5):
    """Reference front-to-back alpha compositing; input already depth-sorted.

    Keeps unknown opacity. Returns class mass, unknown mass, alpha support,
    output label and uncalibrated score. Thresholds are engineering defaults.
    """
    a=np.asarray(alpha,float);d=np.asarray(class_distributions,float);u=np.asarray(unknown,bool)
    if a.ndim!=1 or d.ndim!=2 or d.shape[0]!=len(a) or u.shape!=a.shape:raise ValueError('shape mismatch')
    if not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('invalid alpha')
    if not np.isfinite(d).all() or (d<0).any() or not np.allclose(d[~u].sum(1),1):raise ValueError('invalid class distribution')
    trans=np.concatenate(([1.],np.cumprod(1-a)[:-1]));contrib=trans*a
    mass=(contrib[:,None]*np.where(u[:,None],0,d)).sum(0)
    unknown_mass=float(contrib[u].sum());total=float(contrib.sum())
    label=NO_MAP if total<alpha_min else UNKNOWN
    if total>=alpha_min and mass.sum()>0 and unknown_mass/total<unknown_fraction_max:
        label=int(mass.argmax())
    score=float(mass.max()/total) if total>0 else 0.
    return {'class_mass':mass.tolist(),'unknown_mass':unknown_mass,'alpha':total,'label':label,'score':score}

def score_semantics(gt,pred,classes,ignore=255):
    gt=np.asarray(gt);pred=np.asarray(pred)
    if gt.shape!=pred.shape:raise ValueError('shape mismatch')
    valid=gt!=ignore
    if ((gt[valid]<0)|(gt[valid]>=classes)).any():raise ValueError('GT outside class mapping')
    if ((pred[valid]<NO_MAP)|(pred[valid]>=classes)).any():raise ValueError('prediction outside class mapping')
    g=gt[valid];p=pred[valid];rows=[]
    for c in range(classes):
        tp=int(((g==c)&(p==c)).sum());fn=int(((g==c)&(p!=c)).sum());fp=int(((g!=c)&(p==c)).sum())
        union=tp+fn+fp
        rows.append({'class':c,'tp':tp,'fn':fn,'fp':fp,'gt_support':int((g==c).sum()),'iou':tp/union if union else None})
    present=[r['iou'] for r in rows if r['gt_support']>0]
    labelled=p>=0
    return {'metric':'map-rendered 2D semantic mIoU','pixels':len(g),'classes':rows,
        'miou_gt_present_classes':float(np.mean(present)) if present else None,
        'coverage':float(labelled.mean()) if len(g) else None,
        'unknown':int((p==UNKNOWN).sum()),'no_map_support':int((p==NO_MAP).sum()),
        'conditional_accuracy':float((p[labelled]==g[labelled]).mean()) if labelled.any() else None}

def risk_coverage(gt,pred,scores,ignore=255):
    gt=np.asarray(gt);pred=np.asarray(pred);scores=np.asarray(scores,float)
    if gt.shape!=pred.shape or scores.shape!=gt.shape:raise ValueError('shape mismatch')
    if not np.isfinite(scores).all():raise ValueError('invalid ranking score')
    valid=gt!=ignore;eligible=valid&(pred>=0);n=int(valid.sum())
    rows=[{'coverage':0.,'risk':None,'accepted':0}]
    for threshold in np.unique(scores[eligible])[::-1]:
        selected=eligible&(scores>=threshold);count=int(selected.sum())
        rows.append({'threshold':float(threshold),'accepted':count,'coverage':count/n if n else None,
                     'risk':float((pred[selected]!=gt[selected]).mean())})
    return rows

def native_render_variables(params,transformed,channels,zeros_like,normalize,exp,sigmoid):
    """Native adapter for THREE numeric channels per pass; never mask opacity.

    Caller supplies tensor functions (CPU stubs in tests, torch in GPU runs).
    For 19 classes plus unknown, pad groups to 3, use zero renderer background,
    keep identical opacity in every pass, then apply frozen pixel decisions.
    No palette conversion is used for class decisions.
    """
    scales=params['log_scales']
    if scales.shape[1]==1:scales=scales.repeat(3,axis=1) if isinstance(scales,np.ndarray) else scales.repeat(1,3)
    return {'means3D':transformed['means3D'],'means2D':zeros_like(params['means3D']),
        'colors_precomp':channels,'rotations':normalize(transformed['unnorm_rotations']),
        'scales':exp(scales),'opacities':sigmoid(params['logit_opacities'])}
