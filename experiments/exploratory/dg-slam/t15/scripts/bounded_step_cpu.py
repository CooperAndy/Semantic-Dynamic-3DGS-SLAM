"""Pure CPU mathematical core for proposed local L1 bounded updates.

This is preparation, not an image-renderer implementation or an experiment result.
"""
import numpy as np

def direction(gradient, P):
    g=np.asarray(gradient,dtype=float); P=np.asarray(P,dtype=float)
    assert g.shape==(6,) and P.shape==(6,6)
    assert np.isfinite(g).all() and np.isfinite(P).all()
    assert np.max(abs(P-P.T))<1e-10 and np.linalg.eigvalsh(P)[0]>0
    if np.linalg.norm(g)<=1e-12: return None
    d=-(P@g); d/=np.linalg.norm(d)
    assert g@d<0
    return d

def bounded_trial(gradient,P,current_loss,evaluate):
    """evaluate(d) scores a right-local SE3 step S*d against actual L1 support.

    No target pose is accepted. Caller commits only the returned accepted step.
    """
    d=direction(gradient,P)
    if d is None: return dict(accepted=False,reason='zero_gradient',attempts=0,step=np.zeros(6),loss=current_loss)
    for j in range(6):
        step=.05*(.5**j)*d; value=float(evaluate(step))
        if not np.isfinite(value): raise FloatingPointError('Nonfinite actual trial loss')
        if value<=current_loss+1e-4*float(np.dot(gradient,step)):
            return dict(accepted=True,reason='actual_Armijo_pass',attempts=j+1,step=step,loss=value)
    return dict(accepted=False,reason='six_rejections_stop_case',attempts=6,step=np.zeros(6),loss=current_loss)
