import numpy as np
def opacity_retention(initial,candidate,mask):
 a=np.asarray(initial)[mask].astype(np.float64);b=np.asarray(candidate)[mask].astype(np.float64);assert len(a)>0 and np.isfinite(a).all() and np.isfinite(b).all();return float(np.clip(b,0,1).sum()/max(np.clip(a,0,1).sum(),1e-12))
