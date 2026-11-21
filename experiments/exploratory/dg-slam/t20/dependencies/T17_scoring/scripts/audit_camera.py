"""Independent CPU expm/Armijo replay, without reference poses or GT."""
import numpy as np
from scipy.linalg import expm
def hat(x):
 A=np.zeros((4,4));v,w=x[:3],x[3:];A[:3,3]=v;A[:3,:3]=[[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]];return A
def audit_trace(path,scale):
 with np.load(path) as f:z={k:a.copy() for k,a in f.items()}
 assert all(np.isfinite(a).all() for a in z.values())
 assert np.array_equal(z['initial_c2w'],z['pre_c2w'][0]) and np.array_equal(z['candidate_c2w'],z['post_c2w'][-1])
 assert np.array_equal(z['post_c2w'][:-1],z['pre_c2w'][1:])
 assert 1<=len(z['loss'])<=50
 C=z['C'];assert np.linalg.eigvalsh(C)[0]>0
 maximum=0.;proposals=0
 for k,(A,B,g,ok) in enumerate(zip(z['pre_c2w'],z['post_c2w'],z['gradients'],z['accepted'])):
  ix=np.flatnonzero(z['proposal_iteration']==k)
  if np.linalg.norm(g)<=1e-12:assert len(ix)==0 and not ok
  else:
   d=-(C@g);d/=np.linalg.norm(d)
   for j,i in enumerate(ix):
    step=z['proposal_step'][i];assert np.max(abs(step-.05*.5**j*d))<=1e-12
    predicted=A.astype(float)@expm(hat(np.array(scale)*step));error=float(np.max(abs(predicted-z['proposal_c2w'][i])));maximum=max(maximum,error)
    bound=z['loss'][k]+1e-4*float(g@step);assert abs(bound-z['proposal_armijo'][i])<=1e-8 and g@step<0 and np.linalg.norm(step)<=.05+1e-14
    if j<len(ix)-1 or not ok:assert z['proposal_loss'][i]>bound
   if ok:
    assert 1<=len(ix)<=6 and z['proposal_loss'][ix[-1]]<=z['proposal_armijo'][ix[-1]]
    assert np.array_equal(B,z['proposal_c2w'][ix[-1]])
    if k+1<len(z['loss']):assert abs(z['loss'][k+1]-z['proposal_loss'][ix[-1]])<=1e-5*max(abs(z['proposal_loss'][ix[-1]]),1.)
   else:assert len(ix)==6 and np.array_equal(A,B)
  if not ok:assert k==len(z['loss'])-1
  proposals+=len(ix)
 assert maximum<=2e-6
 return {'status':'PASS','iterations':len(z['loss']),'proposals':proposals,'pose_max_abs':maximum}
