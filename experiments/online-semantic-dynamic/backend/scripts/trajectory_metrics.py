"""Metric-scale SE3 alignment and one-frame relative pose errors."""
import numpy as np
from scipy.spatial.transform import Rotation
def evaluate(estimate,reference):
 E=np.asarray(estimate,np.float64);G=np.asarray(reference,np.float64);assert E.shape==G.shape and E.ndim==3 and E.shape[1:]==(4,4) and len(E)>=3 and np.isfinite(E).all() and np.isfinite(G).all()
 x=E[:,:3,3];y=G[:,:3,3];xc=x-x.mean(0);yc=y-y.mean(0);u,s,vt=np.linalg.svd(xc.T@yc);D=np.diag([1.,1.,np.linalg.det(vt.T@u.T)]);r=vt.T@D@u.T;t=y.mean(0)-r@x.mean(0);A=np.eye(4);A[:3,:3]=r;A[:3,3]=t;aligned=A[None]@E;ape=np.linalg.norm(aligned[:,:3,3]-y,axis=1);relative_G=np.linalg.inv(G[:-1])@G[1:];relative_E=np.linalg.inv(E[:-1])@E[1:];delta=np.linalg.inv(relative_G)@relative_E;trans=np.linalg.norm(delta[:,:3,3],axis=1);rot=np.degrees(Rotation.from_matrix(delta[:,:3,:3]).magnitude())
 return dict(frames=len(E),alignment='SE3_positions_no_scale',scale=1.,alignment_singular_values=s.tolist(),alignment_matrix=A.tolist(),ATE_RMSE_m=float(np.sqrt(np.mean(ape**2))),ATE_mean_m=float(ape.mean()),RPE_translation_RMSE_m_1frame=float(np.sqrt(np.mean(trans**2))),RPE_rotation_RMSE_deg_1frame=float(np.sqrt(np.mean(rot**2)))),dict(APE=ape,RPE_translation=trans,RPE_rotation=rot,aligned_c2w=aligned)
