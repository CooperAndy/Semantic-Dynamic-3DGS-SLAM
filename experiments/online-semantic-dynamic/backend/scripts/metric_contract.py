import numpy as np,json
from scipy.spatial.transform import Rotation
from trajectory_metrics import evaluate
rng=np.random.default_rng(180028);G=np.repeat(np.eye(4)[None],30,axis=0);G[:,:3,:3]=Rotation.random(30,random_state=rng).as_matrix();G[:,:3,3]=rng.normal(0,2,(30,3));H=np.eye(4);H[:3,:3]=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();H[:3,3]=[3,-5,1];E=H[None]@G;a,_=evaluate(E,G);assert max(a['ATE_RMSE_m'],a['RPE_translation_RMSE_m_1frame'],a['RPE_rotation_RMSE_deg_1frame'])<1e-10
scaled=G.copy();scaled[:,:3,3]*=2;b,_=evaluate(scaled,G);assert b['scale']==1 and b['ATE_RMSE_m']>1.
line=np.repeat(np.eye(4)[None],20,axis=0);line[:,0,3]=np.arange(20)*.2;drift=line.copy();drift[:,0,3]+=np.arange(20)*.01;c,_=evaluate(drift,line);assert abs(c['RPE_translation_RMSE_m_1frame']-.01)<1e-12
print(json.dumps(dict(status='PASS_SE3_NO_SCALE_AND_KNOWN_ONE_FRAME_RPE',rigid_ATE=a['ATE_RMSE_m'],scale_error_ATE=b['ATE_RMSE_m'],known_RPE=c['RPE_translation_RMSE_m_1frame'],GPU_calls=0)))
