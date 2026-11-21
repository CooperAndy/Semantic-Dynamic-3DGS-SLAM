from pathlib import Path
import importlib.util,json,numpy as np
from scipy.spatial.transform import Rotation
F=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('t17_eval',F/'scripts/evaluate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
T=np.tile(np.eye(4),(21,1,1));T[:,0,3]=np.arange(21)*.1
same=m.metrics(T,T);assert all(abs(same[k])<1e-12 for k in ['ATE','RPE_t_1','RPE_t_10','RPE_r_1','RPE_r_10'])
Q=np.eye(4);Q[:3,:3]=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();Q[:3,3]=[3,-2,1]
rigid=m.metrics(T,Q@T);assert all(abs(rigid[k])<1e-12 for k in ['ATE','RPE_t_1','RPE_t_10','RPE_r_1','RPE_r_10'])
scaled=T.copy();scaled[:,:3,3]*=2;v=m.metrics(T,scaled)
assert abs(v['RPE_t_1']-.1)<1e-12 and abs(v['RPE_t_10']-1)<1e-12 and abs(v['ATE']-np.std(T[:,0,3]))<1e-12
assert v['pairs_1']==20 and v['pairs_10']==11
out={'status':'PASS','checks':['identical trajectory zero','global SE3 invariance','scale error retained without Sim3','delta1/10 full pairs','Horn versus SVD checked in every metrics call'],'no_real_GT_read':True}
(F/'receipts/CPU_metric_selfcheck.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print('CPU_METRIC_SELFCHECK_PASS')
