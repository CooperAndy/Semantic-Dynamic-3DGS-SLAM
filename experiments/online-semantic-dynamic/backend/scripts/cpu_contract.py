from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation
from pose_chain import predict,select
rng=np.random.default_rng(180028)
def se3(scale):
 T=np.eye(4);T[:3,:3]=Rotation.random(random_state=rng).as_matrix();T[:3,3]=rng.normal(0,scale,3);return T
maximum=0.;checks=0
for _ in range(256):
 previous=se3(200);current=se3(200);G=se3(.1);H=se3(500);selected=previous@G;prediction=predict(previous,current,selected,True);expected=current@G;err=np.max(np.abs(prediction-expected));assert err<1e-9;maximum=max(maximum,float(err));checks+=1
 moved=predict(previous@np.linalg.inv(H),current@np.linalg.inv(H),selected@np.linalg.inv(H),True);assert np.max(np.abs(moved-prediction@np.linalg.inv(H)))<1e-9;checks+=1
 assert np.array_equal(predict(previous,current,previous,False),current);checks+=1
 assert np.array_equal(select(prediction,expected,False),prediction) and np.array_equal(select(prediction,expected,True),expected);checks+=1
j=dict(status='PASS_1024_CONTINUOUS_POSE_PROPAGATION_CONTRACTS',checks=checks,max_error=maximum,GPU_calls=0,scope='Matrix chain, origin invariance, no-feedback bypass and rejected-candidate fallback; no SLAM/GPU trajectory yet.');Path(__file__).with_name('cpu_contract.json').write_text(json.dumps(j,indent=2));print(json.dumps(j))
