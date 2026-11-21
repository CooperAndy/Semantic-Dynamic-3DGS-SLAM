"""Tests catch front ignore leaking background, packing and P2 translation."""
import numpy as np
from semantickitti_projection import project
P=np.array([[10,0,5,2],[0,10,5,0],[0,0,1,0]],float);T=np.eye(4)
p=np.array([[0,0,2,.5],[0,0,5,.5]],np.float32);labels=np.array([0,40+(17<<16)],np.uint32)
gt,stats,uv=project(p,labels,P,T,12,12)
assert uv[0,0]==6 and uv[1,0]==5.4
P[:,3]=0;gt,stats,uv=project(p,labels,P,T,12,12)
assert gt['point_index'][5,5]==0 and gt['label'][5,5]==255,'front ignored point must hide road behind'
gt,stats,uv=project(p[::-1].copy(),labels[::-1].copy(),P,T,12,12)
assert gt['point_index'][5,5]==1 and gt['label'][5,5]==255,'point order is not depth order'
gt,stats,uv=project(p,np.array([252+(12<<16),40],np.uint32),P,T,12,12)
assert gt['label'][5,5]==13 and gt['raw_semantic'][5,5]==252 and gt['instance'][5,5]==12
print('PASS_PACKING_TRANSLATION_ALL_LABEL_ZBUFFER_AND_FRONT_IGNORE')
