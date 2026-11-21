from common import *
from producer import compute,legacy
import numpy as np
H,W=40,160;rgb=np.full((H,W,3),127,dtype=np.uint8);d0=np.full((H,W),5,dtype=np.float32);d1=np.full((H,W),4.5,dtype=np.float32);T0=np.eye(4);T1=np.eye(4);T1[2,3]=.5;K=dict(fx=50.,fy=50.,cx=80.,cy=20.,baseline_m=.5)
first=compute(rgb,rgb,d0,d0,T0,None,None,K);valid=np.ones((H,W),dtype=bool);lr,*_=legacy.confidence_from_stereo(rgb,rgb,d0,d0,25.,valid,1.,.1);assert np.array_equal(first['confidence'],lr)
second=compute(rgb,rgb,d1,d1,T1,d0,T0,K);tc,err,corr=legacy.temporal_confidence(d1,d0,T1,T0,K,valid,.1);assert corr.all() and np.max(err)==0 and np.all(tc==1);assert np.min(second['confidence'][:,6:])>.9999
bad=d1.copy();bad[:,20:23]=0;bad[:,23:26]=81;invalid=compute(rgb,rgb,bad,d1,T1,d0,T0,K);assert np.all(invalid['confidence'][:,20:26]==0) and np.all(invalid['depth_confhard'][:,20:26]==0)
for outputs,d in [(first,d0),(second,d1),(invalid,bad)]:
 assert outputs['confidence'].dtype==outputs['depth_confhard'].dtype==np.float32
 assert outputs['seed_mask'].dtype==outputs['loss_mask'].dtype==np.bool_
 assert np.array_equal(outputs['depth_confhard'],np.where(outputs['confidence']>=.8,d,0).astype(np.float32))
dump(R/'receipts/contract.json',dict(status='PASS_KNOWN_MOVING_PLANE_FIRST_FRAME_AND_INVALID_RANGE',at=now(),first_frame_LR_bitwise=True,known_plane_translation_z_m=.5,temporal_max_error=0.,synthetic_only=True,GT_file_reads=0))
print((R/'receipts/contract.json').read_text())
