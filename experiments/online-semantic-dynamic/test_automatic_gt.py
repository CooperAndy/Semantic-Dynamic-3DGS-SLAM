import unittest
import numpy as np
from semantickitti_projection import project
from automatic_gt import sparse_metrics

class SparseGTControls(unittest.TestCase):
    def test_static_vehicle_missing_label_counts_against_retention(self):
        static=np.ones((1,2),bool)
        gt=dict(static=static,depth=np.ones((1,2)),motion_known=static,moving=~static,
                label=np.full((1,2),13),raw_semantic=np.full((1,2),10))
        rgb=np.zeros((3,1,2));r=sparse_metrics(rgb,rgb,np.ones((1,2)),np.ones((1,2)),np.array([[13,-1]]),gt)
        self.assertEqual(r['static_vehicle_GT_pixels'],2)
        self.assertEqual(r['static_vehicle_label_coverage'],.5)
        self.assertEqual(r['static_vehicle_semantic_accuracy'],.5)
    def project(self,points,labels):
        P=np.c_[np.eye(3),np.zeros(3)];P[0,2]=1;P[1,2]=1
        return project(np.asarray(points,np.float32),np.asarray(labels,np.uint32),P,np.eye(4),3,3)[0]
    def test_occlusion_before_ignore(self):
        g=self.project([[0,0,1,0],[0,0,2,0]],[0,10])
        self.assertEqual(g['point_index'][1,1],0);self.assertEqual(g['label'][1,1],255)
        self.assertEqual(int((g['point_index']>=0).sum()),1)
    def test_packed_instance_and_moving_category(self):
        g=self.project([[0,0,2,0]],[252+(123<<16)])
        self.assertEqual(g['raw_semantic'][1,1],252);self.assertEqual(g['instance'][1,1],123)
        self.assertEqual(g['label'][1,1],13)
    def test_uncovered_depth_stays_in_denominator(self):
        static=np.array([[True,False],[False,False]])
        gt=dict(static=static,depth=np.full((2,2),4,np.float32),motion_known=static,moving=~static&False,label=np.full((2,2),255,np.uint8))
        rgb=np.zeros((3,2,2),np.float32)
        r=sparse_metrics(rgb,rgb,np.zeros((2,2),np.float32),np.zeros((2,2),np.float32),None,gt)
        self.assertEqual(r['static_GT_pixels'],1);self.assertEqual(r['static_depth_MAE_m'],4.)
        self.assertEqual(r['static_depth_coverage'],0.)
    def test_depth_alpha_normalization(self):
        static=np.ones((2,2),bool)
        gt=dict(static=static,depth=np.full((2,2),4,np.float32),motion_known=static,moving=~static,label=np.zeros((2,2),np.uint8))
        rgb=np.zeros((3,2,2),np.float32)
        r=sparse_metrics(rgb,rgb,np.full((2,2),2,np.float32),np.full((2,2),.5,np.float32),np.zeros((2,2),np.int16),gt)
        self.assertEqual(r['static_depth_MAE_m'],0.);self.assertEqual(r['sparse_mIoU'],1.)

if __name__=='__main__':unittest.main()
