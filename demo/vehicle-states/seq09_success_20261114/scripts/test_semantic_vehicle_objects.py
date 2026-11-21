import unittest
from dataclasses import replace
import numpy as np
from fuse_semantic_vehicle_objects import (FusionConfig, component_reasons, compound_conflicts,
                                          deduplicate_frame_observations, match_region, project)


class SemanticObjectTests(unittest.TestCase):
    def test_depth_inconsistent_region_is_not_associated(self):
        cfg=FusionConfig();points=np.zeros((8,3));ids=np.zeros(8,dtype=int)
        mask=np.ones((3,10),dtype=bool);depth=np.full((3,10),15.)
        best,_,reason=match_region(points,ids,np.arange(8),np.zeros(8,dtype=int),np.full(8,10.),mask,depth,cfg)
        self.assertIsNone(best);self.assertEqual(reason,'no_depth_consistent_region_support')

    def test_similarly_supported_adjacent_regions_are_ambiguous(self):
        points=np.zeros((16,3));ids=np.repeat([0,1],8)
        best,_,reason=match_region(points,ids,np.arange(16),np.zeros(16,dtype=int),np.full(16,10.),np.ones((2,20),bool),np.full((2,20),10.),FusionConfig())
        self.assertIsNone(best);self.assertEqual(reason,'ambiguous_regions')

    def test_one_model_score_per_object_frame(self):
        a=dict(source_frame=350,track_id=46,association_score=10.,detection_confidence=.9,mask_area=300,raw_score=.01)
        b=dict(source_frame=350,track_id=52,association_score=20.,detection_confidence=.9,mask_area=500,raw_score=.1)
        result=deduplicate_frame_observations([a,b])
        self.assertEqual(len(result),1);self.assertEqual(result[0]['track_id'],52)
        self.assertEqual(result[0]['merged_same_frame_tracks'],[46,52])

    def test_distinct_simultaneous_instances_flag_compound_region(self):
        a=dict(source_frame=350,track_id=1,bbox=[0,0,50,50],support_world_center=[0,0,0])
        b=dict(source_frame=350,track_id=2,bbox=[60,0,110,50],support_world_center=[3,0,0])
        self.assertEqual(len(compound_conflicts([a,b])),1)
        b['source_frame']=351
        self.assertFalse(compound_conflicts([a,b]))

    def test_oversized_connected_vehicle_region_is_not_accepted_as_one_car(self):
        rng=np.random.default_rng(42);points=rng.uniform([0,0,0],[20,2,2],(2000,3))
        reasons,_,_=component_reasons(points,500,FusionConfig())
        self.assertIn('length_outside_candidate_range',reasons)

    def test_behind_camera_points_have_no_valid_projection(self):
        points=np.array([[1.,2.,-10.],[0.,0.,10.]])
        u,v,z=project(points,dict(position=[0,0,0],rotation=np.eye(3)),np.array([[100,0,50],[0,100,50],[0,0,1]]))
        self.assertEqual((u[0],v[0]),(-1,-1));self.assertEqual((u[1],v[1]),(50,50))


if __name__=='__main__':unittest.main()
