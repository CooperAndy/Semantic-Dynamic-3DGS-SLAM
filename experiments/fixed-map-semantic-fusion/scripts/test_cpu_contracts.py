"""Meaningful CPU contract checks; real renderer execution remains pending."""
import argparse,json,tempfile,unittest
from pathlib import Path
import numpy as np
from semantic_cpu import *

class Contracts(unittest.TestCase):
    def test_unknown_occludes_background(self):
        good=composite([.99,.99],[[0,0],[0,1]],[True,False])
        old=composite([0,.99],[[0,0],[0,1]],[True,False])
        self.assertEqual(good['label'],UNKNOWN);self.assertEqual(old['label'],1)
        self.assertLess(good['class_mass'][1],.011)

    def test_transparent_foreground_and_no_map(self):
        self.assertEqual(composite([.01,.99],[[0,0],[1,0]],[True,False])['label'],0)
        self.assertEqual(composite([.001],[[1,0]],[False])['label'],NO_MAP)

    def test_replay_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            e=Evidence([71,9],2);frame={'seq':'08','camera':2,'frame':3000};d=np.array([[1,0],[0,1]])
            e.update_frame(frame,'v1',d,[1,1]);path=Path(temp)/'state.npz';e.save(path)
            f=Evidence.load(path,[71,9]);self.assertFalse(f.update_frame(frame,'v1',d,[1,1]))
            np.testing.assert_array_equal(f.support,[1,1]);np.testing.assert_array_equal(f.evidence,d)
            with self.assertRaises(ValueError):f.update_frame(frame,'v2',d,[1,1])
            with self.assertRaises(ValueError):Evidence.load(path,[9,71])

    def test_frame_area_normalization(self):
        d,w=aggregate_pixels([0,0,0,1],[1,1,1,1],[.3,.3,.3,.9],2,2)
        np.testing.assert_array_equal(d,[[0,1],[0,1]]);np.testing.assert_array_equal(w,[1,1])

    def test_invalid_frame_does_not_commit(self):
        e=Evidence([1],2)
        with self.assertRaises(ValueError):e.update_frame({'frame':0},'v1',[[0,0]],[1])
        self.assertEqual(e.processed,{});self.assertEqual(e.evidence.sum(),0)

    def test_factorial_shared_validity(self):
        for method in ['S0','S1','S-G','S2','S2-H']:
            self.assertEqual(factor_weights(method,[False,True],[.8,.8],[.5,.5])[0],0)
        np.testing.assert_allclose(factor_weights('S-G',[True],[.1],[.7]),[.7])
        np.testing.assert_allclose(factor_weights('S2',[True],[.1],[.7]),[.07])

    def test_unknown_no_map_are_false_negatives(self):
        r=score_semantics([0,1,1,255],[0,UNKNOWN,NO_MAP,0],2)
        self.assertEqual(r['pixels'],3);self.assertEqual(r['classes'][1]['fn'],2)
        self.assertEqual(r['miou_gt_present_classes'],.5);self.assertAlmostEqual(r['coverage'],1/3)

    def test_risk_ties_and_unreachable_coverage(self):
        rows=risk_coverage([0,1,1],[0,0,NO_MAP],[.8,.8,1.])
        self.assertEqual(len(rows),2);self.assertAlmostEqual(rows[-1]['coverage'],2/3)
        self.assertEqual(rows[-1]['risk'],.5);self.assertIsNone(rows[0]['risk'])

    def test_native_adapter_keeps_unknown_opacity(self):
        p={'means3D':np.zeros((2,3)),'unnorm_rotations':np.array([[1.,0,0,0]]*2),
           'log_scales':np.zeros((2,1)),'logit_opacities':np.array([[5.],[5.]])}
        v=native_render_variables(p,p,np.array([[0,0,1],[0,1,0]]),np.zeros_like,
            lambda x:x/np.linalg.norm(x,axis=1,keepdims=True),np.exp,lambda x:1/(1+np.exp(-x)))
        np.testing.assert_allclose(v['opacities'],1/(1+np.exp(-p['logit_opacities'])))
        self.assertEqual(v['scales'].shape,(2,3))

    def test_early_error_can_be_corrected_without_history_gate(self):
        e=Evidence([1],2)
        e.update_frame({'frame':0},'v1',[[0,1]],[1])
        for frame in [1,2]:e.update_frame({'frame':frame},'v1',[[1,0]],[1])
        self.assertEqual(int(e.evidence.argmax(1)[0]),0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
       'failures':len(result.failures),'errors':len(result.errors),'scope':'CPU reference and adapter contracts only; native GPU rasterizer NOT tested'},indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
