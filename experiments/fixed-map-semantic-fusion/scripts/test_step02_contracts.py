"""Shared-support classification and zero-history state tests for real fusion."""
import unittest,tempfile
import numpy as np
from semantic_cpu import Evidence
from native_semantic_render import decode_mass

class Contracts(unittest.TestCase):
    def test_shared_alpha_stabilizes_threshold_denominator(self):
        # Both channel sums are inside native numerical tolerance around 0.05.
        reference=np.array([[.05]],np.float32)
        a=np.array([[[.049999]],[[0]]],np.float32)
        b=np.array([[[.050001]],[[0]]],np.float32)
        self.assertEqual(decode_mass(a,reference_alpha=reference)['label'][0,0],0)
        self.assertEqual(decode_mass(b,reference_alpha=reference)['label'][0,0],0)
    def test_inconsistent_alpha_is_rejected(self):
        with self.assertRaises(ValueError):decode_mass(np.array([[[.8]],[[0]]]),reference_alpha=np.array([[.5]]))
    def test_hard_rejection_preserves_frame_ledger_without_support(self):
        s=Evidence([7,2],2);frame={'sequence':'08','original_frame':3000}
        s.update_frame(frame,'fixed',[[1,0],[0,1]],[0,0])
        self.assertEqual(len(s.processed),1);self.assertFalse(s.support.any());self.assertFalse(s.evidence.any())
        with tempfile.TemporaryDirectory() as tmp:
            s.save(tmp+'/state.npz');restored=Evidence.load(tmp+'/state.npz',[7,2])
            self.assertFalse(restored.update_frame(frame,'fixed',[[1,0],[0,1]],[0,0]))

if __name__=='__main__':unittest.main(verbosity=2)
