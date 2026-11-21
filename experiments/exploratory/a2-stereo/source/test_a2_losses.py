import unittest
import torch
from a2_losses import residual_scale,robust_loss

class Tests(unittest.TestCase):
 def test_physical_distance_and_clamp(self):
  z=torch.tensor([1.,10.,20.,80.],dtype=torch.float64)
  s=residual_scale(z,None,380.,'distance')
  self.assertAlmostEqual(s[0].item(),.1);self.assertAlmostEqual(s[1].item(),100/380)
  self.assertAlmostEqual(s[2].item(),400/380);self.assertEqual(s[3].item(),3.)
  self.assertTrue((s[1:]>=s[:-1]).all())
 def test_confidence_is_bounded_heuristic(self):
  z=torch.tensor([10.,10.,10.]);c=torch.tensor([1.,.9,.8]);s=residual_scale(z,c,380.,'confidence')
  self.assertTrue((s[1:]>s[:-1]).all())
 def test_huber_value_continuity_and_gradient(self):
  residual=torch.tensor([-.3,0.,.3],dtype=torch.float64,requires_grad=True);d=torch.ones(3,dtype=torch.float64)
  v=robust_loss(residual,d,None,380.,'fixed');g=torch.autograd.grad(v,residual)[0]
  self.assertAlmostEqual(v.item(),.1);self.assertTrue(torch.allclose(g,torch.tensor([-1/3,0,1/3],dtype=torch.float64)))
  eps=1e-6
  for i in range(3):
   p=residual.detach().clone();q=p.clone();p[i]+=eps;q[i]-=eps
   fd=(robust_loss(p,d,None,380.,'fixed')-robust_loss(q,d,None,380.,'fixed'))/(2*eps)
   self.assertAlmostEqual(fd.item(),g[i].item(),places=5)
 def test_invalid_values_rejected(self):
  for z,fb in [(torch.tensor([0.]),380.),(torch.tensor([float('nan')]),380.),(torch.tensor([1.]),0.)]:
   with self.assertRaises(ValueError):residual_scale(z,None,fb,'distance')
  with self.assertRaises(ValueError):residual_scale(torch.tensor([1.]),torch.tensor([1.1]),380.,'confidence')

if __name__=='__main__':unittest.main(verbosity=2)
