import unittest
import torch
try:
    from mixing import mix_query
except ImportError:
    mix_query=None


class MixTest(unittest.TestCase):
    def test_identity_scaling_and_second_derivatives(self):
        self.assertTrue(callable(mix_query),'missing shared query mixer')
        for width in (5,7):
            torch.manual_seed(5)
            q=torch.randn(3,1,width,dtype=torch.float64)
            p=torch.randn(3,width,dtype=torch.float64,requires_grad=True)
            a=torch.zeros(3,3,dtype=torch.float64,requires_grad=True)
            torch.testing.assert_close(mix_query(q,p,a),q,atol=0,rtol=0)
            with torch.no_grad():
                a.copy_(torch.eye(3,dtype=a.dtype))
            delta=mix_query(q,p,a)-q
            torch.testing.assert_close(delta.square().mean(),q.square().mean())
            fn=lambda pm,am: mix_query(q,pm,am)
            self.assertTrue(torch.autograd.gradcheck(fn,(p,a)))
            self.assertTrue(torch.autograd.gradgradcheck(fn,(p,a)))
            self.assertTrue(mix_query(q,p*0,a).isfinite().all())
