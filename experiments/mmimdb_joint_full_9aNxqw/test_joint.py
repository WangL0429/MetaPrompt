import unittest
import torch
try:
    from rule import unit, curvature
except ImportError:
    unit=curvature=None

class JointTest(unittest.TestCase):
    def test_weighted_two_stage_gradient_matches_exact_and_finite_difference(self):
        self.assertTrue(callable(curvature),'missing normalized support curvature')
        w=torch.tensor([.7,-.4],dtype=torch.double,requires_grad=True)
        mats=w.new_tensor([[[2.,.3],[.3,1.]],[[1.,.2],[.2,3.]],[[1.,-.2],[-.2,1.]]])
        weights=w.new_tensor([.2,.3,.5])
        def query(x):
            return (x[0]*x[1]-.35).square()+.3*x.square().sum()
        def fast(x):
            return x-.03*sum(a*unit(m@x) for m,a in zip(mats,weights))
        expected,=torch.autograd.grad(query(fast(w)),w)
        f=fast(w).detach().requires_grad_()
        v,=torch.autograd.grad(query(f),f)
        computed=v-.03*sum(a*curvature(.5*w@(m@w),[w],v) for m,a in zip(mats,weights))
        torch.testing.assert_close(computed,expected)
        eye=torch.eye(2,dtype=w.dtype)*1e-5
        finite=torch.stack([(query(fast(w+e))-query(fast(w-e)))/2e-5 for e in eye])
        torch.testing.assert_close(computed,finite)
        self.assertGreater((computed-v).norm().item(),1e-6)
