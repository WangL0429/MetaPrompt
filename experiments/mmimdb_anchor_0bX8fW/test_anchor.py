import unittest
import torch
try:
    from anchor import soft_grid, worst_drop, eligible
except ModuleNotFoundError:
    soft_grid = worst_drop = eligible = None


class AnchorTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(soft_grid), "missing nine-condition soft Macro-F1")

    def test_grid_mixes_counts_before_division(self):
        y = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
        p = torch.tensor([[[.9,.3],[.1,.7],[.8,.1]], [[.3,.2],[.6,.4],[.4,.5]],
                          [[.7,.1],[.8,.8],[.1,.9]]])
        expected = []
        for mode in ('both','image','text'):
            for r in (.5,.7,.9):
                t = {'text':r,'image':0.,'both':r/2}[mode]
                mixed = p[0]*(1-r) + p[1]*t + p[2]*(r-t)
                expected.append(((2*(mixed*y).sum(0))/(mixed.sum(0)+y.sum(0))).mean())
        torch.testing.assert_close(soft_grid(p.logit(),y), torch.stack(expected))

    def test_positive_drop_only_and_teacher_is_frozen(self):
        ref = torch.tensor([.6,.7,.8],requires_grad=True)
        current = torch.tensor([.59,.65,.82],requires_grad=True)
        loss = worst_drop(ref,current)
        loss.backward()
        self.assertAlmostEqual(loss.item(), .05, places=6)
        self.assertIsNone(ref.grad)
        torch.testing.assert_close(current.grad, torch.tensor([0.,-1.,0.]))
        for change in (0.,.01):
            current = (ref.detach()+change).requires_grad_()
            worst_drop(ref,current).backward()
            self.assertFalse(current.grad.any())

    def test_gate_is_in_percentage_points(self):
        base = [50.]*9
        self.assertTrue(eligible([49.75]+base[1:], base))
        self.assertFalse(eligible([49.749]+[60.]*8, base))
        self.assertFalse(eligible([float('nan')]+base[1:], base))

if __name__ == '__main__':
    unittest.main()
