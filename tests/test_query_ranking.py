import unittest
import torch
from torch.nn import functional as F
from clip.modules.query_ranking import query_ranking_loss


class QueryRankingTests(unittest.TestCase):
    def test_matches_enumerated_class_balanced_pairs_and_chunked_backward(self):
        labels = torch.tensor([[1,0,1],[1,0,1],[0,1,1],[0,0,1],[0,0,1]])
        x = torch.arange(15, dtype=torch.float64).reshape(5,3) / 10
        w = torch.eye(3, dtype=x.dtype, requires_grad=True)
        scores = x @ w
        def loss(z):
            return query_ranking_loss(z, labels, task='multilabel')
        manual = []
        for c in range(2):
            terms = [F.softplus(scores[n,c]-scores[p,c]) for p in range(5)
                     for n in range(5) if labels[p,c] and not labels[n,c]]
            manual.append(torch.stack(terms).mean())
        torch.testing.assert_close(loss(scores), torch.stack(manual).mean())
        expected, = torch.autograd.grad(loss(scores), w)
        detached = scores.detach().requires_grad_()
        grad, = torch.autograd.grad(loss(detached), detached)
        actual = torch.zeros_like(w)
        for idx in torch.arange(5).split(2):
            part = (x[idx] @ w * grad[idx]).sum()
            actual += torch.autograd.grad(part, w)[0]
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual.norm().item(), 0)

    def test_binary_and_multiclass(self):
        a=torch.tensor([-1., .3, 2.], requires_grad=True)
        y=torch.tensor([0,1,1])
        two=torch.stack([a*0,a],dim=1)
        torch.testing.assert_close(query_ranking_loss(a,y,task='binary'),
                                   query_ranking_loss(two,y,task='binary'))
        z=torch.tensor([[2.,0.,1.],[1.,-2.,0.],[-.2,1.,2.]],requires_grad=True)
        y=torch.tensor([0,2,1])
        r=query_ranking_loss(z,y,task='multiclass')
        torch.testing.assert_close(r,query_ranking_loss(z+torch.tensor([[2.],[1.],[-3.]]),y,task='multiclass'))
        self.assertTrue(torch.autograd.grad(r,z)[0].isfinite().all())

    def test_empty_and_unpaired_groups(self):
        for n in (0,1,4):
            z=torch.ones(n,2,requires_grad=True)
            loss=query_ranking_loss(z,z.detach(),task='multilabel')
            self.assertEqual(loss.item(),0.)
            self.assertEqual(torch.autograd.grad(loss,z)[0].abs().sum().item(),0.)
