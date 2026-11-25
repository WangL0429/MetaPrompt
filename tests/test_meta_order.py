from types import SimpleNamespace
import unittest

import torch
from clip.modules.clip_meta_prompt_module import CLIPransformerSS


class QuadraticTask:
    _prompt_param_names = CLIPransformerSS._prompt_param_names
    adapt = CLIPransformerSS._inner_adapt_prompts

    def __init__(self, second_order):
        self.model = torch.nn.Module()
        self.model.prompt_learner = torch.nn.ParameterDict({
            'active': torch.nn.Parameter(torch.tensor(1.7,dtype=torch.float64)),
            'unused': torch.nn.Parameter(torch.tensor(.4,dtype=torch.float64))})
        self.hparams = SimpleNamespace(config={'meta_second_order':True} if second_order else {})

    @staticmethod
    def support(_, params):
        return 1.5 * (params['prompt_learner.active'] - .3).square()


class MetaOrderTest(unittest.TestCase):
    def test_query_gradient_through_two_inner_steps(self):
        lr = .1
        def objective(w):
            for _ in range(2):
                w = w-lr*3*(w-.3)
            return (w-.9)**2 / 2
        eps = 1e-5
        numeric = (objective(1.7+eps)-objective(1.7-eps))/(2*eps)
        for second in (False,True):
            with self.subTest(second_order=second):
                task = QuadraticTask(second)
                fast = task.adapt(None,task.support,inner_lr=lr,inner_steps=2)
                w = fast['prompt_learner.active']
                q = (w-.9).square()/2 + fast['prompt_learner.unused']
                g, unused = torch.autograd.grad(q,(task.model.prompt_learner.active,
                                                     task.model.prompt_learner.unused))
                self.assertAlmostEqual(g.item(), numeric if second else w.item()-.9, places=9)
                self.assertEqual(unused.item(),1.)
