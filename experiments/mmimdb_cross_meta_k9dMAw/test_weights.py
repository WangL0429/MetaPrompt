import unittest
import numpy as np
from meta_weights import choose_weights

class WeightTest(unittest.TestCase):
    def test_uninformative_episode_has_uniform_weights(self):
        np.testing.assert_allclose(choose_weights(np.zeros((3,3))), [1/3]*3)

    def test_reward_and_worst_query_penalty(self):
        # The second direction brings a gain for every query. The first hurts one query.
        delta = np.array([[-2., -2., 3.], [-1., -1., -1.], [.3, .3, .3]])
        w = choose_weights(delta)
        self.assertAlmostEqual(float(w.sum()), 1.)
        self.assertTrue((w >= 0.099).all())
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[1], w[2])
