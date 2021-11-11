from unittest import TestCase
from dmath.statistics.variance import Variance1D

import numpy as np

class TestVariance1D(TestCase):

    def test_init(self):
        with self.assertRaises(Exception):
            Variance1D([[1, 2, 3, 4], [5, 6, 7, 8]])
            Variance1D(np.ones((3, 1, 4)))
        
        # Unsqueezed 1D should be allowed
        a_np = np.ones((1, 1, 10))
        a = Variance1D(a_np)
        self.assertEqual(a.mean, 1)
        self.assertEqual(a.num_samples, 10)
        self.assertEqual(a.ssd, 0)
    
    def test_update(self):
        a_np = np.ones((1, 1, 10))
        a = Variance1D(a_np)

        with self.assertRaises(Exception):
            a.update([[1, 2, 3, 4], [5, 6, 7, 8]])
            a.update(np.ones((3, 1, 4)))

        a.update(np.ones(20)*2)

        self.assertEqual(a.num_samples, 30)
        self.assertAlmostEqual(a.mean, 5/3)
        self.assertAlmostEqual(a.ssd, 10*(5/3-1)**2 + 20*(2-5/3)**2)

    def test_merge(self):
         # Unsqueezed 1D should be allowed
        a_np = np.random.random((1, 1, 10))
        a = Variance1D(a_np)

        b_np = np.random.rand(5)
        b = Variance1D(b_np)
        a.merge(b)
        self.assertAlmostEqual(a.compute(), np.var(np.concatenate((a_np.squeeze(), b_np.squeeze())), ddof=1))
        
        a = Variance1D(a_np, is_sample_variance=False)
        a.merge(b)
        self.assertAlmostEqual(a.compute(), np.var(np.concatenate((a_np.squeeze(), b_np.squeeze())), ddof=0))