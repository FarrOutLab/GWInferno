import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from gwinferno.interpolation import NaturalCubicUnivariateSpline


class TestNaturalCubicUnivariateSpline(unittest.TestCase):
    def setUp(self) -> None:
        self.gr = np.linspace(0, 1, 1000)

    def get_random_diff_array(self, N):
        xs = np.linspace(0, 1, N)
        ys = np.random.normal(size=N)
        scipy_cs = CubicSpline(xs, ys, bc_type="natural")
        gwinf_cs = NaturalCubicUnivariateSpline(xs, ys)
        diff = scipy_cs(self.gr) - gwinf_cs(self.gr)
        return np.abs(diff[self.gr < 0.7])

    def test_random_diffs_varying_knots(self):
        threshold = 5e-4
        Niter = 15
        Nknots = [20, 25, 30]
        diffs = np.array([[self.get_random_diff_array(N) for _ in range(Niter)] for N in Nknots])
        self.assertTrue(np.all(diffs < threshold))

    def tearDown(self) -> None:
        del self.gr
