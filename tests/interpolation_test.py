import unittest

import numpy as np
from scipy.interpolate import BSpline as scipy_BSpline
from scipy.interpolate import CubicSpline

from gwinferno.interpolation import BasisSpline
from gwinferno.interpolation import BSpline
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
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


class TestBSplines(unittest.TestCase):
    def setUp(self) -> None:
        self.gr = np.linspace(0, 1, 1000)
        self.grid = np.linspace(0.001, 1, 1000)
        self.N = 10
        self.cs_pn = np.random.normal(size=self.N)
        self.cs = np.random.uniform(size=self.N)

    def tearDown(self) -> None:
        del self.gr
        del self.N

    def test_bspline_matches_scipy(self):
        bspl = BSpline(self.N)
        dmat = bspl.bases(self.gr).T
        dmat2 = scipy_BSpline(bspl.knots, np.eye(self.N), 3)(self.gr)
        for i in range(self.N):
            self.assertTrue(np.allclose(dmat[:, i], dmat2[:, i]), f"{i}: {dmat[:, i]}, {dmat2[:, i]}")

    def test_basis_spline_norm(self):
        spl = BasisSpline(self.N, normalize=True)
        dmat = spl.bases(self.gr).T
        norm = np.trapz(spl.project(dmat, self.cs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_b_spline_norm(self):
        spl = BSpline(self.N, normalize=True)
        dmat = spl.bases(self.gr).T
        norm = np.trapz(spl.project(dmat, self.cs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_logyb_spline_norm(self):
        spl = LogYBSpline(self.N, normalize=True)
        dmat = spl.bases(self.gr).T
        norm = np.trapz(spl.project(dmat, self.cs_pn), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_logxb_spline_norm(self):
        spl = LogXBSpline(self.N, normalize=True)
        dmat = spl.bases(self.grid).T
        norm = np.trapz(spl.project(dmat, self.cs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_logxlogyb_spline_norm(self):
        spl = LogXLogYBSpline(self.N, normalize=True)
        dmat = spl.bases(self.grid).T
        norm = np.trapz(spl.project(dmat, self.cs_pn), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)
