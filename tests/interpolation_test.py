import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline as scipy_BSpline
from scipy.interpolate import CubicSpline

from gwinferno.interpolation import BasisSpline
from gwinferno.interpolation import BSpline
from gwinferno.interpolation import in_bounds
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
        dmat = spl.bases(self.gr)
        norm = np.trapz(spl.project(dmat, self.cs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_b_spline_norm(self):
        spl = BSpline(self.N, normalize=True)
        dmat = spl.bases(self.gr)
        norm = np.trapz(spl.project(dmat, self.cs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_logyb_spline_norm(self):
        spl = LogYBSpline(self.N, normalize=True)
        dmat = spl.bases(self.gr)
        norm = np.trapz(spl.project(dmat, self.cs_pn), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_logxb_spline_norm(self):
        spl = LogXBSpline(self.N, xrange=(0.001, 1), normalize=True)
        dmat = spl.bases(self.grid)
        norm = np.trapz(spl.project(dmat, self.cs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_logxlogyb_spline_norm(self):
        spl = LogXLogYBSpline(self.N, xrange=(0.001, 1), normalize=True)
        dmat = spl.bases(self.grid)
        norm = np.trapz(spl.project(dmat, self.cs_pn), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=3)


class TestOutOfBoundsHandling(unittest.TestCase):
    """The out-of-bounds convention must be JAX-safe and consistent across every basis variant.

    Encoding out-of-bounds as -inf in the design matrix makes the backward pass evaluate
    ``-inf * 0``, so one out-of-range sample used to turn the *whole* gradient vector into NaN
    while the forward pass still looked correct.
    """

    N = 8
    N_IN, N_OUT = 3, 3
    # (class, xrange). Log-x variants need a strictly positive lower bound.
    CASES = [
        (BSpline, (0.0, 1.0)),
        (LogXBSpline, (0.01, 1.0)),
        (LogYBSpline, (0.0, 1.0)),
        (LogXLogYBSpline, (0.01, 1.0)),
    ]

    def setUp(self) -> None:
        self.coefs = jnp.asarray(np.random.default_rng(4).normal(size=self.N))

    def _probe(self, xrange):
        """Three points strictly inside the domain, then three strictly outside it.

        Note 0.0 is deliberately not used as an out-of-range probe: it lies *inside* xrange for the
        bases whose domain starts at 0. It is covered separately for the log-x bases, where it is
        genuinely outside.
        """
        lo, hi = xrange
        width = hi - lo
        return np.array([lo + 0.1 * width, 0.5 * (lo + hi), hi - 0.1 * width, lo - 1e-2, hi + 0.5, -1.0])

    def test_design_matrix_is_finite(self):
        for cls, xrange in self.CASES:
            with self.subTest(basis=cls.__name__):
                dmat = np.asarray(cls(self.N, xrange=xrange, normalize=False).bases(self._probe(xrange)))
                self.assertTrue(np.all(np.isfinite(dmat)), msg=f"{cls.__name__} design matrix is not finite")

    def test_gradients_are_finite_with_out_of_bounds_samples(self):
        for cls, xrange in self.CASES:
            spl = cls(self.N, xrange=xrange, normalize=False)
            # bases() is host-side, so build the design matrix outside the traced function.
            dmat = spl.bases(self._probe(xrange))
            loss = lambda cs: jnp.sum(spl.project(dmat, cs))  # noqa: E731
            batch = jnp.asarray(np.random.default_rng(9).normal(size=(3, self.N)))
            for label, grad in [
                ("grad", jax.grad(loss)(self.coefs)),
                ("jit(grad)", jax.jit(jax.grad(loss))(self.coefs)),
                ("vmap(grad)", jax.vmap(jax.grad(loss))(batch)),
            ]:
                with self.subTest(basis=cls.__name__, transform=label):
                    self.assertTrue(np.all(np.isfinite(np.asarray(grad))), msg=f"{cls.__name__} {label} is not finite")

    def test_out_of_bounds_points_project_to_zero(self):
        for cls, xrange in self.CASES:
            spl = cls(self.N, xrange=xrange, normalize=False)
            probe = self._probe(xrange)
            with self.subTest(basis=cls.__name__):
                projected = np.asarray(spl.project(spl.bases(probe), self.coefs))
                np.testing.assert_array_equal(projected[self.N_IN :], np.zeros(self.N_OUT))

    def test_non_positive_x_is_out_of_bounds_for_log_x_bases(self):
        # Previously LogXBSpline returned NaN here while the other variants returned 0.
        for cls in (LogXBSpline, LogXLogYBSpline):
            spl = cls(self.N, xrange=(0.01, 1.0), normalize=False)
            with self.subTest(basis=cls.__name__):
                projected = np.asarray(spl.project(spl.bases(np.array([0.0, -1.0, -np.inf])), self.coefs))
                np.testing.assert_array_equal(projected, np.zeros(3))

    def test_in_bounds_mask_agrees_with_the_domain(self):
        for cls, xrange in self.CASES:
            spl = cls(self.N, xrange=xrange, normalize=False)
            probe = self._probe(xrange)
            with self.subTest(basis=cls.__name__):
                np.testing.assert_array_equal(
                    np.asarray(in_bounds(spl.bases(probe))),
                    np.array([True] * self.N_IN + [False] * self.N_OUT),
                )
