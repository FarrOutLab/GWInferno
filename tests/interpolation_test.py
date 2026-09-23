import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline as scipy_BSpline
from scipy.interpolate import CubicSpline

from gwinferno.interpolation import BSpline
from gwinferno.interpolation import in_bounds
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.interpolation import MSpline
from gwinferno.interpolation import NaturalCubicUnivariateSpline
from gwinferno.interpolation import SplineBasis


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
        spl = MSpline(self.N, normalize=True)
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


class TestBasisIdentities(unittest.TestCase):
    """Identities that distinguish the M-spline basis from the B-spline basis.

    These are what stop a rename or restructure from silently swapping the two. The defining
    relation is exact in float32, so any drift at all is a real error rather than noise.
    """

    N = 10

    def setUp(self) -> None:
        self.xs = np.linspace(0.0, 1.0, 500)

    def test_bspline_is_mspline_scaled_by_knot_span(self):
        # B_i(x) == (t_{i+k} - t_i) / k * M_i(x) -- the M -> B conversion, applied in BSpline._bases.
        mspline = MSpline(self.N, normalize=False)
        bspline = BSpline(self.N, normalize=False)
        knots, order = np.asarray(mspline.knots), mspline.order
        scale = np.array([(knots[i + order] - knots[i]) / order for i in range(self.N)])[:, None]
        np.testing.assert_array_equal(np.asarray(mspline.bases(self.xs)) * scale, np.asarray(bspline.bases(self.xs)))

    def test_only_the_bspline_basis_is_a_partition_of_unity(self):
        bspline = BSpline(self.N, normalize=False)
        np.testing.assert_allclose(np.asarray(bspline.bases(self.xs)).sum(axis=0), 1.0, atol=1e-6)

    def test_only_the_mspline_basis_has_unit_integral(self):
        # Interior functions only: those overlapping a boundary are truncated by the xrange clip in
        # bases(), so their integral over the domain is legitimately less than 1.
        mspline = MSpline(self.N, normalize=True)
        np.testing.assert_allclose(np.asarray(mspline.basis_vols)[3:7], 1.0, rtol=1e-5)
        bspline = BSpline(self.N, normalize=True)
        self.assertFalse(np.allclose(np.asarray(bspline.basis_vols)[3:7], 1.0, rtol=1e-5))


class TestSplineBasisStructure(unittest.TestCase):
    """The contract the subclasses are written against, and the sibling relationship.

    MSpline and BSpline share a knot vector and the Cox-de Boor recursion but differ in how the
    basis functions are normalized, so neither is a specialization of the other.
    """

    N = 10
    CASES = [
        (MSpline, (0.0, 1.0)),
        (BSpline, (0.0, 1.0)),
        (LogXBSpline, (0.01, 1.0)),
        (LogYBSpline, (0.0, 1.0)),
        (LogXLogYBSpline, (0.01, 1.0)),
    ]

    def test_mspline_and_bspline_are_siblings(self):
        self.assertFalse(issubclass(BSpline, MSpline))
        self.assertFalse(issubclass(MSpline, BSpline))
        for cls in (MSpline, BSpline):
            self.assertTrue(issubclass(cls, SplineBasis))

    def test_project_equals_unnormalized_projection_times_norm(self):
        coefs = jnp.asarray(np.random.default_rng(1).normal(size=self.N))
        xs = np.linspace(0.05, 0.95, 100)
        for cls, xrange in self.CASES:
            for normalize in (False, True):
                spl = cls(self.N, xrange=xrange, normalize=normalize)
                with self.subTest(basis=cls.__name__, normalize=normalize):
                    dmat = spl.bases(xs)
                    np.testing.assert_allclose(
                        np.asarray(spl.project(dmat, coefs)),
                        np.asarray(spl._project(dmat, coefs)) * float(spl.norm(coefs)),
                        rtol=1e-6,
                    )

    def test_get_coefficients_recovers_known_coefficients(self):
        spl = BSpline(self.N, normalize=False)
        xs = np.linspace(0.05, 0.95, 300)
        truth = jnp.asarray(np.random.default_rng(0).normal(size=self.N))
        ys = spl.project(spl.bases(xs), truth)
        alpha, fit, design_matrix = spl.get_coefficients(xs, ys)
        np.testing.assert_allclose(np.asarray(alpha), np.asarray(truth), atol=1e-4)
        np.testing.assert_allclose(np.asarray(fit), np.asarray(ys), atol=1e-5)
        self.assertEqual(design_matrix.shape, (xs.size, self.N))

    def test_removed_names_explain_themselves(self):
        import gwinferno.interpolation as interpolation

        for name, expected in [("BasisSpline", "MSpline"), ("RectBivariateBasisSpline", "removed"), ("OOB_VAL", "in_bounds")]:
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as ctx:
                    getattr(interpolation, name)
                self.assertIn(expected, str(ctx.exception))
        with self.assertRaises(AttributeError):
            interpolation.NoSuchThing


class TestDegreeParameter(unittest.TestCase):
    """``degree`` follows scipy's convention; ``order`` stays derived from it."""

    def test_order_is_derived_from_degree(self):
        for degree in (1, 2, 3, 4, 5):
            with self.subTest(degree=degree):
                spl = BSpline(10, degree=degree, normalize=False)
                self.assertEqual(spl.order, degree + 1)
                self.assertEqual(len(spl.knots), spl.N + spl.order)

    def test_degree_matches_scipy(self):
        # scipy's BSpline takes the degree directly, so the two conventions must line up.
        for degree in (2, 3, 4):
            spl = BSpline(10, degree=degree, normalize=False)
            with self.subTest(degree=degree):
                xs = np.linspace(0.0, 1.0, 200)
                expected = scipy_BSpline(spl.knots, np.eye(10), degree)(xs)
                np.testing.assert_allclose(np.asarray(spl.bases(xs)).T, expected, atol=1e-6)

    def test_wrong_knot_count_raises_rather_than_asserting(self):
        # An assert would vanish under python -O, silently building a broken basis.
        with self.assertRaises(ValueError) as ctx:
            BSpline(10, knots=np.linspace(0, 1, 3), degree=3, normalize=False)
        self.assertIn("expected 14 knots", str(ctx.exception))


class TestTransformFlags(unittest.TestCase):
    """The log_x / log_y flags that the Log* subclasses are now thin wrappers around."""

    N = 10

    def test_flags_reproduce_the_named_subclasses(self):
        for cls, kwargs, xrange in [
            (LogXBSpline, dict(log_x=True), (0.01, 1.0)),
            (LogYBSpline, dict(log_y=True), (0.0, 1.0)),
            (LogXLogYBSpline, dict(log_x=True, log_y=True), (0.1, 1.0)),
        ]:
            with self.subTest(basis=cls.__name__):
                named = cls(self.N, xrange=xrange, normalize=True)
                flagged = BSpline(self.N, xrange=xrange, normalize=True, **kwargs)
                xs = np.linspace(max(xrange[0], 1e-3), xrange[1], 200)
                coefs = jnp.asarray(np.random.default_rng(5).normal(size=self.N))
                np.testing.assert_array_equal(np.asarray(named.bases(xs)), np.asarray(flagged.bases(xs)))
                np.testing.assert_array_equal(
                    np.asarray(named.project(named.bases(xs), coefs)),
                    np.asarray(flagged.project(flagged.bases(xs), coefs)),
                )

    def test_log_x_rejects_a_non_positive_lower_bound(self):
        # Previously this silently produced an all -inf knot vector.
        with self.assertRaises(ValueError) as ctx:
            BSpline(self.N, xrange=(0.0, 1.0), log_x=True, normalize=False)
        self.assertIn("positive lower bound", str(ctx.exception))

    def test_mspline_rejects_log_y(self):
        # MSpline.norm is built on basis_vols, which assumes a linear y.
        with self.assertRaises(ValueError) as ctx:
            MSpline(self.N, log_y=True, normalize=False)
        self.assertIn("log_y", str(ctx.exception))
