import unittest

import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid

from gwinferno.interpolation import BSpline
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel


class TestPowerlawSplineRedshiftModel(unittest.TestCase):
    """The spline perturbation must come from the basis, so the model is right for any basis choice.

    Uses synthetic redshifts so it runs without the downloaded fixtures.
    """

    N = 8
    LAMB = 2.7

    def setUp(self) -> None:
        rng = np.random.default_rng(21)
        self.z_pe = jnp.asarray(rng.uniform(0.05, 1.4, size=(4, 50)))
        # injections on a dense grid, so the model can be integrated over them directly
        self.z_grid = jnp.linspace(0.05, 1.4, 4000)
        self.coefs = jnp.asarray(rng.normal(size=self.N))
        self.positive_coefs = jnp.asarray(rng.uniform(0.5, 1.5, size=self.N))

    def test_rate_density_is_normalized_for_every_basis(self):
        for basis, coefs in [
            (LogXLogYBSpline, self.coefs),
            (LogYBSpline, self.coefs),
            (LogXBSpline, self.positive_coefs),  # linear-y bases need coefficients keeping f(z) positive
            (BSpline, self.positive_coefs),
        ]:
            with self.subTest(basis=basis.__name__):
                model = PowerlawSplineRedshiftModel(self.N, self.z_pe, self.z_grid, basis=basis)
                p = model(self.z_grid, self.LAMB, coefs)
                self.assertGreaterEqual(float(jnp.min(p)), 0.0)
                self.assertAlmostEqual(float(trapezoid(p, self.z_grid)), 1.0, places=3)

    def test_default_basis_reproduces_the_previous_perturbation(self):
        # The model used to exponentiate a LogXBSpline projection itself; the log-y default now does
        # that inside the basis, so inside the domain nothing should move.
        new = PowerlawSplineRedshiftModel(self.N, self.z_pe, self.z_grid)
        old = PowerlawSplineRedshiftModel(self.N, self.z_pe, self.z_grid, basis=LogXBSpline)
        expected = jnp.power(1.0 + old.zs, self.LAMB) * jnp.exp(old.interpolator.project(old.norm_design_matrix, self.coefs))
        np.testing.assert_allclose(np.asarray(new.rate_of_z(self.LAMB, self.coefs))[1:-1], np.asarray(expected)[1:-1], rtol=1e-5)

    def test_zero_probability_below_zmin(self):
        # zmin is the larger of the PE and injection minima, so PE samples can fall below it.
        z_pe = self.z_pe.at[0, :5].set(0.01)
        model = PowerlawSplineRedshiftModel(self.N, z_pe, self.z_grid)
        p = model(z_pe, self.LAMB, self.coefs)
        np.testing.assert_array_equal(np.asarray(p[0, :5]), np.zeros(5))
        self.assertTrue(bool(jnp.all(p[0, 5:] > 0)))
