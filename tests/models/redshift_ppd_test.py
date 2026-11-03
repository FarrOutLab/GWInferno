import unittest

import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid

from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology
from gwinferno.models.bsplines.single import BSplineRedshift
from gwinferno.models.parametric.parametric import PowerlawRedshiftModel
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel
from gwinferno.postprocess.calculations import calculate_powerlaw_rate_of_z_ppds


class TestRedshiftPPDInterface(unittest.TestCase):
    """Every redshift model's ppd(z, *hyperparameters) takes the hyperparameters of its normalization,
    with a leading draws axis, and returns r(z) such that rate * r(z) is the source-frame rate density.

    hierarchical_likelihood reports rate = N / det_eff / VT with VT = normalization(*hyperparameters) / 1e9 * Tobs,
    so the convention holds iff integrating r(z) over comoving volume and source-frame time gives normalization.
    Uses synthetic samples so it runs without the downloaded fixtures.
    """

    N_SPLINES = 8
    N_DRAWS = 4

    def setUp(self) -> None:
        z = jnp.linspace(0.01, 2.0, 1000)
        dVdc = PLANCK_2015_LVK_Cosmology.dVcdz(z)
        rng = np.random.default_rng(7)
        lamb = jnp.asarray(rng.uniform(1.0, 4.0, size=self.N_DRAWS))
        cs = jnp.asarray(rng.normal(size=(self.N_DRAWS, self.N_SPLINES)))
        # (model, hyperparameter draws in the order its normalization takes them)
        self.cases = {
            "PowerlawRedshiftModel": (PowerlawRedshiftModel(z, z), (lamb,)),
            "PowerlawSplineRedshiftModel": (PowerlawSplineRedshiftModel(self.N_SPLINES, z, z), (lamb, cs)),
            "BSplineRedshift": (BSplineRedshift(self.N_SPLINES, z, z, dVdc, dVdc), (cs,)),
        }

    def test_rate_times_ppd_follows_likelihood_convention(self):
        for name, (model, params) in self.cases.items():
            zs = jnp.linspace(model.zmin, model.zmax, 4000)
            r = model.ppd(zs, *params)
            with self.subTest(model=name):
                self.assertEqual(r.shape, (self.N_DRAWS, len(zs)))
                for i in range(self.N_DRAWS):
                    draw = [p[i] for p in params]
                    integral = trapezoid(r[i] * PLANCK_2015_LVK_Cosmology.dVcdz(zs) / (1 + zs), zs)
                    self.assertAlmostEqual(float(integral / model.normalization(*draw)), 1.0, places=3)

    def test_zero_above_zmax(self):
        for name, (model, params) in self.cases.items():
            with self.subTest(model=name):
                beyond = jnp.array([float(model.zmax) + 0.5, 5.0])
                self.assertTrue(np.all(np.asarray(model.ppd(beyond, *params)) == 0.0))

    def test_powerlaw_spline_ppd_matches_rate_of_z(self):
        model, (lamb, cs) = self.cases["PowerlawSplineRedshiftModel"]
        r = model.ppd(model.zs, lamb, cs)
        for i in range(self.N_DRAWS):
            np.testing.assert_allclose(np.asarray(r[i]), np.asarray(model.rate_of_z(lamb[i], cs[i])), rtol=1e-6)

    def test_calculate_powerlaw_rate_of_z_ppds_is_deprecated(self):
        model, (lamb,) = self.cases["PowerlawRedshiftModel"]
        with self.assertWarns(FutureWarning):
            calculate_powerlaw_rate_of_z_ppds(lamb, jnp.ones_like(lamb), model)
