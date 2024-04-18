import unittest

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck15
from jax import random
from scipy.integrate import cumtrapz as scipy_cumtrapz
from scipy.stats import truncnorm

from gwinferno.interpolation import BSpline
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.numpyro_distributions import BSplineDistribution
from gwinferno.numpyro_distributions import Cosine
from gwinferno.numpyro_distributions import Powerlaw
from gwinferno.numpyro_distributions import PowerlawRedshift
from gwinferno.numpyro_distributions import PSplineCoeficientPrior
from gwinferno.numpyro_distributions import Sine
from gwinferno.numpyro_distributions import cumtrapz


class TestJaxCumtrapz(unittest.TestCase):
    def setUp(self) -> None:
        self.gr = np.linspace(0, 1, 50)
        self.tn = truncnorm(-1, 1, loc=0.5, scale=0.1)

    def test_cumtrapz(self):
        scipy_int = scipy_cumtrapz(np.exp(self.tn.logpdf(self.gr)), self.gr).tolist()
        gwinf_int = cumtrapz(jnp.exp(self.tn.logpdf(self.gr)), self.gr).tolist()
        for s, g in zip(scipy_int, gwinf_int):
            self.assertAlmostEqual(s, g, places=4)

    def tearDown(self) -> None:
        del self.gr
        del self.tn


class TestNPDistributions(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = jnp.linspace(0.001, 1, 1000)
        self.x_interps = jnp.linspace(0.001, 1, 10)
        self.y_interps = random.normal(random.PRNGKey(0), shape=(10,))
        self.gr = jnp.linspace(0, 1, 1000)
        self.cs_pn = random.normal(random.PRNGKey(1), shape=(20,))
        self.cs = random.uniform(random.PRNGKey(2), shape=(20,))

    def tearDown(self) -> None:
        del self.grid
        del self.x_interps
        del self.y_interps
        del self.gr
        del self.cs

    def test_cosine(self):
        d = Cosine(minimum=-np.pi / 2.0, maximum=np.pi / 2.0)
        grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, 1000)
        lpdfs = d.log_prob(grid)
        self.assertAlmostEqual(jnp.trapz(jnp.exp(lpdfs), grid), 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= -np.pi / 2.0) & (samps <= np.pi / 2.0)))

    def test_sine(self):
        d = Sine(minimum=0.0, maximum=np.pi)
        grid = np.linspace(0, np.pi, 1000)
        lpdfs = d.log_prob(grid)
        self.assertAlmostEqual(jnp.trapz(jnp.exp(lpdfs), grid), 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.0) & (samps <= np.pi)))

    def test_powerlaw(self):
        d = Powerlaw(alpha=1.0, minimum=0.001, maximum=1.0)
        lpdfs = d.log_prob(self.grid)
        norm = jnp.trapz(jnp.exp(lpdfs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.001) & (samps <= 1.0)))

    def test_powerlaw_redshift(self):
        z_grid = jnp.linspace(0.001, 1, 1000)
        dVcdz_grid = Planck15.differential_comoving_volume(z_grid).value * 4.0 * jnp.pi
        d = PowerlawRedshift(lamb=0.0, maximum=1.0, zgrid=z_grid, dVcdz=dVcdz_grid)
        lpdfs = d.log_prob(self.grid)
        norm = jnp.trapz(jnp.exp(lpdfs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.001) & (samps <= 1.0)))

    def test_bspline_distribution(self):
        grid_dmat = BSpline(20, normalize=True).bases(self.gr)
        d = BSplineDistribution(minimum=0.0, maximum=1.0, cs=self.cs, grid=self.gr, grid_dmat=grid_dmat)
        lpdfs = d.log_prob(self.gr)
        norm = jnp.trapz(jnp.exp(lpdfs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.0) & (samps <= 1.0)))

    def test_pspline_distribution(self):
        cs = PSplineCoeficientPrior(20, 1.0).sample(random.PRNGKey(0), sample_shape=(1,))
        grid_dmat = BSpline(20, normalize=True).bases(self.gr)
        d = BSplineDistribution(minimum=0.0, maximum=1.0, cs=cs, grid=self.gr, grid_dmat=grid_dmat)
        lpdfs = d.log_prob(self.gr)
        norm = jnp.trapz(jnp.exp(lpdfs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.0) & (samps <= 1.0)))

    def test_logy_bspline_distribution(self):
        grid_dmat = LogYBSpline(20, normalize=True).bases(self.gr)
        d = BSplineDistribution(minimum=0.0, maximum=1.0, cs=self.cs_pn, grid=self.gr, grid_dmat=grid_dmat)
        lpdfs = d.log_prob(self.gr)
        norm = jnp.trapz(jnp.exp(lpdfs), self.gr)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.0) & (samps <= 1.0)))

    def test_logx_bspline_distribution(self):
        grid_dmat = LogXBSpline(20, normalize=True).bases(self.grid)
        d = BSplineDistribution(minimum=0.001, maximum=1.0, cs=self.cs, grid=self.grid, grid_dmat=grid_dmat)
        lpdfs = d.log_prob(self.grid)
        norm = jnp.trapz(jnp.exp(lpdfs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.001) & (samps <= 1.0)))

    def test_logxy_bspline_distribution(self):
        grid_dmat = LogXLogYBSpline(20, normalize=True).bases(self.grid)
        d = BSplineDistribution(minimum=0.001, maximum=1.0, cs=self.cs_pn, grid=self.grid, grid_dmat=grid_dmat)
        lpdfs = d.log_prob(self.grid)
        norm = jnp.trapz(jnp.exp(lpdfs), self.grid)
        self.assertAlmostEqual(norm, 1.0, places=4)
        samps = d.sample(random.PRNGKey(0), sample_shape=(100,))
        self.assertTrue(jnp.all((samps >= 0.001) & (samps <= 1.0)))
