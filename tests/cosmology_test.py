import unittest

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as astropy_cosmology
from astropy.cosmology import z_at_value

from gwinferno.cosmology import PLANCK_2015_Cosmology as cosmology
from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology as lvk_cosmology
from gwinferno.cosmology import PLANCK_2015_LVK_Ho
from gwinferno.cosmology import PLANCK_2015_LVK_OmegaMatter

astropy_lvk_cosmology = FlatLambdaCDM(H0=PLANCK_2015_LVK_Ho * 1e-3, Om0=PLANCK_2015_LVK_OmegaMatter)


class TestDefaultCosmology(unittest.TestCase):
    def setUp(self) -> None:
        self.zs = np.linspace(1e-9, 3.0, 1000)
        self.dLs_mpc = np.linspace(1e-2, 3000, 1000)

    def tearDown(self) -> None:
        del self.zs
        del self.dLs_mpc

    def test_redshift_to_luminosity_distance(self):
        threshold = 0.01
        dLs_mpc = cosmology.z2DL(self.zs)
        dLs_mpc_astropy = astropy_cosmology.luminosity_distance(self.zs).value
        frac_errs = jnp.abs(dLs_mpc - dLs_mpc_astropy) / dLs_mpc
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_luminosity_distance_to_redshift(self):
        threshold = 0.01
        zs = cosmology.DL2z(self.dLs_mpc)
        zs_astropy = z_at_value(astropy_cosmology.luminosity_distance, self.dLs_mpc * u.Mpc)
        frac_errs = jnp.abs(zs - zs_astropy) / zs
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_luminosity_distance_squared_prior_in_redshift(self):
        threshold = 0.02
        dl_gpc = cosmology.z2DL(self.zs) / 1e3
        dl_gpc_astropy = astropy_cosmology.luminosity_distance(self.zs).to(u.Gpc).value
        dl2_prior = dl_gpc**2 * (dl_gpc / (1 + self.zs) + (1 + self.zs) * cosmology.dDcdz(self.zs) / 1e3)
        dl2_prior_astropy = dl_gpc_astropy**2 * (
            dl_gpc_astropy / (1 + self.zs) + (1 + self.zs) * astropy_cosmology.hubble_distance.to(u.Gpc).value / astropy_cosmology.efunc(self.zs)
        )
        frac_errs = jnp.abs(dl2_prior - dl2_prior_astropy) / dl2_prior
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_redshift_to_differential_comoving_volume(self):
        threshold = 0.00125
        logdVcdzs = cosmology.logdVcdz(self.zs)
        logdVcdzs_astropy = np.log(astropy_cosmology.differential_comoving_volume(self.zs).value) + np.log(4.0 * np.pi)
        frac_errs = jnp.abs(logdVcdzs - logdVcdzs_astropy) / logdVcdzs
        self.assertTrue(jnp.all(frac_errs < threshold))


class TestLVKCosmology(unittest.TestCase):
    def setUp(self) -> None:
        self.zs = np.linspace(1e-9, 3.0, 1000)
        self.dLs_mpc = np.linspace(1e-2, 3000, 1000)

    def tearDown(self) -> None:
        del self.zs
        del self.dLs_mpc

    def test_redshift_to_luminosity_distance(self):
        threshold = 0.01
        dLs_mpc = lvk_cosmology.z2DL(self.zs)
        dLs_mpc_astropy = astropy_lvk_cosmology.luminosity_distance(self.zs).value
        frac_errs = jnp.abs(dLs_mpc - dLs_mpc_astropy) / dLs_mpc
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_luminosity_distance_to_redshift(self):
        threshold = 0.01
        zs = lvk_cosmology.DL2z(self.dLs_mpc)
        zs_astropy = z_at_value(astropy_lvk_cosmology.luminosity_distance, self.dLs_mpc * u.Mpc)
        frac_errs = jnp.abs(zs - zs_astropy) / zs
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_luminosity_distance_squared_prior_in_redshift(self):
        threshold = 0.02
        dl_gpc = lvk_cosmology.z2DL(self.zs) / 1e3
        dl_gpc_astropy = astropy_lvk_cosmology.luminosity_distance(self.zs).to(u.Gpc).value
        dl2_prior = dl_gpc**2 * (dl_gpc / (1 + self.zs) + (1 + self.zs) * lvk_cosmology.dDcdz(self.zs) / 1e3)
        dl2_prior_astropy = dl_gpc_astropy**2 * (
            dl_gpc_astropy / (1 + self.zs)
            + (1 + self.zs) * astropy_lvk_cosmology.hubble_distance.to(u.Gpc).value / astropy_lvk_cosmology.efunc(self.zs)
        )
        frac_errs = jnp.abs(dl2_prior - dl2_prior_astropy) / dl2_prior
        self.assertTrue(jnp.all(frac_errs < threshold))

    def test_redshift_to_differential_comoving_volume(self):
        threshold = 0.00125
        logdVcdzs = lvk_cosmology.logdVcdz(self.zs)
        logdVcdzs_astropy = np.log(astropy_lvk_cosmology.differential_comoving_volume(self.zs).value) + np.log(4.0 * np.pi)
        frac_errs = jnp.abs(logdVcdzs - logdVcdzs_astropy) / logdVcdzs
        self.assertTrue(jnp.all(frac_errs < threshold))
