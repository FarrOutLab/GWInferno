import os
import unittest

import jax.numpy as jnp
import numpy as np
import xarray as xr
from astropy.cosmology import Planck15 as Planck15_astropy
from jax.scipy.integrate import trapezoid

from gwinferno.models.bsplines.single import Base1DBSplineModel
from gwinferno.models.bsplines.single import BSplineChiEffective
from gwinferno.models.bsplines.single import BSplineChiPrecess
from gwinferno.models.bsplines.single import BSplineMass
from gwinferno.models.bsplines.single import BSplineRatio
from gwinferno.models.bsplines.single import BSplineRedshift
from gwinferno.models.bsplines.single import BSplineSpinMagnitude
from gwinferno.models.bsplines.single import BSplineSpinTilt
from gwinferno.models.bsplines.single import BSplineSymmetricChiEffective
from gwinferno.preprocess.data_collection import load_injection_dataset
from gwinferno.models.bsplines.single import Planck15
from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology


class TestBase1DBSplineModel(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("~/unit_tests/data"):
            pref = "~/unit_tests/data"
        else:
            pref = "tests/data"
            self.data_dir = pref
        self.inj_file = f"{pref}/injections.h5"
        self.pedict = self.load_data()
        self.injdict = self.load_injection_dataset(through_o4a=False, through_o3=True)
        self.nsplines = 10
        self.coefs = jnp.ones((self.nsplines,))

    def tearDown(self) -> None:
        del self.data_dir
        del self.inj_file
        del self.pedict
        del self.injdict
        del self.nsplines

    def load_injection_dataset(self, **kwargs):
        p_names = ["mass_1", "mass_ratio", "redshift", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        injections = load_injection_dataset(self.inj_file, p_names, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"]).to_array()
        injdata = jnp.asarray(injections.data[0])
        injdict = {k: injdata[i] for i, k in enumerate(injections.param.values)}
        return injdict

    def load_data(self):
        loaded_dataset = xr.load_dataset(f"{self.data_dir}/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        pedata = jnp.asarray(dataarray.data)
        pedict = {k: pedata[:, i, :] for i, k in enumerate(dataarray.param.values)}
        return pedict

    def spline_shape(self, model, pe_x, inj_x, redshift=False):

        if redshift:
            pe_dvcdz = jnp.array(Planck15_astropy.differential_comoving_volume(pe_x).value * 4 * jnp.pi)
            inj_dvcdz = jnp.array(Planck15_astropy.differential_comoving_volume(inj_x).value * 4 * jnp.pi)
            bspline = model(self.nsplines, pe_x, inj_x, pe_dvcdz, inj_dvcdz)
            pe_pdf = bspline(self.coefs, pe_samples=True)
            inj_pdf = bspline(self.coefs, pe_samples=False)

        else:
            bspline = model(self.nsplines, pe_x, inj_x)
            pe_pdf = bspline(self.coefs, pe_samples=True)
            inj_pdf = bspline(self.coefs, pe_samples=False)

        self.assertEqual(pe_x.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_x.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

    def test_functions(self):
        self.spline_shape(BSplineSpinMagnitude, self.pedict["a_1"], self.injdict["a_1"])
        self.spline_shape(Base1DBSplineModel, self.pedict["mass_1"], self.injdict["mass_1"])
        self.spline_shape(BSplineSpinTilt, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"])
        self.spline_shape(BSplineChiEffective, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"])
        self.spline_shape(BSplineChiEffective, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"])
        self.spline_shape(BSplineSymmetricChiEffective, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"])
        self.spline_shape(BSplineChiPrecess, self.pedict["a_1"], self.injdict["a_1"])
        self.spline_shape(BSplineRatio, self.pedict["mass_ratio"], self.injdict["mass_ratio"])
        self.spline_shape(BSplineMass, self.pedict["mass_1"], self.injdict["mass_1"])
        self.spline_shape(BSplineRedshift, self.pedict["redshift"], self.injdict["redshift"], redshift=True)

    def test_cosmology(self):
        self.assertIs(Planck15, PLANCK_2015_LVK_Cosmology)


class TestOneDimensionalModelsAreDensities(unittest.TestCase):
    """Every 1-D shape model must be a normalized, non-negative density.

    The population model is separable, dN/dV(theta) = R * p_pop(theta), so each factor has to
    integrate to 1 over its domain. If one does not, the shape posterior is unaffected, since an
    overall factor cancels between the numerator and the selection term, but the inferred rate is
    divided by that factor, which varies with the coefficients and so distorts the rate posterior.

    A linear-y basis cannot satisfy this for unconstrained real coefficients

    Uses synthetic samples so it runs without the downloaded fixtures.
    """

    N = 8
    N_DRAWS = 25

    def _check(self, model, grid, name):
        rng = np.random.default_rng(17)
        for draw in range(self.N_DRAWS):
            pdf = model(jnp.asarray(rng.normal(size=self.N)), pe_samples=True)
            with self.subTest(model=name, draw=draw):
                self.assertGreaterEqual(float(jnp.min(pdf)), 0.0, msg=f"{name} produced a negative density")
                self.assertAlmostEqual(float(trapezoid(pdf, grid)), 1.0, places=3, msg=f"{name} is not normalized")

    def test_spin_models_are_densities(self):
        for cls, (lo, hi) in [
            (BSplineSpinMagnitude, (0.0, 1.0)),
            (BSplineSpinTilt, (-1.0, 1.0)),
            (BSplineChiEffective, (-1.0, 1.0)),
            (BSplineChiPrecess, (0.0, 1.0)),
        ]:
            grid = jnp.linspace(lo, hi, 4000)
            self._check(cls(self.N, grid, grid), grid, cls.__name__)

    def test_symmetric_chi_effective_is_a_density(self):
        # It evaluates at |chi_eff| on [0, 1] and halves, which integrates to 1 over [-1, 1] only
        # if the underlying spline is itself normalized.
        grid = jnp.linspace(-1.0, 1.0, 6000)
        self._check(BSplineSymmetricChiEffective(self.N, grid, grid), grid, "BSplineSymmetricChiEffective")
