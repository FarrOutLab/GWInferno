import os
import unittest

import jax.numpy as jnp
import numpy as np
import xarray as xr
from astropy.cosmology import Planck15 as Planck15_astropy
from jax.scipy.integrate import trapezoid

from gwinferno.interpolation import BSpline
from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
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


class TestBSplineRedshiftNormalization(unittest.TestCase):
    """Checks that R(z) is a properly normalized rate density for every basis choice.

    Uses synthetic samples so it runs without the downloaded ``tests/data`` fixtures. Note that
    ``coefs = ones(n)`` cannot detect these failures: the basis is a partition of unity, so
    ``sum_i c_i B_i(z)`` is then constant in z and every normalization error degenerates to an
    overall factor. Randomly drawn coefficients are required.
    """

    LOG_Y_BASES = (LogYBSpline, LogXLogYBSpline)
    LINEAR_Y_BASES = (BSpline, LogXBSpline)

    def setUp(self) -> None:
        self.n_splines = 8
        self.z = jnp.linspace(0.01, 2.0, 1000)
        self.dVdc = PLANCK_2015_LVK_Cosmology.dVcdz(self.z)
        rng = np.random.default_rng(12345)
        # Linear-y bases model r(z) directly, so they need coefficients that keep it positive.
        self.coefs = jnp.asarray(rng.normal(size=self.n_splines))
        self.positive_coefs = jnp.asarray(rng.uniform(0.5, 1.5, size=self.n_splines))

    def _model(self, basis, **kwargs):
        return BSplineRedshift(self.n_splines, self.z, self.z, self.dVdc, self.dVdc, basis=basis, **kwargs)

    def test_rate_integrates_to_unity_for_every_basis(self):
        for basis in self.LOG_Y_BASES + self.LINEAR_Y_BASES:
            coefs = self.coefs if basis in self.LOG_Y_BASES else self.positive_coefs
            for normalize in (False, True):
                with self.subTest(basis=basis.__name__, normalize=normalize):
                    rate = self._model(basis, normalize=normalize)(coefs, pe_samples=True)
                    self.assertAlmostEqual(float(trapezoid(rate, self.z)), 1.0, places=4)

    def test_result_is_independent_of_basis_normalization(self):
        # normalize scales r(z) by a constant that cancels between __call__ and normalization().
        for basis in self.LOG_Y_BASES + self.LINEAR_Y_BASES:
            coefs = self.coefs if basis in self.LOG_Y_BASES else self.positive_coefs
            with self.subTest(basis=basis.__name__):
                unnormed = np.asarray(self._model(basis, normalize=False)(coefs, pe_samples=True))
                normed = np.asarray(self._model(basis, normalize=True)(coefs, pe_samples=True))
                np.testing.assert_allclose(normed, unnormed, rtol=1e-4)

    def test_log_y_bases_are_invariant_under_constant_coefficient_shift(self):
        # Partition of unity gives sum_i (c_i + d) B_i(z) = sum_i c_i B_i(z) + d, so for a log-y
        # basis a constant shift is an overall factor exp(d) that normalization() must cancel.
        for basis in self.LOG_Y_BASES:
            model = self._model(basis, normalize=False)
            rate = np.asarray(model(self.coefs, pe_samples=True))
            for shift in (1.0, -2.5):
                with self.subTest(basis=basis.__name__, shift=shift):
                    shifted = np.asarray(model(self.coefs + shift, pe_samples=True))
                    np.testing.assert_allclose(shifted, rate, rtol=1e-4)

    def test_out_of_range_samples_have_zero_rate(self):
        # Samples outside xrange must contribute no rate. Projecting through the interpolator
        # gives 0 there; exponentiating the zero-filled pdf would wrongly give 1.
        z = jnp.array([0.5, 1.0, 5.0])  # 5.0 lies outside the default xrange=(1e-4, 2.3)
        dVdc = PLANCK_2015_LVK_Cosmology.dVcdz(z)
        model = BSplineRedshift(self.n_splines, z, z, dVdc, dVdc)
        self.assertEqual(float(model(self.coefs, pe_samples=True)[2]), 0.0)
