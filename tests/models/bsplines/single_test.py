import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from astropy.cosmology import Planck15 as Planck15_astropy
from jax.scipy.integrate import trapezoid

from gwinferno.cosmology import PLANCK_2015_LVK_Cosmology
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
from gwinferno.models.bsplines.single import Planck15
from gwinferno.preprocess.data_collection import load_injection_dataset


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


class TestPPD(unittest.TestCase):
    """ppd must reproduce the model as fit, including a non-default basis and custom breakpoints.

    Uses synthetic samples so it runs without the downloaded fixtures.
    """

    N = 8
    N_DRAWS = 5

    def setUp(self) -> None:
        self.coefs = jnp.asarray(np.random.default_rng(3).normal(size=(self.N_DRAWS, self.N)))

    def _assert_ppd_matches_model(self, model, x):
        ppd = model.ppd(x, self.coefs)
        self.assertEqual(ppd.shape, (self.N_DRAWS, len(x)))
        expected = jax.vmap(lambda cs: model(cs, pe_samples=True))(self.coefs)
        np.testing.assert_allclose(np.asarray(ppd), np.asarray(expected), rtol=1e-6)
        return ppd

    def test_matches_model_with_custom_basis_and_breakpoints(self):
        # Non-uniform breakpoints in q; n_df - degree + 1 = 6 of them for a cubic with 8 dof.
        breakpoints = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
        x = jnp.linspace(0.01, 1.2, 500)  # extends past both ends of the domain (0.05, 1)
        model = BSplineRatio(self.N, x, x, qmin=0.05, basis=LogXLogYBSpline, breakpoints=breakpoints)
        ppd = self._assert_ppd_matches_model(model, x)

        outside = (x < 0.05) | (x > 1.0)
        self.assertTrue(np.all(np.asarray(ppd[:, outside]) == 0.0))
        self.assertTrue(np.all(np.asarray(ppd[:, ~outside]) > 0.0))
        # The default model differs, which is why helpers that rebuild it from defaults are deprecated.
        default = BSplineRatio(self.N, x, x, qmin=0.05)
        self.assertFalse(np.allclose(np.asarray(default.ppd(x, self.coefs)), np.asarray(ppd)))

    def test_symmetric_chi_effective_matches_model(self):
        x = jnp.linspace(-1.0, 1.0, 301)
        self._assert_ppd_matches_model(BSplineSymmetricChiEffective(self.N, x, x), x)

    def test_redshift_ppd_times_rate_follows_likelihood_convention(self):
        # hierarchical_likelihood reports rate = N / det_eff / VT with VT = normalization(cs) / 1e9 * Tobs,
        # so rate * r(z) integrated over comoving volume and source-frame time must give rate * VT.
        z = jnp.linspace(0.01, 2.0, 2000)
        dVdc = PLANCK_2015_LVK_Cosmology.dVcdz(z)
        model = BSplineRedshift(self.N, z, z, dVdc, dVdc)
        rate, Tobs = 30.0, 1.5
        r_of_z = model.ppd(z, self.coefs)
        for cs, r in zip(self.coefs, r_of_z):
            vt = model.normalization(cs) / 1e9 * Tobs
            n_expected = trapezoid(rate * r * dVdc / (1 + z), z) / 1e9 * Tobs
            self.assertAlmostEqual(float(n_expected / (rate * vt)), 1.0, places=3)
            # and it is the same r(z) that the likelihood weights use
            np.testing.assert_allclose(np.asarray(r * dVdc / (1 + z) / model.normalization(cs)), np.asarray(model(cs)), rtol=1e-5)
