import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.scipy.integrate import trapezoid

from gwinferno.models.bsplines.separable import BSplineEffectiveSpinDims
from gwinferno.models.bsplines.separable import BSplineIIDComponentMasses
from gwinferno.models.bsplines.separable import BSplineIIDSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIIDSpinTilts
from gwinferno.models.bsplines.separable import BSplineIndependentComponentMasses
from gwinferno.models.bsplines.separable import BSplineIndependentSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIndependentSpinTilts
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.bsplines.separable import BSplinePrimaryPowerlawRatio
from gwinferno.models.bsplines.separable import PLPeakPrimaryBSplineRatio
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
        self.mmin = 5.0
        self.mmax = 100.0

    def tearDown(self) -> None:
        del self.data_dir
        del self.inj_file
        del self.pedict
        del self.injdict
        del self.nsplines
        del self.mmin
        del self.mmax

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

    def spin_spline_shape(self, model, pe_x, inj_x, IID=True):

        if IID:
            bspline = model(self.nsplines, pe_x, pe_x, inj_x, inj_x)
            pe_pdf = bspline(self.coefs, pe_samples=True)
            inj_pdf = bspline(self.coefs, pe_samples=False)
        else:
            bspline = model(self.nsplines, self.nsplines, pe_x, pe_x, inj_x, inj_x)
            pe_pdf = bspline(self.coefs, self.coefs, pe_samples=True)
            inj_pdf = bspline(self.coefs, self.coefs, pe_samples=False)

        self.assertEqual(pe_x.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_x.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

    def test_spin_functions(self):
        self.spin_spline_shape(BSplineIIDSpinMagnitudes, self.pedict["a_1"], self.injdict["a_1"])
        self.spin_spline_shape(BSplineIIDSpinTilts, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"])
        self.spin_spline_shape(BSplineIndependentSpinMagnitudes, self.pedict["a_1"], self.injdict["a_1"], IID=False)
        self.spin_spline_shape(BSplineIndependentSpinTilts, self.pedict["cos_tilt_1"], self.injdict["cos_tilt_1"], IID=False)

    def test_BSplinePrimaryPowerlawRatio(self):
        beta = 3.0

        pe_m1 = self.pedict["mass_1"]
        inj_m1 = self.injdict["mass_1"]
        pe_q = self.pedict["mass_ratio"]
        inj_q = self.injdict["mass_ratio"]
        bspline_pl = BSplinePrimaryPowerlawRatio(self.nsplines, pe_m1, inj_m1, mmin=self.mmin, mmax=self.mmax)

        pe_pdf = bspline_pl(pe_m1, pe_q, beta, self.mmin, self.coefs, pe_samples=True)
        inj_pdf = bspline_pl(inj_m1, inj_q, beta, self.mmin, self.coefs, pe_samples=False)

        self.assertEqual(pe_m1.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_m1.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

        pe_nonzero_outside = jnp.sum(jnp.where(jnp.less(pe_m1, self.mmin) | jnp.greater(pe_m1, self.mmax), pe_pdf, 0))
        inj_nonzero_outside = jnp.sum(jnp.where(jnp.less(inj_m1, self.mmin) | jnp.greater(inj_m1, self.mmax), inj_pdf, 0))

        self.assertEqual(pe_nonzero_outside, 0, msg="PE PDF not properly truncated")
        self.assertEqual(inj_nonzero_outside, 0, msg="Inj PDF not properly truncated")

    def test_PLPeakPrimaryBSplineRatio(self):

        pe_m1 = self.pedict["mass_1"]
        inj_m1 = self.injdict["mass_1"]
        pe_q = self.pedict["mass_ratio"]
        inj_q = self.injdict["mass_ratio"]

        plpeak_spline = PLPeakPrimaryBSplineRatio(self.nsplines, pe_q, inj_q)

        pe_pdf = plpeak_spline(
            pe_m1, alpha=-3, mmin=self.mmin, mmax=self.mmax, peak_mean=10.0, peak_sd=3.0, peak_frac=0.8, coefs=self.coefs, pe_samples=True
        )
        inj_pdf = plpeak_spline(
            inj_m1, alpha=-3, mmin=self.mmin, mmax=self.mmax, peak_mean=10.0, peak_sd=3.0, peak_frac=0.8, coefs=self.coefs, pe_samples=False
        )

        self.assertEqual(pe_m1.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_m1.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

        pe_nonzero_outside = jnp.sum(jnp.where(jnp.less(pe_m1, self.mmin) | jnp.greater(pe_m1, self.mmax), pe_pdf, 0))
        inj_nonzero_outside = jnp.sum(jnp.where(jnp.less(inj_m1, self.mmin) | jnp.greater(inj_m1, self.mmax), inj_pdf, 0))

        self.assertEqual(pe_nonzero_outside, 0, msg="PE PDF not properly truncated")
        self.assertEqual(inj_nonzero_outside, 0, msg="Inj PDF not properly truncated")

    def test_BSplinePrimaryBSplineRatio(self):

        pe_m1 = self.pedict["mass_1"]
        inj_m1 = self.injdict["mass_1"]
        pe_q = self.pedict["mass_ratio"]
        inj_q = self.injdict["mass_ratio"]

        spline = BSplinePrimaryBSplineRatio(
            self.nsplines, self.nsplines, pe_m1, inj_m1, pe_q, inj_q, m1min=self.mmin, m2min=self.mmin, mmax=self.mmax
        )

        pe_pdf = spline(self.coefs, self.coefs, pe_samples=True)
        inj_pdf = spline(self.coefs, self.coefs, pe_samples=False)

        self.assertEqual(pe_m1.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_m1.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

        pe_nonzero_outside = jnp.sum(jnp.where(jnp.less(pe_m1, self.mmin) | jnp.greater(pe_m1, self.mmax), pe_pdf, 0))
        inj_nonzero_outside = jnp.sum(jnp.where(jnp.less(inj_m1, self.mmin) | jnp.greater(inj_m1, self.mmax), inj_pdf, 0))

        self.assertEqual(pe_nonzero_outside, 0, msg="PE PDF not properly truncated")
        self.assertEqual(inj_nonzero_outside, 0, msg="Inj PDF not properly truncated")

    def component_mass(self, model, IID=True, **kwargs):

        pe_m1 = self.pedict["mass_1"]
        inj_m1 = self.injdict["mass_1"]
        pe_m2 = self.pedict["mass_ratio"] * self.pedict["mass_1"]
        inj_m2 = self.injdict["mass_ratio"] * self.injdict["mass_1"]

        if IID:
            spline = model(self.nsplines, pe_m1, pe_m2, inj_m1, inj_m2, **kwargs)
            pe_pdf = spline(self.coefs, beta=3.0, pe_samples=True)
            inj_pdf = spline(self.coefs, beta=3.0, pe_samples=False)
        else:
            spline = model(self.nsplines, self.nsplines, pe_m1, pe_m2, inj_m1, inj_m2, **kwargs)
            pe_pdf = spline(self.coefs, self.coefs, beta=3.0, pe_samples=True)
            inj_pdf = spline(self.coefs, self.coefs, beta=3.0, pe_samples=False)

        self.assertEqual(pe_m1.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_m1.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

    def test_BSplineComponentMass(self):

        self.component_mass(BSplineIIDComponentMasses, IID=True, mmin=self.mmin, mmax=self.mmax)
        self.component_mass(BSplineIndependentComponentMasses, IID=False, mmin1=self.mmin, mmax1=self.mmax, mmin2=self.mmin, mmax2=self.mmax)

    def test_BSplineEffectiveSpinDims(self):

        spline = BSplineEffectiveSpinDims(
            self.nsplines, self.nsplines, self.pedict["cos_tilt_1"], self.pedict["a_1"], self.injdict["cos_tilt_1"], self.injdict["a_1"]
        )

        pe_pdf = spline(self.coefs, self.coefs, pe_samples=True)
        inj_pdf = spline(self.coefs, self.coefs, pe_samples=False)

        self.assertEqual(self.pedict["a_1"].shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(self.injdict["cos_tilt_1"].shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")


class TestBSplinePrimaryBSplineRatioPPD(unittest.TestCase):
    """ppd delegates to the component models, keeping per-model settings, using synthetic samples."""

    n_m, n_q, n_draws = 10, 6, 4

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.mcoefs = jnp.asarray(rng.normal(size=(self.n_draws, self.n_m)))
        self.qcoefs = jnp.asarray(rng.normal(size=(self.n_draws, self.n_q)))

    def test_ppd_matches_components_without_m2min(self):
        n_m, n_q, n_draws = self.n_m, self.n_q, self.n_draws
        mcoefs, qcoefs = self.mcoefs, self.qcoefs
        m1 = jnp.linspace(3.0, 120.0, 400)
        q = jnp.linspace(0.01, 1.0, 300)
        knots_m = np.geomspace(1.0, 300.0, n_m + 4)  # custom, non-uniform knots for the mass spline
        model = BSplinePrimaryBSplineRatio(n_m, n_q, m1, m1, q, q, mmax=100.0, m1min=5.0, m2min=None, qmin=0.05, kwargs_m={"knots": knots_m})

        ppds = model.ppd(m1, q, mcoefs, qcoefs)

        self.assertEqual(set(ppds), {"mass_1", "mass_ratio"})
        self.assertEqual(ppds["mass_1"].shape, (n_draws, len(m1)))
        self.assertEqual(ppds["mass_ratio"].shape, (n_draws, len(q)))
        for i in range(n_draws):
            np.testing.assert_allclose(np.asarray(ppds["mass_1"][i]), np.asarray(model.primary_model(mcoefs[i])), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(ppds["mass_ratio"][i]), np.asarray(model.ratio_model(qcoefs[i])), rtol=1e-6)
        self.assertTrue(np.all(np.asarray(ppds["mass_1"][:, (m1 < 5.0) | (m1 > 100.0)]) == 0.0))
        self.assertTrue(np.all(np.asarray(ppds["mass_ratio"][:, q < 0.05]) == 0.0))

    def test_m2min_truncates_and_normalizes_the_joint(self):
        m2min, mmax = 5.0, 100.0
        ms = jnp.linspace(m2min, mmax, 800)
        qs = jnp.linspace(m2min / mmax, 1.0, 800)
        M, Q = jnp.meshgrid(ms, qs)  # samples on a grid, so the joint can be integrated
        model = BSplinePrimaryBSplineRatio(self.n_m, self.n_q, M, M, Q, Q, mmax=mmax, m1min=m2min, m2min=m2min)
        ppds = model.ppd(ms, qs, self.mcoefs, self.qcoefs)

        for i in range(self.n_draws):
            joint = model(self.mcoefs[i], self.qcoefs[i])
            with self.subTest(draw=i):
                self.assertTrue(np.all(np.asarray(joint)[np.asarray(Q * M) < m2min] == 0.0))
                self.assertAlmostEqual(float(trapezoid(trapezoid(joint, qs, axis=0), ms)), 1.0, places=2)
                # ppd returns the marginals of that same joint
                np.testing.assert_allclose(ppds["mass_1"][i], trapezoid(joint, qs, axis=0), rtol=2e-2, atol=1e-4)
                # skipping q = qmin, where the exact marginal is 0 but the grid reference keeps one cell at m1 = mmax
                np.testing.assert_allclose(ppds["mass_ratio"][i, 1:], trapezoid(joint, ms, axis=1)[1:], rtol=2e-2, atol=1e-3)
                self.assertAlmostEqual(float(trapezoid(ppds["mass_1"][i], ms)), 1.0, places=3)
                self.assertAlmostEqual(float(trapezoid(ppds["mass_ratio"][i], qs)), 1.0, places=3)

    def test_qmin_is_required_without_m2min(self):
        x = jnp.linspace(0.1, 1.0, 10)
        with self.assertRaises(ValueError):
            BSplinePrimaryBSplineRatio(self.n_m, self.n_q, x, x, x, x, m2min=None)


class TestPLPeakPrimaryBSplineRatioM2min(unittest.TestCase):
    """m2min truncates the joint to q * m1 >= m2min and renormalizes it, using samples on a grid."""

    def test_m2min_truncates_and_normalizes_the_joint(self):
        m2min, mmin, mmax, n_q = 5.0, 4.0, 100.0, 6
        ms = jnp.linspace(mmin, mmax, 800)
        qs = jnp.linspace(m2min / mmax, 1.0, 800)
        M, Q = jnp.meshgrid(ms, qs)
        coefs = jnp.asarray(np.random.default_rng(1).normal(size=n_q))
        hyper = dict(alpha=-2.5, mmax=mmax, peak_mean=30.0, peak_sd=4.0, peak_frac=0.1, coefs=coefs)

        untruncated = PLPeakPrimaryBSplineRatio(n_q, Q, Q, qmin=m2min / mmax)
        model = PLPeakPrimaryBSplineRatio(n_q, Q, Q, m2min=m2min, qmin=m2min / mmax)
        # mmin is a hyperparameter, so the normalization must trace under jit
        joint = jax.jit(lambda mmin: model(M, mmin=mmin, **hyper))(mmin)

        self.assertTrue(np.all(np.asarray(joint)[np.asarray(Q * M) < m2min] == 0.0))
        self.assertAlmostEqual(float(trapezoid(trapezoid(joint, qs, axis=0), ms)), 1.0, places=2)
        allowed = np.asarray(Q * M) >= m2min
        ratio = np.asarray(joint)[allowed] / np.asarray(untruncated(M, mmin=mmin, **hyper))[allowed]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)  # the same density, rescaled by 1 / Z
