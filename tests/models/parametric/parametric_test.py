import os
import unittest

import jax.numpy as jnp
import xarray as xr
from jax.scipy.integrate import trapezoid

from gwinferno.models.parametric.parametric import PowerlawRedshiftModel
from gwinferno.preprocess.data_collection import load_injection_dataset


class TestPowerlawRedshift(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("~/unit_tests/data"):
            pref = "~/unit_tests/data"
        else:
            pref = "tests/data"
            self.data_dir = pref
        self.inj_file = f"{pref}/injections.h5"
        self.pedict = self.load_data()
        self.injdict = self.load_injection_dataset(through_o4a=False, through_o3=True)
        self.pe_z = self.pedict["redshift"]
        self.inj_z = self.injdict["redshift"]
        self.lamb = 2.7
        self.z_powerlaw = PowerlawRedshiftModel(self.pe_z, self.inj_z)
        self.zmax = self.z_powerlaw.zmax

    def load_injection_dataset(self, **kwargs):
        p_names = ["mass_1", "mass_ratio", "redshift", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        injections = load_injection_dataset(self.inj_file, p_names, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"])
        injdata = jnp.asarray(injections.data)
        injdict = {k: injdata[i] for i, k in enumerate(injections.param.values)}
        return injdict

    def load_data(self):
        loaded_dataset = xr.load_dataset(f"{self.data_dir}/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        pedata = jnp.asarray(dataarray.data)
        pedict = {k: pedata[:, i, :] for i, k in enumerate(dataarray.param.values)}
        return pedict

    def test_norm(self):
        zs = self.z_powerlaw.zs
        integral = trapezoid(self.z_powerlaw.prob(zs, self.z_powerlaw.dVdz_, self.lamb) / self.z_powerlaw.normalization(self.lamb), zs)
        self.assertAlmostEqual(integral, 1.0, places=3, msg="Redshift Powerlaw not noramlized")

    def test_shape(self):
        pe_pdf = self.z_powerlaw(self.pe_z, self.lamb)
        inj_pdf = self.z_powerlaw(self.inj_z, self.lamb)

        pe_log_pdf = self.z_powerlaw.log_prob(self.pe_z, self.lamb)
        inj_log_pdf = self.z_powerlaw.log_prob(self.inj_z, self.lamb)

        self.assertEqual(self.pe_z.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(self.inj_z.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")

        self.assertEqual(self.pe_z.shape, pe_log_pdf.shape, msg="PE sample shape different than PE Log PDF shape")
        self.assertEqual(self.inj_z.shape, inj_log_pdf.shape, msg="Inj sample shape different than Inj Log PDF shape")

    def test_bounds(self):

        pdf_pe = self.z_powerlaw(self.pe_z, self.lamb)
        pdf_inj = self.z_powerlaw(self.inj_z, self.lamb)

        z_nonzero_pe = jnp.sum(jnp.where(self.pe_z > self.zmax, pdf_pe, 0))
        z_nonzero_inj = jnp.sum(jnp.where(self.inj_z > self.zmax, pdf_inj, 0))

        self.assertEqual(z_nonzero_pe, 0, msg="PE PDF not properly truncated")
        self.assertEqual(z_nonzero_inj, 0, msg="Inj PDF not properly truncated")
