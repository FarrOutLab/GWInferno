import os
import unittest

import jax.numpy as jnp
import xarray as xr

from gwinferno.models.bsplines.single import Base1DBSplineModel
from gwinferno.preprocess.data_collection import load_injections


class TestBase1DBsplineModel(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("~/unit_tests/data"):
            pref = "~/unit_tests/data"
        else:
            pref = "tests/data"
            self.data_dir = pref
        self.inj_file = f"{pref}/injections.h5"
        self.pedict = self.load_data()
        self.injdict = self.load_injections(through_o4a=False, through_o3=True)
        self.nsplines = 10
        self.coefs = jnp.ones((self.nsplines,))

    def tearDown(self) -> None:
        del self.data_dir
        del self.inj_file
        del self.pedict
        del self.injdict
        del self.nsplines

    def load_injections(self, **kwargs):
        p_names = ["mass_1"]
        injections = load_injections(self.inj_file, p_names, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"])
        injdata = jnp.asarray(injections.data)
        injdict = {k: injdata[i] for i, k in enumerate(injections.param.values)}
        return injdict

    def load_data(self):
        loaded_dataset = xr.load_dataset(f"{self.data_dir}/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        pedata = jnp.asarray(dataarray.data)
        pedict = {k: pedata[:, i, :] for i, k in enumerate(dataarray.param.values)}
        return pedict

    def test_spline_shape(self):
        pe_x = self.pedict["mass_1"]
        inj_x = self.injdict["mass_1"]
        bspline = Base1DBSplineModel(self.nsplines, pe_x, inj_x)

        inj_pdf = bspline.funcs[0](self.coefs)
        pe_pdf = bspline.funcs[1](self.coefs)
        self.assertEqual(pe_x.shape, pe_pdf.shape, msg="PE sample shape different than PE PDF shape")
        self.assertEqual(inj_x.shape, inj_pdf.shape, msg="Inj sample shape different than Inj PDF shape")
