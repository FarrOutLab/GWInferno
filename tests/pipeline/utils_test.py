import unittest
import numpy as np

from jax import random
from numpyro.infer import MCMC
from numpyro.infer import NUTS
import jax.numpy as jnp
import xarray as xr

from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import setup_bspline_mass_models
from gwinferno.pipeline.utils import setup_bspline_spin_models
from gwinferno.pipeline.utils import setup_powerlaw_spline_redshift_model
from gwinferno.pipeline.utils import bspline_mass_prior
from gwinferno.pipeline.utils import bspline_spin_prior
from gwinferno.pipeline.utils import bspline_redshift_prior
from gwinferno.pipeline.utils import posterior_dict_to_xarray
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.preprocess.data_collection import load_injections

class TestTruncatedModelInference(unittest.TestCase):
    def setUp(self) -> None:
        self.inj_file = f"tests/data/injections.h5"
        self.param_names = ["mass_1", "mass_ratio", "redshift", "prior", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        self.pedict, self.Nobs, self.Nsamples = self.load_data()
        self.injdict, self.total_inj, self.obs_time = self.load_injs(through_o4a=False, through_o3=True)

    def load_data(self, max_samps=100):
        loaded_dataset = xr.load_dataset(f"tests/data/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        pedata = jnp.asarray(dataarray.data)
        Nobs = pedata.shape[0]
        Nsamples = pedata.shape[-1]
        idxs = np.random.choice(Nsamples, size=max_samps, replace=False)
        pedict = {k: pedata[:, i, idxs] for i, k in enumerate(dataarray.param.values)}
        return pedict, Nobs, max_samps

    def load_injs(self, **kwargs):
        p_names = self.param_names.copy()
        p_names.remove("prior")
        injections = load_injections(self.inj_file, p_names, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"])
        injdata = jnp.asarray(injections.data)
        total_inj = injections.attrs["total_generated"]
        obs_time = injections.attrs["analysis_time"]
        injdict = {k: injdata[i] for i, k in enumerate(injections.param.values)}
        return injdict, float(total_inj), obs_time
    
    def test_load_base_parser(self):
        load_base_parser()

    def test_setup_bspline_mass_models(self):
        setup_bspline_mass_models(self.pedict, self.injdict, 10, 10, 5.0, 100.0)

    
    def test_setup_bspline_spin_models(self):
        setup_bspline_spin_models(self.pedict, self.injdict, 10, 10, IID = False, a2_nsplines = 10, ct2_nsplines = 10)
        setup_bspline_spin_models(self.pedict, self.injdict, 10, 10, IID = True)

    def test_setup_powerlaw_spline_redshift_model(self):
        setup_powerlaw_spline_redshift_model(self.pedict, self.injdict, 10)
    
    def setup_prior_model(self):

        def model():
            bspline_mass_prior(m_nsplines=10, q_nsplines=10, name = 'test_mass')
            bspline_spin_prior(a_nsplines=10, ct_nsplines=10, a_tau=25, ct_tau=25, IID = True, name = 'test_iid')
            bspline_spin_prior(a_nsplines=10, ct_nsplines=10, a_tau=25, ct_tau=25, IID = False, name = 'test_ind')
            bspline_redshift_prior(z_nsplines=10, z_tau=1, name = 'test_z')

        return model
    
    def test_prior_model(self):
        RNG = random.PRNGKey(3)
        kernel = NUTS(self.setup_prior_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG)
        samples = mcmc.get_samples()

        posterior_dict_to_xarray(samples)
    
    def test_pdf_dict_to_xarray(self):

        n_samples = 100
        one = np.ones(200)
        two = np.ones(300)
        pdf_1 = np.ones((n_samples,one.shape[0]))
        pdf_2 = np.ones((n_samples,two.shape[0]))

        pdf_dict1 = {"one": pdf_1, "two": pdf_2}
        param_dict = {"one": one, "two": two}

        pdf_dict_to_xarray(pdf_dict1, param_dict, n_samples)

        
        pdf_1b = np.ones((n_samples,one.shape[0]))
        pdf_2b = np.ones((n_samples,two.shape[0]))

        pdf_dict2 = {"one": [pdf_1, pdf_1b], "two": [pdf_2, pdf_2b]}

        pdf_dict_to_xarray(pdf_dict2, param_dict, n_samples, subpop_names=['A', 'B'])



