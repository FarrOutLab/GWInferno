import os
import unittest

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from jax import random
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.pipeline.utils import bspline_mass_prior
from gwinferno.pipeline.utils import bspline_redshift_prior
from gwinferno.pipeline.utils import bspline_spin_prior
from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.pipeline.utils import posterior_dict_to_xarray
from gwinferno.pipeline.utils import setup_bspline_mass_models
from gwinferno.pipeline.utils import setup_bspline_spin_models
from gwinferno.pipeline.utils import setup_powerlaw_spline_redshift_model
from gwinferno.postprocess.calculations import calculate_bspline_mass_ppds
from gwinferno.postprocess.calculations import calculate_bspline_spin_ppds
from gwinferno.postprocess.calculations import calculate_powerlaw_spline_rate_of_z_ppds
from gwinferno.postprocess.plot import plot_mass_pdfs
from gwinferno.postprocess.plot import plot_rate_of_z_pdfs
from gwinferno.postprocess.plot import plot_spin_pdfs
from gwinferno.postprocess.postprocess import PopSummaryWriteOut
from gwinferno.preprocess.data_collection import load_injections


class TestModelUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.inj_file = "tests/data/injections.h5"
        self.param_names = ["mass_1", "mass_ratio", "redshift", "prior", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
        self.pedict, self.Nobs, self.Nsamples, self.events = self.load_data()
        self.injdict, self.total_inj, self.obs_time = self.load_injs(through_o4a=False, through_o3=True)
        self.nspline_dict = {"m1": 10, "q": 10, "a1": 10, "tilt1": 10, "a2": 10, "tilt2": 10}
        self.n_samples = 5

    def load_data(self, max_samps=100):
        loaded_dataset = xr.load_dataset("tests/data/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        events = dataarray.indexes["variable"].tolist()
        pedata = jnp.asarray(dataarray.data)
        Nobs = pedata.shape[0]
        Nsamples = pedata.shape[-1]
        idxs = np.random.choice(Nsamples, size=max_samps, replace=False)
        pedict = {k: pedata[:, i, idxs] for i, k in enumerate(dataarray.param.values)}
        return pedict, Nobs, max_samps, events

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
        setup_bspline_spin_models(self.pedict, self.injdict, 10, 10, IID=False, a2_nsplines=10, ct2_nsplines=10)
        setup_bspline_spin_models(self.pedict, self.injdict, 10, 10, IID=True)

    def test_prior_model(self):
        zmodel = setup_powerlaw_spline_redshift_model(self.pedict, self.injdict, 10)

        def model():
            bspline_mass_prior(m_nsplines=10, q_nsplines=10, name="test_mass")
            bspline_spin_prior(a_nsplines=10, ct_nsplines=10, a_tau=25, ct_tau=25, IID=True, name="test_iid")
            bspline_spin_prior(a_nsplines=10, ct_nsplines=10, a_tau=25, ct_tau=25, IID=False, name="test_ind")
            bspline_redshift_prior(z_nsplines=10, z_tau=1, name="test_z")
            numpyro.sample("lamb", dist.Normal(0, 3))

        RNG = random.PRNGKey(3)
        kernel = NUTS(model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=self.n_samples)
        mcmc.run(RNG)
        samples = mcmc.get_samples()

        m_pdfs, ms, q_pdfs, qs = calculate_bspline_mass_ppds(samples["mass_cs_test_mass"], samples["q_cs_test_mass"], self.nspline_dict, 5.0, 100.0)
        apdfs, aa, ctpdfs, cc = calculate_bspline_spin_ppds(samples["a_cs_test_iid"], samples["tilt_cs_test_iid"], self.nspline_dict)
        apdfs_1, apdfs_2, aa, ctpdfs_1, ctpdfs_2, cc = calculate_bspline_spin_ppds(
            samples["a1_cs_test_ind"],
            samples["tilt1_cs_test_ind"],
            self.nspline_dict,
            a2_cs=samples["a2_cs_test_ind"],
            tilt2_cs=samples["tilt2_cs_test_ind"],
        )

        rs, zs = calculate_powerlaw_spline_rate_of_z_ppds(samples["lamb"], samples["z_cs_test_z"], jnp.ones_like(samples["lamb"]), zmodel)

        plot_mass_pdfs([m_pdfs], [q_pdfs], ms, qs, ["test"], "test", "tests", save=False)
        plot_spin_pdfs([apdfs], [ctpdfs], aa, cc, ["test"], "test", "tests", save=False)
        plot_rate_of_z_pdfs(rs, zs, "test", "tests", save=False)

        post_xr = posterior_dict_to_xarray(samples)
        post_xr.to_netcdf("tests/data/posteriors.h5")

        pdf_dict1 = {"m1": m_pdfs, "q": q_pdfs, "a1": apdfs, "tilt1": ctpdfs, "z": rs}
        param_dict1 = {"m1": ms, "q": qs, "a1": aa, "tilt1": cc, "z": zs}

        pdf_xr = pdf_dict_to_xarray(pdf_dict1, param_dict1, self.n_samples)
        pdf_xr.to_netcdf("tests/data/pdfs.h5")

        pdf_dict2 = {"m1": m_pdfs, "q": q_pdfs, "a1": apdfs_1, "tilt1": ctpdfs_1, "a2": apdfs_2, "tilt2": ctpdfs_2, "z": rs}
        param_dict2 = {"m1": ms, "q": qs, "a1": aa, "tilt1": cc, "a2": aa, "tilt2": cc, "z": zs}

        pdf_dict_to_xarray(pdf_dict2, param_dict2, self.n_samples)

        popfile_path = "tests/data/popsummary.h5"

        empty_ev = np.zeros_like(self.events).tolist()
        hyperparams = ["mass_cs_test_mass", "q_cs_test_mass", "a_cs_test_iid", "tilt_cs_test_iid", "z_cs_test_z"]
        empty_hyp = np.zeros_like(hyperparams).tolist()
        params = ["m1", "a1", "tilt1", "q", "z"]

        if os.path.exists(popfile_path):
            os.remove(popfile_path)

        popfile = PopSummaryWriteOut(
            popfile_path,
            events=self.events,
            event_waveforms=empty_ev,
            event_sample_IDs=empty_ev,
            hyperparameter_names=hyperparams,
            hyperparameter_latex_labels=empty_hyp,
            event_parameters=params,
            hyperparameter_descriptions=empty_hyp,
        )

        popfile.save_hypersamples("tests/data/posteriors.h5")
        popfile.save_rates_on_grids("tests/data/pdfs.h5", rate_names=["m1_pdfs", "a1_pdfs", "tilt1_pdfs", "q_pdfs", "z_pdfs"], grid_params=params)

        os.remove("tests/data/posteriors.h5")
        os.remove("tests/data/pdfs.h5")
        os.remove(popfile_path)

    # def test_pdf_dict_to_xarray(self):

    #     n_samples = 100
    #     one = np.ones(200)
    #     two = np.ones(300)
    #     pdf_1 = np.ones((n_samples, one.shape[0]))
    #     pdf_2 = np.ones((n_samples, two.shape[0]))

    #     pdf_dict1 = {"one": pdf_1, "two": pdf_2}
    #     param_dict = {"one": one, "two": two}

    #     pdf_dict_to_xarray(pdf_dict1, param_dict, n_samples)

    #     pdf_1b = np.ones((n_samples, one.shape[0]))
    #     pdf_2b = np.ones((n_samples, two.shape[0]))

    #     pdf_dict2 = {"one": [pdf_1, pdf_1b], "two": [pdf_2, pdf_2b]}

    #     pdf_dict_to_xarray(pdf_dict2, param_dict, n_samples, subpop_names=["A", "B"])
