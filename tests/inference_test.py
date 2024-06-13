import glob
import os
import unittest

import deepdish as dd
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.scipy.integrate import trapezoid
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.models.gwpopulation.gwpopulation import PowerlawRedshiftModel
from gwinferno.models.gwpopulation.gwpopulation import powerlaw_primary_ratio_pdf
from gwinferno.pipeline.analysis import construct_hierarchical_model
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.parser import ConfigReader
from gwinferno.pipeline.parser import load_model_from_python_file
from gwinferno.preprocess.data_collection import load_posterior_data, load_injections


def norm_mass_model(alpha, beta, mmin, mmax):
    ms = jnp.linspace(3, 100, 500)
    qs = jnp.linspace(0.01, 1, 300)
    mm, qq = jnp.meshgrid(ms, qs)
    p_mq = powerlaw_primary_ratio_pdf(mm, qq, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
    return trapezoid(trapezoid(p_mq, qs, axis=0), ms)


class TestTruncatedModelInference(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("~/unit_tests/data"):
            pref = "~/unit_tests/data"
        else:
            pref = "tests/data"
        self.data_dir = pref
        self.inj_file = f"{pref}/injections.h5"
        self.param_names = ["mass_1", "mass_ratio", "redshift", "prior"]
        self.param_map = {p: i for i, p in enumerate(self.param_names)}
        self.pedict, self.Nobs, self.Nsamples = self.load_data()
        self.injdict, self.total_inj, self.obs_time = self.load_injections(through_o4a=False, through_o3=True)
        self.z_model = self.setup_redshift_model()
        self.truncated_numpyro_model = self.setup_numpyro_model()

    def tearDown(self) -> None:
        del self.data_dir
        del self.inj_file
        del self.pedict
        del self.injdict
        del self.total_inj
        del self.obs_time
        del self.Nobs
        del self.Nsamples
        del self.param_map
        del self.param_names
        del self.z_model
        del self.truncated_numpyro_model

    def load_data(self, max_samps=100):
        pe_samples = dd.io.load(
            f"{self.data_dir}/GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5"
        )  # load_posterior_samples(self.data_dir, spin=False, max_samples=250)
        names = [k for k in pe_samples.keys()]
        pedata = jnp.array([[pe_samples[e][p] for e in names] for p in self.param_names])
        Nobs = pedata.shape[1]
        Nsamples = pedata.shape[-1]
        idxs = np.random.choice(Nsamples, size=max_samps, replace=False)
        pedict = {k: pedata[self.param_map[k]][:, idxs] for k in self.param_names}
        return pedict, Nobs, max_samps

    def test_load_pe_samples(self):
        fns = glob.glob(f"{self.data_dir}/S*.h5")
        evs = [s.split("/")[-1].replace(".h5", "") for s in fns]
        run_map = {e: "C01:Mixed" for e in evs}
        pe_samples, names = load_posterior_data(self.data_dir, run_map=run_map, spin=False)
        pedata = jnp.array([[pe_samples[e][p].values for e in names] for p in self.param_names])
        Nobs = pedata.shape[1]
        Nsamples = pedata.shape[-1]
        pedict = {k: pedata[self.param_map[k]] for k in self.param_names}
        for param in pedict.keys():
            self.assertEqual(pedict[param].shape, (Nobs, Nsamples))

    def test_pe_shape(self):
        for param in self.pedict.keys():
            self.assertEqual(self.pedict[param].shape, (self.Nobs, self.Nsamples))

    def load_injections(self, **kwargs):
        injections = load_injections(self.inj_file, spin=False, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"])
        injdata = jnp.array([injections[k] for k in self.param_names])
        total_inj = injections["total_generated"]
        obs_time = injections["analysis_time"]
        injdict = {k: injdata[self.param_map[k]] for k in self.param_names}
        return injdict, float(total_inj), obs_time

    def test_injection_shape(self):
        self.assertGreater(self.total_inj, len(self.injdict[self.param_names[0]]))

    def setup_redshift_model(self):
        return PowerlawRedshiftModel(z_pe=self.pedict["redshift"], z_inj=self.injdict["redshift"])

    def setup_numpyro_model(self):
        def model(pedict, injdict, z_model, Nobs, total_inj, obs_time, sample_prior=False):
            alpha = numpyro.sample("alpha", dist.Normal(0, 2))
            beta = numpyro.sample("beta", dist.Normal(0, 2))
            lamb = numpyro.sample("lamb", dist.Normal(0, 2))
            if not sample_prior:
                mmin = 5.0
                mmax = 85.0

                def get_weights(m1, q, z, prior):
                    p_m1q = powerlaw_primary_ratio_pdf(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
                    p_z = z_model(z, lamb)
                    wts = p_m1q * p_z / prior
                    return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

                peweights = get_weights(pedict["mass_1"], pedict["mass_ratio"], pedict["redshift"], pedict["prior"])
                injweights = get_weights(injdict["mass_1"], injdict["mass_ratio"], injdict["redshift"], injdict["prior"])
                hierarchical_likelihood(
                    peweights,
                    injweights,
                    total_inj=total_inj,
                    Nobs=Nobs,
                    Tobs=obs_time,
                    surv_hypervolume_fct=z_model.normalization,
                    vtfct_kwargs=dict(lamb=lamb),
                    marginalize_selection=False,
                    min_neff_cut=False,
                    posterior_predictive_check=True,
                    pedata=pedict,
                    injdata=injdict,
                    param_names=[
                        "mass_1",
                        "mass_ratio",
                        "redshift",
                    ],
                    m1min=mmin,
                    m2min=mmin,
                    mmax=mmax,
                )
            else:
                mmin = numpyro.sample("mmin", dist.Uniform(3, 10))
                mmax = numpyro.sample("mmax", dist.Uniform(50, 100))

        return model

    def test_truncated_prior_sample(self):
        RNG = random.PRNGKey(3)
        kernel = NUTS(self.truncated_numpyro_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, self.pedict, self.injdict, self.z_model, self.Nobs, self.total_inj, self.obs_time, sample_prior=True)
        samples = mcmc.get_samples()
        self.assertEqual(samples["alpha"].shape, (5,))
        self.assertEqual(samples["beta"].shape, (5,))
        self.assertEqual(samples["lamb"].shape, (5,))

    def test_truncated_posterior_sample(self):
        RNG = random.PRNGKey(4)
        kernel = NUTS(self.truncated_numpyro_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, self.pedict, self.injdict, self.z_model, self.Nobs, self.total_inj, self.obs_time, sample_prior=False)
        samples = mcmc.get_samples()
        self.assertEqual(samples["alpha"].shape, (5,))
        self.assertEqual(samples["beta"].shape, (5,))
        self.assertEqual(samples["lamb"].shape, (5,))

    def test_config_reader(self):
        config_reader = ConfigReader()
        config_reader.parse("gwinferno/pipeline/config_files/example_config.yml")
        model_dict, prior_dict = config_reader.models, config_reader.priors
        data_conf, sampler_conf, likelihood_kwargs = config_reader.data_conf, config_reader.sampler_conf, config_reader.likelihood_kwargs
        sampling_params, label, outdir = config_reader.sampling_params, config_reader.label, config_reader.outdir
        model = construct_hierarchical_model(model_dict, prior_dict, **likelihood_kwargs)
        del data_conf, sampler_conf, sampling_params, label, outdir, model, likelihood_kwargs, prior_dict

    def test_config_py_reader(self):
        config_reader = ConfigReader()
        config_reader.parse("gwinferno/pipeline/config_files/example_config_python_model.yml")
        model_dict, prior_dict = config_reader.models, config_reader.priors
        data_conf, sampler_conf, likelihood_kwargs = config_reader.data_conf, config_reader.sampler_conf, config_reader.likelihood_kwargs
        sampling_params, label, outdir = config_reader.sampling_params, config_reader.label, config_reader.outdir
        model = load_model_from_python_file(model_dict.pop("file_path"))
        self.assertFalse(prior_dict)
        self.assertFalse(model_dict)
        del data_conf, sampler_conf, sampling_params, label, outdir, model, likelihood_kwargs, prior_dict
