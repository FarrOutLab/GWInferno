import glob
import os
import unittest
from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import xarray as xr
from jax import random
from jax import value_and_grad
from jax.flatten_util import ravel_pytree
from jax.scipy.integrate import trapezoid
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer.util import potential_energy
from numpyro.infer.util import unconstrain_fn

from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.parametric.parametric import PowerlawRedshiftModel
from gwinferno.models.parametric.parametric import powerlaw_primary_ratio_pdf
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel
from gwinferno.pipeline.analysis import construct_hierarchical_model
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.parser import ConfigReader
from gwinferno.pipeline.parser import load_model_from_python_file
from gwinferno.preprocess.data_collection import load_injection_dataset
from gwinferno.preprocess.data_collection import load_posterior_dataset


def norm_mass_model(alpha, beta, mmin, mmax):
    ms = jnp.linspace(3, 100, 500)
    qs = jnp.linspace(0.01, 1, 300)
    mm, qq = jnp.meshgrid(ms, qs)
    p_mq = powerlaw_primary_ratio_pdf(mm, qq, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
    return trapezoid(trapezoid(p_mq, qs, axis=0), ms)


class TestModelInference(unittest.TestCase):
    def setUp(self) -> None:
        self.mmin = 5.0
        self.mmax = 85.0

        if os.path.exists("~/unit_tests/data"):
            pref = "~/unit_tests/data"
        else:
            pref = "tests/data"
        self.data_dir = pref
        self.inj_file = f"{pref}/injections.h5"
        self.param_names = ["mass_1", "mass_ratio", "redshift", "prior", "chi_eff"]
        self.pedict, self.Nobs, self.Nsamples = self.load_data()
        self.injdict, self.total_inj, self.obs_time = self.load_injs(through_o4a=False, through_o3=True)
        self.parametric_model = self.setup_parametric_model()
        self.bspline_model = self.setup_bspline_model()

    def tearDown(self) -> None:
        del self.data_dir
        del self.inj_file
        del self.pedict
        del self.injdict
        del self.total_inj
        del self.obs_time
        del self.Nobs
        del self.Nsamples
        del self.param_names
        del self.z_parametric_model
        del self.mass_bspline_model
        del self.z_bspline_model
        del self.parametric_model
        del self.bspline_model

    def load_data(self, max_samps=100):
        loaded_dataset = xr.load_dataset(f"{self.data_dir}/xarray_GWTC3_BBH_69evs_downsampled_1000samps_nospin.h5")
        dataarray = loaded_dataset.to_array()
        pedata = jnp.asarray(dataarray.data)
        Nobs = pedata.shape[0]
        Nsamples = pedata.shape[-1]
        idxs = np.random.choice(Nsamples, size=max_samps, replace=False)
        pedict = {k: pedata[:, i, idxs] for i, k in enumerate(dataarray.param.values)}
        return pedict, Nobs, max_samps

    def test_load_pe_samples(self):
        fns = glob.glob(f"{self.data_dir}/S*.h5")
        evs = [s.split("/")[-1].replace(".h5", "") for s in fns]
        run_map = {}

        for ev, file in zip(evs, fns):
            run_map[ev] = {"file_path": file, "waveform": "C01:Mixed", "redshift_prior": "euclidean", "catalog": "GWTC-3"}
        p_names = self.param_names.copy()
        p_names.remove("prior")
        pe_catalog = load_posterior_dataset(catalog_metadata=run_map, param_names=p_names).to_array()
        pedata = jnp.asarray(pe_catalog.data[0])
        Nobs = pedata.shape[0]
        Nsamples = pedata.shape[-1]
        self.pedict = {k: pedata[:, i] for i, k in enumerate(pe_catalog.param.values)}
        for param in self.pedict.keys():
            self.assertEqual(self.pedict[param].shape, (Nobs, Nsamples))

    def test_pe_shape(self):
        for param in self.pedict.keys():
            self.assertEqual(self.pedict[param].shape, (self.Nobs, self.Nsamples))

    def load_injs(self, **kwargs):
        p_names = self.param_names.copy()
        p_names.remove("prior")
        injections = load_injection_dataset(self.inj_file, p_names, through_o3=kwargs["through_o3"], through_o4a=kwargs["through_o4a"]).to_array()
        injdata = jnp.asarray(injections.data[0])
        total_inj = injections.attrs["total_generated"]
        obs_time = injections.attrs["analysis_time"]
        injdict = {k: injdata[i] for i, k in enumerate(injections.param.values)}
        return injdict, float(total_inj), obs_time

    def test_injection_shape(self):
        self.assertGreater(self.total_inj, len(self.injdict[self.param_names[0]]))

    def init_cached_parametric_models(self):
        self.z_parametric_model = PowerlawRedshiftModel(
            z_pe=self.pedict["redshift"],
            z_inj=self.injdict["redshift"],
        )

    def init_bspline_models(self):
        self.m1_nbases = 10
        self.q_nbases = 5
        self.z_nbases = 5
        self.mass_bspline_model = BSplinePrimaryBSplineRatio(
            self.m1_nbases,
            self.q_nbases,
            self.pedict["mass_1"],
            self.injdict["mass_1"],
            self.pedict["mass_ratio"],
            self.injdict["mass_ratio"],
            m1min=self.mmin,
            m2min=self.mmin,
            mmax=self.mmax,
        )
        self.z_bspline_model = PowerlawSplineRedshiftModel(
            self.z_nbases,
            self.pedict["redshift"],
            self.injdict["redshift"],
        )

    def setup_parametric_model(self):
        self.init_cached_parametric_models()
        self.parametric_model_args = (
            self.pedict,
            self.injdict,
            self.z_parametric_model,
            self.Nobs,
            self.total_inj,
            self.obs_time,
        )
        self.parametric_test_params = {
            "alpha": jnp.array(3.5),
            "beta": jnp.array(1.1),
            "lamb": jnp.array(2.9),
            "unscaled_rate": jnp.array(30.0),
        }

        def model(pedict, injdict, z_model, Nobs, total_inj, obs_time, sample_prior=False, log_likelihood=False):
            alpha = numpyro.sample("alpha", dist.Normal(0, 2))
            beta = numpyro.sample("beta", dist.Normal(0, 2))
            lamb = numpyro.sample("lamb", dist.Normal(0, 2))
            if not sample_prior:

                def get_weights(m1, q, z, prior):
                    p_m1q = powerlaw_primary_ratio_pdf(m1, q, alpha=alpha, beta=beta, mmin=self.mmin, mmax=self.mmax)
                    p_z = z_model(z, lamb)
                    wts = p_m1q * p_z / prior
                    return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

                peweights = get_weights(pedict["mass_1"], pedict["mass_ratio"], pedict["redshift"], pedict["prior"])
                injweights = get_weights(injdict["mass_1"], injdict["mass_ratio"], injdict["redshift"], injdict["prior"])
                if not log_likelihood:
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
                        m1min=self.mmin,
                        m2min=self.mmin,
                        mmax=self.mmax,
                    )

                else:
                    hierarchical_likelihood(
                        jnp.log(peweights),
                        jnp.log(injweights),
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
                        m1min=self.mmin,
                        m2min=self.mmin,
                        mmax=self.mmax,
                        log=True,
                    )

        return model

    def setup_bspline_model(self):
        self.init_bspline_models()
        self.bspline_test_params = {
            "m1_coefs": random.normal(random.PRNGKey(0), (self.m1_nbases,)),
            "q_coefs": random.normal(random.PRNGKey(1), (self.q_nbases,)),
            "z_coefs": jnp.ones(self.z_nbases),
            "lamb": jnp.array(2.9),
            "unscaled_rate": jnp.array(30.0),
        }
        self.bspline_model_args = (
            self.pedict,
            self.injdict,
            self.mass_bspline_model,
            self.z_bspline_model,
            self.Nobs,
            self.total_inj,
            self.obs_time,
        )

        def model(pedict, injdict, mass_model, z_model, Nobs, total_inj, obs_time, sample_prior=False, log_likelihood=False):
            m1_nbases = mass_model.primary_model.n_splines
            q_nbases = mass_model.ratio_model.n_splines
            z_nbases = z_model.n_splines

            m1_coef = numpyro.sample("m1_coefs", dist.Normal(0, 6), sample_shape=(m1_nbases,))
            q_coef = numpyro.sample("q_coefs", dist.Normal(0, 6), sample_shape=(q_nbases,))
            lamb = numpyro.sample("lamb", dist.Normal(0, 3))
            z_coef = numpyro.sample("z_coefs", dist.Normal(0, 6), sample_shape=(z_nbases,))

            if not sample_prior:

                def get_weights(z, prior, pe_samples=False):
                    p_m1q = mass_model(m1_coef, q_coef, pe_samples=pe_samples)
                    p_z = z_model(z, lamb, z_coef)
                    wts = p_m1q * p_z / prior
                    return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

                peweights = get_weights(pedict["redshift"], pedict["prior"], pe_samples=True)
                injweights = get_weights(injdict["redshift"], injdict["prior"], pe_samples=False)
                if not log_likelihood:
                    hierarchical_likelihood(
                        peweights,
                        injweights,
                        total_inj=total_inj,
                        Nobs=Nobs,
                        Tobs=obs_time,
                        surv_hypervolume_fct=z_model.normalization,
                        vtfct_kwargs=dict(lamb=lamb, cs=z_coef),
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
                        m1min=self.mmin,
                        m2min=self.mmin,
                        mmax=self.mmax,
                    )

                else:
                    hierarchical_likelihood(
                        jnp.log(peweights),
                        jnp.log(injweights),
                        total_inj=total_inj,
                        Nobs=Nobs,
                        Tobs=obs_time,
                        surv_hypervolume_fct=z_model.normalization,
                        vtfct_kwargs=dict(lamb=lamb, cs=z_coef),
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
                        m1min=self.mmin,
                        m2min=self.mmin,
                        mmax=self.mmax,
                        log=True,
                    )

        return model

    def test_parametric_model(self):
        params = unconstrain_fn(
            self.parametric_model,
            self.parametric_model_args,
            {"log_likelihood": False},
            self.parametric_test_params,
        )
        potential_fn = partial(
            potential_energy,
            self.parametric_model,
            self.parametric_model_args,
            {"log_likelihood": False},
        )
        pe, z_grad = value_and_grad(potential_fn)(params)
        z_grad_flat = ravel_pytree(z_grad)[0]
        self.assertTrue(jnp.isfinite(pe), msg="Potential energy is not finite for test case")
        self.assertTrue(jnp.all(jnp.isfinite(z_grad_flat)), msg="Gradient is not finite; don't expect to find valid initial parameters")

    def test_bspline_model(self):
        params = unconstrain_fn(
            self.bspline_model,
            self.bspline_model_args,
            {"log_likelihood": False},
            self.bspline_test_params,
        )
        potential_fn = partial(
            potential_energy,
            self.bspline_model,
            self.bspline_model_args,
            {"log_likelihood": False},
        )
        pe, z_grad = value_and_grad(potential_fn)(params)
        z_grad_flat = ravel_pytree(z_grad)[0]
        self.assertTrue(jnp.isfinite(pe), msg="Potential energy is not finite for test case")
        self.assertTrue(jnp.all(jnp.isfinite(z_grad_flat)), msg="Gradient is not finite; don't expect to find valid initial parameters")

    # def test_bspline_model_in_log(self):
    #     params = unconstrain_fn(
    #         self.bspline_model,
    #         self.bspline_model_args,
    #         {'log_likelihood': True},
    #         self.bspline_test_params,
    #     )
    #     potential_fn = partial(
    #         potential_energy,
    #         self.bspline_model,
    #         self.bspline_model_args,
    #         {'log_likelihood': True},
    #     )
    #     pe, z_grad = value_and_grad(potential_fn)(params)
    #     z_grad_flat = ravel_pytree(z_grad)[0]
    #     self.assertTrue(jnp.isfinite(pe), msg="Potential energy is not finite for test case")
    #     self.assertTrue(jnp.all(jnp.isfinite(z_grad_flat)), msg="Gradient is not finite; don't expect to find valid initial parameters")

    @pytest.mark.skip(reason="slow...")
    def test_parametric_prior_sample(self):
        RNG = random.PRNGKey(3)
        kernel = NUTS(self.parametric_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, *self.parametric_model_args, sample_prior=True)
        samples = mcmc.get_samples()
        self.assertEqual(samples["alpha"].shape, (5,))
        self.assertEqual(samples["beta"].shape, (5,))
        self.assertEqual(samples["lamb"].shape, (5,))

    @pytest.mark.skip(reason="slow...")
    def test_bspline_prior_sample(self):
        RNG = random.PRNGKey(4)
        kernel = NUTS(self.bspline_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, *self.bspline_model_args, sample_prior=True)
        samples = mcmc.get_samples()
        self.assertEqual(samples["m1_coefs"].shape, (5, self.m1_nbases))
        self.assertEqual(samples["q_coefs"].shape, (5, self.q_nbases))
        self.assertEqual(samples["z_coefs"].shape, (5, self.z_nbases))
        self.assertEqual(samples["lamb"].shape, (5,))

    @pytest.mark.skip(reason="slow...")
    def test_parametric_posterior_sample(self):
        RNG = random.PRNGKey(5)
        kernel = NUTS(self.parametric_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, *self.parametric_model_args, sample_prior=False)
        samples = mcmc.get_samples()
        self.assertEqual(samples["alpha"].shape, (5,))
        self.assertEqual(samples["beta"].shape, (5,))
        self.assertEqual(samples["lamb"].shape, (5,))

    @pytest.mark.skip(reason="slow...")
    def test_bspline_posterior_sample(self):
        RNG = random.PRNGKey(6)
        kernel = NUTS(self.bspline_model, max_tree_depth=2, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
        mcmc.run(RNG, *self.bspline_model_args, sample_prior=False, log_likelihood=False)
        samples = mcmc.get_samples()
        self.assertEqual(samples["m1_coefs"].shape, (5, self.m1_nbases))
        self.assertEqual(samples["q_coefs"].shape, (5, self.q_nbases))
        self.assertEqual(samples["z_coefs"].shape, (5, self.z_nbases))
        self.assertEqual(samples["lamb"].shape, (5,))

    # def test_bspline_posterior_sample_in_log(self):
    #     RNG = random.PRNGKey(6)
    #     kernel = NUTS(self.bspline_model, max_tree_depth=2, adapt_mass_matrix=False)
    #     mcmc = MCMC(kernel, num_warmup=5, num_samples=5)
    #     mcmc.run(RNG, *self.bspline_model_args, sample_prior=False, log_likelihood=True)
    #     samples = mcmc.get_samples()
    #     self.assertEqual(samples["m1_coefs"].shape, (5, self.m1_nbases))
    #     self.assertEqual(samples["q_coefs"].shape, (5, self.q_nbases))
    #     self.assertEqual(samples["z_coefs"].shape, (5, self.z_nbases))
    #     self.assertEqual(samples["lamb"].shape, (5,))

    def test_config_reader(self):
        config_reader = ConfigReader()
        config_reader.parse("examples/config_files/config.yml")
        model_dict, prior_dict = config_reader.models, config_reader.priors
        data_conf, sampler_conf, likelihood_kwargs = config_reader.data_conf, config_reader.sampler_conf, config_reader.likelihood_kwargs
        sampling_params, label, outdir = config_reader.sampling_params, config_reader.label, config_reader.outdir
        model = construct_hierarchical_model(model_dict, prior_dict, **likelihood_kwargs)
        del data_conf, sampler_conf, sampling_params, label, outdir, model, likelihood_kwargs, prior_dict

    def test_config_py_reader(self):
        config_reader = ConfigReader()
        config_reader.parse("examples/config_files/config_w_py_model.yml")
        model_dict, prior_dict = config_reader.models, config_reader.priors
        data_conf, sampler_conf, likelihood_kwargs = config_reader.data_conf, config_reader.sampler_conf, config_reader.likelihood_kwargs
        sampling_params, label, outdir = config_reader.sampling_params, config_reader.label, config_reader.outdir
        model = load_model_from_python_file(model_dict.pop("file_path"))
        self.assertFalse(prior_dict)
        self.assertFalse(model_dict)
        del data_conf, sampler_conf, sampling_params, label, outdir, model, likelihood_kwargs, prior_dict
