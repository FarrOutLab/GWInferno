from importlib import import_module

import jax.numpy as jnp
import numpyro
import yaml
from jax import random
import numpy as np
from astropy.cosmology import Planck15
import arviz as az
import matplotlib.pyplot as plt
import deepdish as dd
import sys

from .analysis import hierarchical_likelihood_in_log
from ..preprocess.data_collection import load_catalog_from_metadata
from ..preprocess.selection import load_injections


NP_KERNEL_MAP = {"NUTS": numpyro.infer.NUTS, "HMC": numpyro.infer.HMC}


class PopModel(object):
    def __init__(self, model=None, params=[]):
        self.model = model
        self.params = params


class PopPrior(object):
    def __init__(self, dist, params={}):
        self.dist = dist
        self.params = params


def load_dist_from_string(dist):
    split_d = dist.split(".")
    module = ".".join(split_d[:-1])
    function = split_d[-1]
    return getattr(import_module(module), function)


def load_config_from_yaml(yml_file):
    with open(yml_file, "r") as f:
        yml = yaml.safe_load(f)
    model_dict = {}
    prior_dict = {}
    sampling_params = []
    for param in yml.keys():
        if param in ["data_args", "sampler_args", "label"]:
            continue
        for hp in yml[param]["hyper_params"]:
            if "prior" in yml[param]["hyper_params"][hp] and "prior_params" in yml[param]["hyper_params"][hp]:
                prior_dict[f"{param}_{hp}"] = PopPrior(
                    load_dist_from_string(yml[param]["hyper_params"][hp]["prior"]), yml[param]["hyper_params"][hp]["prior_params"]
                )
                sampling_params.append(f"{param}_{hp}")
            elif "value" in yml[param]["hyper_params"][hp]:
                prior_dict[f"{param}_{hp}"] = yml[param]["hyper_params"][hp]["value"]
        model_dict[param] = PopModel(load_dist_from_string(yml[param]["model"]), [p for p in yml[param]["hyper_params"]])
    return model_dict, prior_dict, yml["data_args"], yml["sampler_args"], sampling_params, yml['label']


def construct_hierarchical_model(model_dict, prior_dict):
    source_param_names = [k for k in model_dict.keys()]
    hyper_params = {k: None for k in prior_dict.keys()}
    pop_models = {k: None for k in model_dict.keys()}

    def model(samps, injs, Ninj, Nobs, Tobs, z_grid, dVcdz_grid):
        for k, v in prior_dict.items():
            try:
                hyper_params[k] = numpyro.sample(k, v.dist(**v.params))
            except AttributeError:
                hyper_params[k] = v

        for k, v in model_dict.items():
            hps = {p: hyper_params[f'{k}_{p}'] for p in v.params}
            if k == 'redshift':
                pop_models[k] = v.model(**hps, zgrid=z_grid, dVcdz=dVcdz_grid)
            else:
                pop_models[k] = v.model(**hps)

        inj_weights = jnp.sum(jnp.array([pop_models[k].log_prob(injs[k]) for k in source_param_names]), axis=0) - jnp.log(injs["prior"])
        pe_weights = jnp.sum(jnp.array([pop_models[k].log_prob(samps[k]) for k in source_param_names]), axis=0) - jnp.log(samps["prior"])

        def shvf(lamb):
            return pop_models["redshift"].norm

        hierarchical_likelihood_in_log(
            pe_weights,
            inj_weights,
            total_inj=Ninj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=shvf,
            vtfct_kwargs={"lamb": hyper_params["redshift_lamb"]},
            marginalize_selection=False,
            min_neff_cut=True,
            posterior_predictive_check=True,
            pedata=samps,
            injdata=injs,
            param_names=source_param_names,
            m1min=hyper_params["mass_1_minimum"],
            m2min=2.0,
            mmax=hyper_params["mass_1_maximum"],
        )

    return model


def setup(data_conf, params):
    catalog_info = data_conf["catalog_summary_json"]
    params.append("prior")
    spin = "a_1" in params or "chi_eff" in params
    (pe_dict, names), injfile, far_thresh, metadata_dir = load_catalog_from_metadata(catalog_info, spin=spin, downsample=True, max_samples=100000)
    injections = load_injections(injfile, 1.0 / far_thresh, spin=spin, semianalytic=data_conf['seminanalytic_injs'])
    Ninj = injections.pop("total_generated")
    Nobs = len(names)
    Tobs = injections.pop("analysis_time")
    nPost = len(pe_dict[names[0]]['mass_1'])
    new_pe_dict = {p: np.zeros((Nobs, nPost)) for p in params}
    for i,ev in enumerate(names):
        for p in params:
            new_pe_dict[p][i,:] = pe_dict[ev][p].values
    pe_dict = {p: jnp.array(new_pe_dict[p]) for p in params}
    inj_dict = {p: jnp.array(injections[p]) for p in params}
    return pe_dict, inj_dict, Ninj, Nobs, Tobs


def run_inference(config_yml, PRNG_seed=0):
    model_dict, prior_dict, data_conf, sampler_conf, sampling_params, label = load_config_from_yaml(config_yml)
    model = construct_hierarchical_model(model_dict, prior_dict)

    pe_dict, inj_dict, Ninj, Nobs, Tobs = setup(data_conf, [k for k in model_dict.keys()])
    z_max = max([jnp.max(pe_dict['redshift']), jnp.max(inj_dict['redshift'])])

    kernel = NP_KERNEL_MAP[sampler_conf["kernel"]](model, **sampler_conf["kernel_kwargs"])
    mcmc = numpyro.infer.MCMC(kernel, **sampler_conf["mcmc_kwargs"])

    z_grid = jnp.linspace(1e-9, z_max, 1500)
    dVcdz_grid = jnp.array(Planck15.differential_comoving_volume(np.array(z_grid)).value * 4.0 * np.pi)

    rng_key = random.PRNGKey(PRNG_seed)
    rng_key, rng_key_ = random.split(rng_key)
    mcmc.run(rng_key_, pe_dict, inj_dict, Ninj=Ninj, Nobs=Nobs, Tobs=Tobs, z_grid=z_grid, dVcdz_grid=dVcdz_grid)
    mcmc.print_summary()
    return mcmc, sampling_params, label


def plot_trace_dump_samples(mcmc, var_names, label):
    idata = az.from_numpyro(mcmc)
    fig = az.plot_trace(idata, var_names=var_names)
    plt.savefig(f"{label}_trace.png")

    samples = mcmc.get_samples()
    dd.io.save(f"{label}_posterior_samples.h5", samples)

if __name__ == "__main__":
    mcmc, var_names, label = run_inference(sys.argv[1])

    var_names.append("rate")
    var_names.append("log_nEffs")
    var_names.append("log_nEff_inj")
    var_names.append("log_l")

    plot_trace_dump_samples(mcmc, var_names, label)