#!/usr/bin/env python
from argparse import ArgumentParser

import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from numpyro.infer import MCMC

from gwinferno.pipeline.analysis import NP_KERNEL_MAP
from gwinferno.pipeline.analysis import construct_hierarchical_model
from gwinferno.pipeline.parser import ConfigReader
from gwinferno.preprocess.data_collection import load_catalog_from_metadata
from gwinferno.preprocess.selection import load_injections

az.style.use("arviz-darkgrid")


def load_args():
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--inspect", action="store_true", default=False)
    return parser.parse_args()


def setup(data_conf, params):
    catalog_info = data_conf["catalog_summary_json"]
    params.append("prior")
    spin = "a_1" in params or "chi_eff" in params
    (pe_dict, names), injfile, far_thresh, _ = load_catalog_from_metadata(catalog_info, spin=spin, downsample=True, max_samples=100000)
    injections = load_injections(injfile, 1.0 / far_thresh, spin=spin, semianalytic=data_conf["seminanalytic_injs"])
    Ninj = injections.pop("total_generated")
    Nobs = len(names)
    Tobs = injections.pop("analysis_time")
    nPost = len(pe_dict[names[0]]["mass_1"])
    new_pe_dict = {p: np.zeros((Nobs, nPost)) for p in params}
    for i, ev in enumerate(names):
        for p in params:
            new_pe_dict[p][i, :] = pe_dict[ev][p].values
    pe_dict = {p: jnp.array(new_pe_dict[p]) for p in params}
    inj_dict = {p: jnp.array(injections[p]) for p in params}
    return pe_dict, inj_dict, Ninj, Nobs, Tobs


def run_inference(config_yml, inspect=False, PRNG_seed=0):
    config_reader = ConfigReader()
    config_reader.parse(config_yml)
    model_dict, prior_dict = config_reader.models, config_reader.priors
    data_conf, sampler_conf, likelihood_kwargs = config_reader.data_args, config_reader.sampler_args, config_reader.likelihood_kwargs
    sampling_params, label, outdir = config_reader.sampling_params, config_reader.label, config_reader.outdir
    if inspect:
        print("MODEL DICT: \n", model_dict)
        print("PRIOR DICT: \n", prior_dict)
    if "file_path" in model_dict:
        model = ""  # oad_model_from_python_file(model_dict["file_path"])
    else:
        model = construct_hierarchical_model(model_dict, prior_dict, **likelihood_kwargs)

    pe_dict, inj_dict, Ninj, Nobs, Tobs = setup(data_conf, [k for k in model_dict.keys()])

    kernel = NP_KERNEL_MAP[sampler_conf["kernel"]](model, **sampler_conf["kernel_kwargs"])
    mcmc = MCMC(kernel, **sampler_conf["mcmc_kwargs"])

    rng_key = random.PRNGKey(PRNG_seed)
    rng_key, rng_key_ = random.split(rng_key)
    mcmc.run(rng_key_, pe_dict, inj_dict, Ninj=Ninj, Nobs=Nobs, Tobs=Tobs)
    mcmc.print_summary()
    return mcmc, sampling_params, label, outdir


def plot_trace_dump_samples(mcmc, var_names, label):
    idata = az.from_numpyro(mcmc)
    az.plot_trace(idata, var_names=var_names)
    plt.savefig(f"{label}_trace.png")

    samples = mcmc.get_samples()
    dd.io.save(f"{label}_posterior_samples.h5", samples)


if __name__ == "__main__":
    args = load_args()
    mcmc, var_names, lab, outdir = run_inference(args.config_file, args.inspect)
    label = f"{outdir}/{lab}"
    var_names.append("rate")
    var_names.append("log_nEffs")
    var_names.append("log_nEff_inj")
    var_names.append("log_l")

    plot_trace_dump_samples(mcmc, var_names, label)
