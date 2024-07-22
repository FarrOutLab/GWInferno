import os

import numpyro
from jax import random
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.pipeline.utils import setup_bspline_mass_models
from gwinferno.pipeline.utils import setup_bspline_spin_models
from gwinferno.pipeline.utils import setup_powerlaw_spline_redshift_model


def setup_result_dir(parsargs):
    """construct a directory to save results to

    Args:
        parsargs (): args from argument parser

    Returns:
        label (str): label for file names
        full_dir (str): result directory
    """
    label = parsargs.run_label + f"_{parsargs.warmup}w_{parsargs.samples}s_rng{parsargs.rngkey}"
    result_directory = parsargs.result_dir + "/" + parsargs.run_label
    full_dir = f"{result_directory}/rngnum-{parsargs.rngkey}"
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    print(f"result files will be saved in directory: {full_dir}")
    return label, full_dir


def run_analysis(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, parsargs, skip_inference=False):
    """run MCMC

    Args:
        numpyro_model (func): numpyro model that defines priors, population model, and likelihood
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injecitons
        constants (dict): dictionary of relevant constants
        param_names (list of strs): list of parameters
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        parsargs (ArgumentParser): args from ArgumentParser.parse_args()
        skip_inference (bool, optional): If True, does not perform inference. Defaults to False.

    Returns:
        if skip_inference == False:
            posterior (dict): dictionary of posterior samples
            z_model (obj): redshift model (needed for later calculations)
        if skip_inference == True:
            z_model
    """
    mass_models = setup_bspline_mass_models(pedict, injdict, nspline_dict["m1"], nspline_dict["q"], mmin=parsargs.mmin, mmax=parsargs.mmax)
    mag_model, tilt_model = setup_bspline_spin_models(
        pedict, injdict, nspline_dict["a1"], nspline_dict["tilt1"], IID=False, a2_nsplines=nspline_dict["a2"], ct2_nsplines=nspline_dict["tilt2"]
    )
    z_model = setup_powerlaw_spline_redshift_model(pedict, injdict, nspline_dict["redshift"])

    if not skip_inference:
        nChains = parsargs.chains
        numpyro.set_host_device_count(nChains)
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=parsargs.warmup, num_samples=parsargs.samples, num_chains=nChains)

        rng_key = random.PRNGKey(parsargs.rngkey)
        rng_key, catkey, rng_key_ = random.split(rng_key, num=3)

        mcmc.run(
            rng_key_,
            pedict,
            injdict,
            constants["nObs"],
            constants["obs_time"],
            constants["total_inj"],
            mass_models,
            mag_model,
            tilt_model,
            z_model,
            parsargs.mmin,
            parsargs.mmax,
            nspline_dict,
            param_names,
        )
        posterior = mcmc.get_samples()

        return posterior, z_model

    else:
        return z_model
