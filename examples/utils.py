import os

import numpyro
import arviz as az
import matplotlib.pyplot as plt
from jax import random
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.pipeline.utils import setup_bspline_mass_models
from gwinferno.pipeline.utils import setup_bspline_spin_models
from gwinferno.pipeline.utils import setup_bspline_spin_models_2d
from gwinferno.pipeline.utils import setup_bspline_primary_massratio_chieff_models
from gwinferno.pipeline.utils import setup_powerlaw_spline_redshift_model

from gwinferno.models.bsplines.joint import BivariateBSplineMassRatioChiEff
from gwinferno.models.parametric.parametric import PowerlawRedshiftModel


def setup_result_dir(parsargs):
    """construct a directory to save results to

    Args:
        parsargs (): args from argument parser

    Returns:
        label (str): label for file names
        full_dir (str): result directory
    """
    label = parsargs.run_label + f'_{parsargs.warmup}w_{parsargs.samples}s_rng{parsargs.rngkey}'
    result_directory = parsargs.result_dir+ '/' + parsargs.run_label
    full_dir = f'{result_directory}/rngnum-{parsargs.rngkey}/{parsargs.warmup}w_{parsargs.samples}s'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    print(f'result files will be saved in directory: {full_dir}')
    return label, full_dir


def run_bspline_analysis(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, parsargs, skip_inference=False):
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
        pedict, injdict, nspline_dict["a1"], nspline_dict["tilt1"], IID=False, a2_nsplines=nspline_dict["a2"], ct2_nsplines=nspline_dict["tilt2"])
    z_model = setup_powerlaw_spline_redshift_model(pedict, injdict, nspline_dict["redshift"])

    if not skip_inference:
        nChains = parsargs.chains
        numpyro.set_host_device_count(nChains)
        kernel = NUTS(numpyro_model)#, max_tree_depth=11)#, init_strategy=initialization.init_to_median,  step_size=2e-3)
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
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        trace_plots = az.plot_trace(mcmc, compact=True)

        return posterior, z_model, trace_plots

    else:
        return z_model

def run_bspline_analysis_2d(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, parsargs, skip_inference=False):
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
    mag_tilt_model = setup_bspline_spin_models_2d(pedict, injdict, nspline_dict["a1"], nspline_dict["tilt1"], IID=True,
                                                  a2_nsplines=nspline_dict["a2"], ct2_nsplines=nspline_dict["tilt2"])
    z_model = setup_powerlaw_spline_redshift_model(pedict, injdict, nspline_dict["redshift"])

    if not skip_inference:
        nChains = parsargs.chains
        numpyro.set_host_device_count(nChains)
        kernel = NUTS(numpyro_model)#, max_tree_depth=11)#, init_strategy=initialization.init_to_median,  step_size=2e-3)
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
            mag_tilt_model,
            z_model,
            parsargs.mmin,
            parsargs.mmax,
            nspline_dict,
            param_names,
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        return posterior, z_model

    else:
        return z_model
    
def run_massratio_chieff_bspline_analysis_2d(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, hyper_params_dict, parsargs, skip_inference=False):
    """run MCMC
    
    Args:
        numpyro_model (func): numpyro model that defines priors, population model, and likelihood
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injecitons
        constants (dict): dictionary of relevant constants
        param_names (list of strs): list of parameters
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        hyper_params_dict (dict): dictionary containing hyperparameters
        parsargs (ArgumentParser): args from ArgumentParser.parse_args()
        skip_inference (bool, optional): If True, does not perform inference. Defaults to False.

    Returns:
        if skip_inference == False:
            posterior (dict): dictionary of posterior samples
            z_model (obj): redshift model (needed for later calculations)
        if skip_inference == True:
            z_model
    """
    primary_model, chiq_model = setup_bspline_primary_massratio_chieff_models(pedict, injdict, nspline_dict["m1"], nspline_dict["chi_eff"], nspline_dict["q"],
                                                                     mmin=parsargs.mmin, mmax=parsargs.mmax)
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
            primary_model,
            chiq_model,
            z_model,
            parsargs.mmin,
            parsargs.mmax,
            nspline_dict,
            hyper_params_dict,
            param_names,
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        trace_plots = az.plot_trace(mcmc, compact=True)

        return mcmc, posterior, z_model, trace_plots
    
    else:
        return z_model
    
def run_powerlaw_peak_bbspline_chiq_analysis(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, hyper_params_dict, parsargs, skip_inference=False):
    """run MCMC
    
    Args:
        numpyro_model (func): numpyro model that defines priors, population model, and likelihood
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injecitons
        constants (dict): dictionary of relevant constants
        param_names (list of strs): list of parameters
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        hyper_params_dict (dict): dictionary containing hyperparameters
        parsargs (ArgumentParser): args from ArgumentParser.parse_args()
        skip_inference (bool, optional): If True, does not perform inference. Defaults to False.

    Returns:
        if skip_inference == False:
            posterior (dict): dictionary of posterior samples
            z_model (obj): redshift model (needed for later calculations)
        if skip_inference == True:
            z_model
    """

    chiq_model = BivariateBSplineMassRatioChiEff(ndofs=(nspline_dict['chi_eff'], nspline_dict['q']),
                                                 pe_vals=(pedict['chi_eff'], pedict['mass_ratio']),
                                                 inj_vals=(injdict['chi_eff'], injdict['mass_ratio']),
                                                 q_min=parsargs.mmin/parsargs.mmax, orders=hyper_params_dict['chiq_order'])
    z_model = PowerlawRedshiftModel(pedict['redshift'], injdict['redshift'])

    if not skip_inference:
        nChains = parsargs.chains
        numpyro.set_host_device_count(nChains)
        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=parsargs.warmup, num_samples=parsargs.samples, num_chains=nChains)

        rng_key = random.PRNGKey(parsargs.rngkey)
        rng_key, catkey, rng_key_ = random.split(rng_key, num=3)

        with numpyro.validation_enabled():
            mcmc.run(
                rng_key_,
                pedict,
                injdict,
                constants["nObs"],
                constants["obs_time"],
                constants["total_inj"],
                chiq_model,
                z_model,
                parsargs.mmin,
                parsargs.mmax,
                nspline_dict,
                hyper_params_dict,
                param_names,
            )
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        trace_plots = az.plot_trace(mcmc, compact=True)

        return mcmc, posterior, z_model, trace_plots
    
    else:
        return z_model

def run_powerlawpeak_analysis(numpyro_model, pedict, injdict, constants, param_names, parsargs, skip_inference=False):
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

    z_model = PowerlawRedshiftModel(pedict['redshift'], injdict['redshift'])

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
            z_model,
            parsargs.mmin,
            parsargs.mmax,
            param_names,
        )
        posterior = mcmc.get_samples()

        return posterior, z_model

    else:
        return z_model