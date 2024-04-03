import arviz as az
import numpy as np
import xarray as xr
from jax import device_get
from jax import random
from jax.tree_util import tree_map
from numpyro.diagnostics import effective_sample_size
from numpyro.diagnostics import split_gelman_rubin
from numpyro.infer import MCMC

from .types import GWInfernoData


def compute_rhats(data, threshold=1.001, num_chains=1):
    if num_chains == 1:
        data = tree_map(lambda x: x[None, ...], data)
    rhats = []
    for name, value in data.items():
        value = device_get(value)
        rhat = split_gelman_rubin(value)
        if isinstance(rhat, np.ndarray):
            rhats.extend(rhat)
        else:
            rhats.append(rhat)
    sel = np.array(rhats) < threshold
    tot = len(sel)
    keep_sampling = True
    percent_not_converged = (sum(~sel) / tot) * 100
    if (sum(~sel) / tot) * 100 < 0.1:
        keep_sampling = False
        print("it is recommended to stop sampling")
    else:
        print(f"it is recommended to continue sampling as {percent_not_converged:.0f}% of parameters have rhat above {threshold}")

    return keep_sampling


def compute_effective_sample_size(data, threshold=1000, num_chains=1):
    if num_chains == 1:
        data = tree_map(lambda x: x[None, ...], data)
    for name, value in data.items():
        n_eff = effective_sample_size(value)
    keep_sampling = True
    if n_eff > threshold:
        keep_sampling = False
        print("it is recommended to stop sampling")
    else:
        print(f"it is recommended to continue sampling as {n_eff:.0f} is below the threshold value of {threshold}")
    return keep_sampling


def checkpoint(
    kernel,
    rng_key,
    file_name,
    threshold,
    statistic="rhat",
    file_path="",
    num_warmup=50000,
    max_samples=100000,
    num_chains=1,
    thinning=1,
    model_kwargs={},
    mcmc_kwargs={},
):

    if statistic == "rhat":
        statfunc = compute_rhats

    elif statistic == "n_eff":
        statfunc = compute_effective_sample_size
    else:
        raise ValueError("only effective sample size (n_eff) and gelman rubin diagnostic (rhat) are supported")

    num_samples = int(max_samples / 10)
    MCMC_RNG, PRIOR_RNG, _RNG = random.split(rng_key, num=3)

    mcmc = MCMC(kernel, thinning=thinning, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, **mcmc_kwargs)

    mcmc.run(MCMC_RNG, **model_kwargs)

    count = 1
    first_10percent_of_samples = mcmc.get_samples()
    first_10percent_idata = az.from_numpyro(mcmc)
    GWInfernoData(posterior=first_10percent_idata.posterior).to_netcdf(file_path + file_name + f"_{num_samples}x{count}_dataset.h5")
    print(f"checkpoint file {count} saved")
    keep_sampling = statfunc(first_10percent_of_samples, threshold=threshold, num_chains=num_chains)

    while keep_sampling:
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
        next_10percent_of_samples = mcmc.get_samples()
        next_10percent_idata = az.from_numpyro(mcmc)
        count += 1
        GWInfernoData(posterior=next_10percent_idata.posterior).to_netcdf(
            file_path + file_name + f"_{num_samples}x{count}_dataset.h5",
        )
        print(f"checkpoint file {count} saved")
        if count == 10:
            keep_sampling = False
            print("number of samples has exceeded max samples, sampling stopped")
        else:
            keep_sampling = statfunc(next_10percent_of_samples, threshold=threshold, num_chains=num_chains)

    if count > 1:
        dataset = GWInfernoData.from_netcdf(file_path + file_name + f"_{num_samples}x1_dataset.h5")
        dataset = dataset.posterior
        for i in np.arange(2, count + 1):
            dat = GWInfernoData.from_netcdf(file_path + file_name + f"_{num_samples}x{i}_dataset.h5")
            dat.posterior["draw"] = dat.posterior["draw"] + num_samples * (i - 1)
            dataset = xr.merge([dataset, dat.posterior])
        GWInfernoData(posterior=dataset).to_netcdf(file_path + file_name + f"_{num_samples*count}s_merged{count}_dataset.h5")

        return dataset

    else:
        return first_10percent_idata.posterior



def new_checkpoint(
    kernel,
    rng_key,
    file_label,
    file_path="",
    num_warmup=50000,
    max_samples=100000,
    split = 10,
    num_chains=1,
    thinning=1,
    model_kwargs={},
    mcmc_kwargs={},
):

    num_samples = int(max_samples / split)
    MCMC_RNG, PRIOR_RNG, _RNG = random.split(rng_key, num=3)

    mcmc = MCMC(kernel, thinning=thinning, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, **mcmc_kwargs)
    mcmc.run(MCMC_RNG, **model_kwargs)

    idata = az.from_numpyro(mcmc)
    mass_matrix = mcmc._last_state.adapt_state.inverse_mass_matrix
    step_size = mcmc._last_state.adapt_state.step_size

    key = list(mass_matrix.keys())[0]

    matrix = xr.Dataset({'mass_matrix': (list(key), np.asarray(mass_matrix[key]))}, attrs = {'step_size': step_size})

    GWInfernoData(posterior = idata.posterior, mass_matrix = matrix).to_netcdf(file_path + file_label + f"_{num_samples}x{count}_dataset.h5")
    print(f"checkpoint file {count} saved")
    del matrix, idata
    count = 1
    while count < split:

        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)
        new_mass_matrix = mcmc._last_state.adapt_state.inverse_mass_matrix
        new_step_size = mcmc._last_state.adapt_state.step_size
        count += 1
        idata = az.from_numpyro(mcmc)
        matrix = xr.Dataset({'mass_matrix': (list(key), np.asarray(new_mass_matrix[key]))}, attrs = {'step_size': new_step_size})
        GWInfernoData(posterior = idata.posterior, mass_matrix = matrix).to_netcdf(file_path + f'{file_label}_mass_matrix_{count}.h5')

        del matrix, idata


    print('merging data')
    dataset = GWInfernoData.from_netcdf(file_path + file_label + f"_{num_samples}x1_dataset.h5")
    dataset = dataset.posterior
    for i in np.arange(2, count + 1):
        dat = GWInfernoData.from_netcdf(file_path + file_label + f"_{num_samples}x{i}_dataset.h5")
        dat.posterior["draw"] = dat.posterior["draw"] + num_samples * (i - 1)
        dataset = xr.merge([dataset, dat.posterior])
    GWInfernoData(posterior=dataset).to_netcdf(file_path + file_label + f"_{num_samples*count}s_merged{count}_dataset.h5")

    return dataset