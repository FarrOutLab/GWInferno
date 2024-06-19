"""
a module that stores functions for reading in and processing injection search results
"""

import h5py
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import random


def get_o4a_cumulative_injection_dict(file, param_names, ifar=1, snr=10):
    with h5py.File(file, "r") as ff:
        total_generated = ff.attrs["total_generated"]
        analysis_time = ff.attrs["analysis_time"]
        injections = np.asarray(ff["events"][:])

    found = injections["semianalytic_observed_phase_maximized_snr_net"] >= snr

    for key in injections.dtype.names:
        if "far" in key:
            found |= injections[key] <= 1 / ifar

    inj_weights = injections[found]["weights"]

    injs = dict(
        mass_1=injections["mass1_source"][found],
        mass_2=injections["mass2_source"][found],
        mass_ratio=injections["mass2_source"][found] / injections["mass1_source"][found],
        redshift=injections["redshift"][found],
    )

    inj_weights = inj_weights
    total_generated = int(total_generated)
    analysis_time = analysis_time / 365.25 / 24 / 60 / 60

    injs["prior"] = jnp.exp(injections["lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"][found]) / inj_weights

    if "mass_ratio" in param_names:
        injs["prior"] *= injections["mass1_source"][found]

    if ("a_1" in param_names) | ("chi_eff" in param_names):
        for ii in [1, 2]:
            injs[f"a_{ii}"] = (
                injections[f"spin{ii}x"][found] ** 2 + injections[f"spin{ii}y"][found] ** 2 + injections[f"spin{ii}z"][found] ** 2
            ) ** 0.5
            injs[f"cos_tilt_{ii}"] = injections[f"spin{ii}z"][found] / injs[f"a_{ii}"]
        injs["prior"] *= (2 * np.pi * injs["a_1"] ** 2) * (2 * np.pi * injs["a_2"] ** 2)

    injdata = np.array([injs[param] for param in list(injs.keys())])

    inj_array = xr.DataArray(
        injdata,
        dims=["param", "injection"],
        coords={"param": list(injs.keys()), "injection": np.arange(sum(found))},
        attrs={"total_generated": total_generated, "analysis_time": analysis_time},
    )

    return inj_array


def get_o3_cumulative_injection_dict(fi, param_names, ifar=1, snr=10, spin=False, additional_cuts=None):
    """
    Based from the function load_injection_data() at:
    https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe/-/blob/master/gwpopulation_pipe/vt_helper.py#L66
    """
    with h5py.File(fi, "r") as ff:
        data = ff["injections"]
        found = np.zeros_like(data["mass1_source"][()], dtype=bool)
        for key in data:
            if "ifar" in key.lower():
                found = found | (data[key][()] > ifar)
            if "name" in data.keys():
                gwtc1 = (data["name"][()] == b"o1") | (data["name"][()] == b"o2")
                found = found | (gwtc1 & (data["optimal_snr_net"][()] > snr))
        if additional_cuts is not None:
            for k in additional_cuts.keys():
                found = found | (data[k][()] >= additional_cuts[k])
        n_found = sum(found)
        injs = dict(
            mass_1=data["mass1_source"][()][found],
            mass_2=data["mass2_source"][()][found],
            mass_ratio=data["mass2_source"][()][found] / data["mass1_source"][()][found],
            redshift=data["redshift"][()][found],
        )

        total_generated = int(data.attrs["total_generated"][()])
        analysis_time = data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60

        injs["prior"] = data["sampling_pdf"][()][found]

        if ("a_1" in param_names) | ("chi_eff" in param_names):
            for ii in [1, 2]:
                injs[f"a_{ii}"] = (
                    data.get(f"spin{ii}x", np.zeros(n_found))[()][found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[()][found] ** 2
                    + data[f"spin{ii}z"][()][found] ** 2
                ) ** 0.5
                injs[f"cos_tilt_{ii}"] = data[f"spin{ii}z"][()][found] / injs[f"a_{ii}"]

            injs["prior"] *= (2 * np.pi * injs["a_1"] ** 2) * (2 * np.pi * injs["a_2"] ** 2)

        if "mass_ratio" in param_names:
            injs["prior"] *= data["mass1_source"][()][found]

    injdata = np.array([np.asarray(injs[param]) for param in list(injs.keys())])
    inj_array = xr.DataArray(
        injdata,
        dims=["param", "injection"],
        coords={"param": list(injs.keys()), "injection": np.arange(sum(found))},
        attrs={"total_generated": total_generated, "analysis_time": analysis_time},
    )

    return inj_array


def resample_injections(rng_key, model_prob, injdata, Ndraw, param_map, **kwargs):
    wts = model_prob(injdata, **kwargs) / injdata[param_map["prior"], :]
    p = wts / jnp.sum(wts)
    Ndet = len(p)
    # draw the maximum number of samples
    N = int((jnp.sum(wts)) ** 2 // jnp.sum(wts * wts))
    norm = jnp.sum(wts) / Ndraw
    idxs = random.choice(rng_key, Ndet, shape=[N], replace=True, p=p)
    injdata_new = injdata.at[:, idxs].get()
    p_new = model_prob(injdata_new, **kwargs) / norm
    injdata_new = injdata_new.at[param_map["prior"], :].set(p_new)
    s2_new = jnp.sum(wts * wts) / (Ndraw * Ndraw) - norm * norm / Ndraw
    Neff_new = norm * norm / s2_new
    return (injdata_new, N, Neff_new)
