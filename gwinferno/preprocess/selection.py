"""
a module that stores functions for reading in and processing injection search results
"""

import h5py
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm import trange

from .conversions import chieff_from_q_component_spins
from .conversions import chip_from_q_component_spins
from .priors import chi_effective_prior_from_isotropic_spins
from .priors import joint_prior_from_isotropic_spins


def get_injection_dict(fi, ifar=1, snr=11, spin=False, additional_cuts=None):
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
            total_generated=int(data.attrs["total_generated"][()]),
            analysis_time=data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60,
        )
        if spin:
            for ii in [1, 2]:
                injs[f"a_{ii}"] = (
                    data.get(f"spin{ii}x", np.zeros(n_found))[()][found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[()][found] ** 2
                    + data[f"spin{ii}z"][()][found] ** 2
                ) ** 0.5
                injs[f"cos_tilt_{ii}"] = data[f"spin{ii}z"][()][found] / injs[f"a_{ii}"]
        injs["prior"] = data["sampling_pdf"][()][found] * data["mass1_source"][()][found]
        if spin:
            injs["prior"] *= (2 * np.pi * injs["a_1"] ** 2) * (2 * np.pi * injs["a_2"] ** 2)
    return injs


def get_semianlytic_injection_dict(fi, snr=8, additional_cuts=None):
    with h5py.File(fi, "r") as ff:
        data = ff["injections"]
        found = np.zeros_like(data["mass1_source"][()], dtype=bool)
        found = data["optimal_snr_l"][()] > snr
        if additional_cuts is not None:
            for k in additional_cuts.keys():
                found = found | data[k][()] > additional_cuts[k]
        injs = dict(
            mass_1=data["mass1_source"][()][found],
            mass_2=data["mass2_source"][()][found],
            mass_ratio=data["mass2_source"][()][found] / data["mass1_source"][()][found],
            redshift=data["redshift"][()][found],
            total_generated=int(data.attrs["total_generated"][()]),
            analysis_time=data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60,
        )
        injs["prior"] = data["sampling_pdf"][()][found] * data["mass1_source"][()][found]
    return injs


def load_injections(
    injfile,
    ifar_threshold=1,
    snr_threshold=11,
    spin=False,
    semianalytic=False,
    additional_cuts=None,
):
    if semianalytic:
        return get_semianlytic_injection_dict(injfile, additional_cuts=additional_cuts)
    else:
        return get_injection_dict(
            injfile,
            spin=spin,
            ifar=ifar_threshold,
            snr=snr_threshold,
            additional_cuts=additional_cuts,
        )


def convert_component_spin_injections_to_chieff(injdata, param_map, chip=False):
    new_params = ["mass_1", "mass_ratio", "redshift", "chi_eff", "prior"]
    if chip:
        new_params.append("chi_p")
    old_inj_shape = injdata.shape
    new_inj_shape = (len(new_params), old_inj_shape[1])
    new_inj_data = np.zeros(new_inj_shape)
    new_pmap = {p: i for i, p in enumerate(new_params)}
    chi_eff = chieff_from_q_component_spins(
        injdata[param_map["mass_ratio"]],
        injdata[param_map["a_1"]],
        injdata[param_map["a_2"]],
        injdata[param_map["cos_tilt_1"]],
        injdata[param_map["cos_tilt_2"]],
    )
    if chip:
        chi_p = chip_from_q_component_spins(
            injdata[param_map["mass_ratio"]],
            injdata[param_map["a_1"]],
            injdata[param_map["a_2"]],
            injdata[param_map["cos_tilt_1"]],
            injdata[param_map["cos_tilt_2"]],
        )
    new_prior = np.zeros_like(injdata[param_map["prior"]])
    for ii in trange(new_prior.shape[0]):
        if chip:
            new_prior[ii] = (
                injdata[param_map["prior"], ii]
                / ((2 * np.pi * injdata[param_map["a_1"], ii] ** 2) * (2 * np.pi * injdata[param_map["a_2"], ii] ** 2))
                * joint_prior_from_isotropic_spins(
                    np.array(injdata[param_map["mass_ratio"], ii]),
                    1.0,
                    np.array(chi_eff[ii]),
                    np.array(chi_p[ii]),
                )
            )
        else:
            new_prior[ii] = (
                injdata[param_map["prior"], ii]
                / ((2 * np.pi * injdata[param_map["a_1"], ii] ** 2) * (2 * np.pi * injdata[param_map["a_2"], ii] ** 2))
                * chi_effective_prior_from_isotropic_spins(
                    np.array(injdata[param_map["mass_ratio"], ii]),
                    1.0,
                    np.array(chi_eff[ii]),
                )
            )
    for p in new_params:
        if p not in ["prior", "chi_eff", "chi_p"]:
            new_inj_data[new_pmap[p]] = injdata[param_map[p]]
        elif p == "prior":
            new_inj_data[new_pmap[p]] = new_prior
        elif p == "chi_eff":
            new_inj_data[new_pmap[p]] = chi_eff
        else:
            new_inj_data[new_pmap[p]] = chi_p
    return new_inj_data, new_pmap


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
