"""
a module that stores functions for collecting input data (GW Posteriors and Injections)
"""

import json

import deepdish as dd
import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import trange

from ..cosmology import PLANCK_2018_Cosmology as cosmo
from .conversions import chieff_from_q_component_spins
from .conversions import chip_from_q_component_spins
from .priors import chi_effective_prior_from_isotropic_spins
from .priors import joint_prior_from_isotropic_spins
from .selection import convert_component_spin_injections_to_chieff
from .selection import load_injections

GWTC1 = [
    "GW150914",
    "GW151012",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170729",
    "GW170809",
    "GW170814",
    "GW170817",
    "GW170818",
    "GW170823",
]
EVENTS_WITH_NS = [
    "GW170817",
    "GW190425",
    "GW190814",
    "GW190426_152155",
    "GW190917_114630",
    "GW200105_162426",
    "GW200115_042309",
]


def dl_2_prior_on_z(z):
    dl = cosmo.z2DL(z) / 1e3
    return dl**2 * (dl / (1 + z) + (1 + z) * cosmo.dDcdz(z, mpc=True) / 1e3)


def p_m1src_q_z_lal_pe_prior(posts, spin=False):
    z_max = 1.9
    for ev in posts.keys():
        if max(posts[ev]["redshift"]) > z_max:
            z_max = max(posts[ev]["redshift"])
    zs = jnp.linspace(0, z_max * 1.01, 1000)
    p_z = dl_2_prior_on_z(zs)
    p_z /= jnp.trapz(p_z, zs)
    for event, post in posts.items():
        posts[event]["prior"] = jnp.interp(np.array(post["redshift"]), zs, p_z) * post["mass_1"] * (1 + post["redshift"]) ** 2
        if spin:
            posts[event]["prior"] /= 4
    return posts


def _standardize_new_post(df, param_mapping):
    try:
        return pd.DataFrame({key: df[new_key] for key, new_key in param_mapping.items()})
    except KeyError:  # Catch error if we use this for posterior samples with no spin
        param_map = dict(
            mass_1="mass_1_source",
            mass_2="mass_2_source",
            mass_ratio="mass_ratio",
            redshift="redshift",
        )
        return pd.DataFrame({key: df[new_key] for key, new_key in param_map.items()})


def _standardize_GWTC1_post(df):
    post = pd.DataFrame()
    post["redshift"] = cosmo.DL2z(df["luminosity_distance_Mpc"])
    for ii in [1, 2]:
        post[f"mass_{ii}"] = df[f"m{ii}_detector_frame_Msun"] / (1 + post["redshift"])
        post[f"a_{ii}"] = df[f"spin{ii}"]
        post[f"cos_tilt_{ii}"] = df[f"costilt{ii}"]
    post["mass_ratio"] = post["mass_2"] / post["mass_1"]
    return post


def _standardize_posterior_fmt(df, k):
    param_mapping = dict(
        mass_1="mass_1_source",
        mass_2="mass_2_source",
        mass_ratio="mass_ratio",
        redshift="redshift",
        a_1="a_1",
        a_2="a_2",
        cos_tilt_1="cos_tilt_1",
        cos_tilt_2="cos_tilt_2",
    )
    if isinstance(df, pd.DataFrame) and k not in GWTC1:
        post = _standardize_new_post(df, param_mapping)
    elif k in GWTC1:  # GWTC1 posterior fmt has detector frame -- we need to convert to source
        post = _standardize_GWTC1_post(df)
    else:
        raise ValueError(f"Event {k} not able to converted to correct parameters...")
    return post


def downsample_posteriors_to_consistent_nsamps(posteriors, max_samples=10000):
    data = {}
    names = []
    for ev, posterior in posteriors.items():
        max_samples = min(len(posterior), max_samples)
        names.append(ev)
    print(f"Saving {max_samples} samples for each of the {len(names)} events")
    for ev, posterior in posteriors.items():
        data[ev] = posterior.sample(max_samples)
    return data, names


def preprocess_data(data_dir, run_map, ignore=[], spin=False, max_samples=10000, no_downsample=False):
    posteriors = {}
    for event, wf in run_map.items():
        if event in ignore:
            continue
        print(f"loading {wf} for {event}")
        if wf == "MDC":
            with h5py.File(f"{data_dir}/{event}.h5", "r") as ff:
                post = pd.DataFrame(dict(ff[wf]["posterior_samples"]))
                posteriors[event] = post
        elif event not in GWTC1:
            with h5py.File(f"{data_dir}/{event}.h5", "r") as ff:
                post = pd.DataFrame(ff[wf]["posterior_samples"][:])
                posteriors[event] = post
        else:
            with h5py.File(f"{data_dir}/{event}.h5", "r") as ff:
                post = np.array(ff[f"{wf}_posterior"])
                posteriors[event] = post
    posteriors = p_m1src_q_z_lal_pe_prior(
        {k: _standardize_posterior_fmt(v, k) for k, v in posteriors.items()},
        spin=spin,
    )
    print(f"Loaded {len(posteriors.keys())} Single Event Posteriors")
    if no_downsample:
        return posteriors, [ev for ev in posteriors.keys()]
    posteriors, names = downsample_posteriors_to_consistent_nsamps(posteriors, max_samples=max_samples)
    return posteriors, names


def apply_priors(posteriors, spin=False, downsample=True, max_samples=10000):
    posteriors = p_m1src_q_z_lal_pe_prior(
        {k: _standardize_posterior_fmt(v, k) for k, v in posteriors.items()},
        spin=spin,
    )
    print(f"Loaded {len(posteriors.keys())} Single Event Posteriors")
    if downsample:
        posteriors, _ = downsample_posteriors_to_consistent_nsamps(posteriors, max_samples=max_samples)
    return posteriors, [ev for ev in posteriors.keys()]


def _load_single_posterior(fi, ev, wf):
    if wf == "MDC":
        with h5py.File(fi, "r") as ff:
            post = pd.DataFrame(dict(ff[wf]["posterior_samples"]))
    elif ev not in GWTC1:
        with h5py.File(fi, "r") as ff:
            post = pd.DataFrame(ff[wf]["posterior_samples"][:])
    else:
        with h5py.File(fi, "r") as ff:
            post = np.array(ff[f"{wf}_posterior"])
    return post


def load_catalog_from_metadata(catalog_summary_file, **kwargs):
    with open(catalog_summary_file, "r") as f:
        catsum = json.load(f)
    posteriors = {}
    for ev in catsum.keys():
        if ev == "Injections" or ev == "metadata_directory" or ev == "FAR_threshold":
            continue
        posteriors[ev] = _load_single_posterior(catsum[ev]["path"], ev, catsum[ev]["waveform"])
    return apply_priors(posteriors, **kwargs), catsum["Injections"], catsum["FAR_threshold"], catsum["metadata_directory"]


def load_posterior_samples(data_dir, run_map=None, keyfile=None, ignore=None, bbh=True, spin=False, max_samples=10000, no_downsample=False):
    if run_map is None:
        if keyfile is None:
            keyfile = f"{data_dir}/keys_to_read.json"
        with open(keyfile, "r") as f:
            run_map = json.load(f)
    if ignore is None:
        ignore = EVENTS_WITH_NS if bbh else []
    posteriors, names = preprocess_data(data_dir, run_map, ignore=ignore, spin=spin, max_samples=max_samples, no_downsample=no_downsample)
    return posteriors, names


def setup_posterior_samples_and_injections(data_dir, inj_file, param_names=None, chi_eff=False, chi_p=False, save=False, jax_device=None):
    """
    Sets up posterior sample and injections to be used during inference or saves them to a file

        Parameters:
            data_dir (strs): path to directory where posterior samples are stored
            inj_file (strs): path to injection file
            param_names (list of strs): list of desired parameters to be saved in new file
            chi_eff (bool): True converts spin magnitude parameters to chi_eff.
            chi_p (bool): True converts spin magnitude parameters to chi_p
            save (bool): True saves the desired output to a .h5 datafile

        Returns:
            pedata (jax.numpy array): posterior samples for the parameters specified in param_names
            injdata (jax.numpy array): injection data for the parameters specified in param_names
            param_map (dict): dictionary that associates an index to each name in param_names
            inj_attributes (dict): dictionary that includes total number of generated injections
                                    analysis time
            names (list of strs): list with the names of each GW event
    """

    if param_names is None:
        param_names = ["mass_1", "mass_ratio", "redshift", "prior"]
    if "a_1" in param_names or "cos_tilt_1" in param_names:
        spin = True
    else:
        spin = False
    injections = load_injections(inj_file, spin=spin)
    pe_samples, names = load_posterior_samples(data_dir, spin=spin)
    param_map = {p: i for i, p in enumerate(param_names)}
    inj_attributes = {
        "total_generated": injections["total_generated"],
        "analysis_time": injections["analysis_time"],
    }
    if chi_eff and spin:
        pedata = np.array([[pe_samples[e][p] for e in names] for p in param_names])
        injdata = np.array([injections[k] for k in param_names])
        if save:
            mag_data = {
                "injdata": jnp.array(injdata),
                "pedata": jnp.array(pedata),
                "param_map": param_map,
                "total_generated": injections["total_generated"],
                "analysis_time": injections["analysis_time"],
            }
            dd.io.save("posterior_samples_and_injections_spin_magnitude", mag_data)

        pedata, new_pmap = convert_component_spin_posteriors_to_chieff(pedata, param_map, chip=chi_p)
        injdata, new_pmap = convert_component_spin_injections_to_chieff(injdata, param_map, chip=chi_p)
        param_map = new_pmap
        pedata = jnp.array(pedata)
        injdata = jnp.array(pedata)
        if save:
            mag_data = {
                "injdata": injdata,
                "pedata": pedata,
                "param_map": param_map,
                "total_generated": injections["total_generated"],
                "analysis_time": injections["analysis_time"],
            }
            dd.io.save("posterior_samples_and_injections_chi_effective.h5", mag_data)

    else:
        pedata = jnp.array([[pe_samples[e][p] for e in names] for p in param_names])
        injdata = jnp.array([injections[k] for k in param_names])
        if save:
            mag_data = {
                "injdata": injdata,
                "pedata": pedata,
                "param_map": param_map,
                "total_generated": injections["total_generated"],
                "analysis_time": injections["analysis_time"],
            }
            dd.io.save("posterior_samples_and_injections_spin_magnitude.h5", mag_data)

    return pedata, injdata, param_map, inj_attributes, names


def convert_component_spin_posteriors_to_chieff(pedata, param_map, chip=False):
    new_params = ["mass_1", "mass_ratio", "redshift", "chi_eff", "prior"]
    if chip:
        new_params.append("chi_p")
    old_pe_shape = pedata.shape
    new_pe_shape = (len(new_params), old_pe_shape[1], old_pe_shape[2])
    new_pe_data = np.zeros(new_pe_shape)
    new_pmap = {p: i for i, p in enumerate(new_params)}
    chi_eff = chieff_from_q_component_spins(
        pedata[param_map["mass_ratio"]],
        pedata[param_map["a_1"]],
        pedata[param_map["a_2"]],
        pedata[param_map["cos_tilt_1"]],
        pedata[param_map["cos_tilt_2"]],
    )
    if chip:
        chi_p = chip_from_q_component_spins(
            pedata[param_map["mass_ratio"]],
            pedata[param_map["a_1"]],
            pedata[param_map["a_2"]],
            pedata[param_map["cos_tilt_1"]],
            pedata[param_map["cos_tilt_2"]],
        )
    new_prior = np.zeros_like(pedata[param_map["prior"]])
    for jj in trange(new_prior.shape[0]):
        for ii in range(new_prior.shape[1]):
            if chip:
                new_prior[jj, ii] = (
                    pedata[param_map["prior"], jj, ii]
                    * 4
                    * joint_prior_from_isotropic_spins(
                        np.array(pedata[param_map["mass_ratio"], jj, ii]),
                        1.0,
                        np.array(chi_eff[jj, ii]),
                        np.array(chi_p[jj, ii]),
                    )
                )
            else:
                new_prior[jj, ii] = (
                    pedata[param_map["prior"], jj, ii]
                    * 4
                    * chi_effective_prior_from_isotropic_spins(
                        np.array(pedata[param_map["mass_ratio"], jj, ii]),
                        1.0,
                        np.array(chi_eff[jj, ii]),
                    )
                )
    for p in new_params:
        if p not in ["prior", "chi_eff", "chi_p"]:
            new_pe_data[new_pmap[p]] = pedata[param_map[p]]
        elif p == "prior":
            new_pe_data[new_pmap[p]] = new_prior
        elif p == "chi_eff":
            new_pe_data[new_pmap[p]] = chi_eff
        else:
            new_pe_data[new_pmap[p]] = chi_p
    return new_pe_data, new_pmap
