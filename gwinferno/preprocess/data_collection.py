"""
a module that stores functions for collecting input data (GW Posteriors and Injections)
"""

import json

import arviz as az
import h5py
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.scipy.integrate import trapezoid
from tqdm import trange

from ..cosmology import PLANCK_2015_Cosmology as cosmo
from .conversions import chieff_from_q_component_spins
from .conversions import chip_from_q_component_spins
from .priors import chi_effective_prior_from_isotropic_spins
from .priors import joint_prior_from_isotropic_spins
from .selection import get_o3_cumulative_injection_dict
from .selection import get_o4a_cumulative_injection_dict


def collect_data(run_map):
    posteriors = {}
    for ev in list(run_map.keys()):
        with h5py.File(run_map[ev]["file_path"], "r") as f:
            wf = run_map[ev]["waveform"]
            z_prior = run_map[ev]["redshift_prior"]
            catalog = run_map[ev]["catalog"]
            if catalog == "GWTC-1":
                post = f[wf][:]
            else:
                post = f[wf]["posterior_samples"][:]
            posteriors[ev] = {"posterior": post, "redshift_prior": z_prior, "catalog": catalog}
    return posteriors


def format_data(posteriors):
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

    max_samples = 10000
    event_names = list(posteriors.keys())
    dataset = {}
    for ev in event_names:

        if posteriors[ev]["catalog"] == "GWTC-1":
            redshift = cosmo.DL2z(posteriors[ev]["posterior"]["luminosity_distance_Mpc"])
            mass = []
            spin = []
            tilt = []
            for ii in [1, 2]:
                mass.append(posteriors[ev]["posterior"][f"m{ii}_detector_frame_Msun"] / (1 + redshift))
                spin.append(posteriors[ev]["posterior"][f"spin{ii}"])
                tilt.append(posteriors[ev]["posterior"][f"costilt{ii}"])
            mass_ratio = mass[1] / mass[0]
            data = np.array([mass[0], mass[1], mass_ratio, redshift, spin[0], spin[1], tilt[0], tilt[1]])

        else:
            data = np.array([posteriors[ev]["posterior"][param_mapping[param]] for param in list(param_mapping.keys())])
        max_samples = min(data.shape[1], max_samples)
        data_array = xr.DataArray(
            data,
            dims=["param", "samples"],
            coords={"param": list(param_mapping.keys()), "samples": np.arange(0, data.shape[1])},
            attrs={"redshift_prior": posteriors[ev]["redshift_prior"], "catalog": posteriors[ev]["catalog"]},
        )
        dataset[ev] = data_array

    downsampled_data = {}
    for ev in list(dataset.keys()):
        downsamp = dataset[ev].sel({"samples": np.random.choice(dataset[ev].samples.values, max_samples, replace=False)})
        downsampled_data[ev] = downsamp.assign_coords(samples=np.arange(max_samples))

    catalog_dataset = xr.Dataset(data_vars=downsampled_data, coords={"param": list(param_mapping.keys()), "samples": np.arange(0, max_samples)})
    return catalog_dataset


def dl_2_prior_on_z(z, euclidean=False):
    if euclidean:
        dl = cosmo.z2DL(z) / 1e3
        return dl**2 * (dl / (1 + z) + (1 + z) * cosmo.dDcdz(z) / 1e3)
    else:
        return cosmo.dVcdz(z) * 4 * np.pi / (1 + z)


def evaluate_prior(full_catalog, param_names):
    if "redshift" in param_names:
        z_max = 1.9
        cat_z_max = full_catalog.sel(param="redshift").max().to_array().max().values
        z_max = cat_z_max if cat_z_max > z_max else z_max
        cat_z_max
        zs = jnp.linspace(0, z_max * 1.01, 1000)
        p_z_euclid = dl_2_prior_on_z(zs, euclidean=True)
        p_z_comoving = dl_2_prior_on_z(zs)
        p_z_euclid /= trapezoid(p_z_euclid, zs)
        p_z_comoving /= trapezoid(p_z_comoving, zs)

    events = list(full_catalog.data_vars)
    num_events = len(events)
    num_samples = full_catalog["samples"].shape[0]

    priors = jnp.zeros((num_events, 1, num_samples))
    for i, ev in enumerate(events):
        prior = jnp.ones(num_samples)
        if "redshift" in param_names:
            p_z = p_z_euclid if full_catalog[ev].attrs["redshift_prior"] == "euclidean" else p_z_comoving
            prior *= jnp.interp(full_catalog[ev].sel(param="redshift").values, zs, p_z)
        if "mass_1" in param_names:
            prior *= (1 + full_catalog[ev].sel(param="mass_1").values) ** 2  # flat detector components
        if "mass_raito" in param_names:
            prior *= full_catalog[ev].sel(param="mass_1").values
        if "a_1" in param_names:
            prior *= 1 / 4
        priors = priors.at[i].set(prior)

    prior_array = xr.DataArray(
        priors, dims=["event", "param", "samples"], coords={"param": ["prior"], "samples": np.arange(num_samples), "event": events}
    )
    catalog_array = full_catalog.to_array(dim="event")

    new_full_catalog = xr.concat([catalog_array, prior_array], dim="param")

    return new_full_catalog


def load_posterior_data(run_map=None, key_file=None, param_names=["mass_1", "mass_ratio", "redshift"]):
    if run_map is None:
        with open(key_file, "r") as f:
            run_map = json.load(f)
        if key_file is None:
            raise AssertionError("run_map or key_file must be specified")

    posterior_dict = collect_data(run_map)
    catalog = format_data(posterior_dict)
    full_catalog = evaluate_prior(catalog, param_names)

    if "chi_eff" in param_names:
        new_pe = convert_component_spins_to_chieff(full_catalog, param_names)
        remove = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]

        remove.append("mass_ratio") if "mass_2" in param_names else remove.append("mass_2")
        new_pe = new_pe.drop_sel(param=remove)
        return new_pe

    else:
        remove = list(np.setxor1d(full_catalog.param.values, np.array(param_names)))
        remove.remove("prior")
        full_catalog = full_catalog.drop_sel(param=remove)

        return full_catalog


def load_injections(injfile, param_names, through_o4a=True, through_o3=False, ifar_threshold=1, snr_threshold=11, additional_cuts=None):

    if through_o4a:
        injs = get_o4a_cumulative_injection_dict(
            injfile,
            param_names=param_names,
            ifar=ifar_threshold,
            snr=snr_threshold,
        )

    elif through_o3:

        injs = get_o3_cumulative_injection_dict(injfile, param_names, ifar=ifar_threshold, snr=snr_threshold, additional_cuts=additional_cuts)

    else:
        raise AssertionError("One kwarg `through_o3` or `through_o4a` must be true. Please specify which injection file you are using.")

    if "chi_eff" in param_names:
        new_injs = convert_component_spins_to_chieff(injs, param_names, injections=True)
        remove = ["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]

        remove.append("mass_ratio") if "mass_2" in param_names else remove.append("mass_2")
        new_injs = new_injs.drop_sel(param=remove)
        return new_injs

    else:
        remove = list(np.setxor1d(injs.param.values, np.array(param_names)))
        remove.remove("prior")
        return injs.drop_sel(param=remove)


def load_posterior_samples_and_injections(key_file, injfile, parameter_names, outdir, ifar_threshold=1, snr_threshold=11, save=False):
    pe_array = load_posterior_data(key_file=key_file, param_names=parameter_names).to_dataset(name="posteriors")
    inj_array = load_injections(injfile, parameter_names, ifar_threshold=ifar_threshold, snr_threshold=snr_threshold).to_dataset(name="injections")

    idata = az.InferenceData(pe_data=pe_array, inj_data=inj_array)

    label = ""
    for i in parameter_names:
        label = label + f"-{i}"

    if save:
        idata.to_netcdf(outdir + f"/xarray_through-o4a_posterior_samples_and_injections{label}.h5")

    return idata


def convert_component_spins_to_chieff(dat_array, param_names, injections=False):

    chip = True if "chi_p" in param_names else False

    q = dat_array.sel(param="mass_ratio").values
    a_1 = dat_array.sel(param="a_1").values
    a_2 = dat_array.sel(param="a_2").values
    tilt_1 = dat_array.sel(param="cos_tilt_1").values
    tilt_2 = dat_array.sel(param="cos_tilt_2").values
    prior = dat_array.sel(param="prior").values

    chi_eff = chieff_from_q_component_spins(
        q,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
    )
    if chip:
        chi_p = chip_from_q_component_spins(
            q,
            a_1,
            a_2,
            tilt_1,
            tilt_2,
        )

    if injections:

        new_prior = np.zeros_like(prior)
        for ii in trange(new_prior.shape[0]):
            if chip:
                new_prior[ii] = (
                    prior[ii]
                    / ((2 * jnp.pi * a_1[ii] ** 2) * (2 * jnp.pi * a_2[ii] ** 2))
                    * jnp.asarray(
                        joint_prior_from_isotropic_spins(
                            np.array(q[ii]),
                            1.0,
                            np.array(chi_eff[ii]),
                            np.array(chi_p[ii]),
                        )
                    )
                )
            else:

                new_prior[ii] = (
                    prior[ii]
                    / ((2 * jnp.pi * a_1[ii] ** 2) * (2 * jnp.pi * a_2[ii] ** 2))
                    * jnp.asarray(
                        chi_effective_prior_from_isotropic_spins(
                            np.asarray(q[ii]),
                            1.0,
                            np.asarray(chi_eff[ii]),
                        )
                    )[0]
                )

    else:
        new_prior = np.zeros_like(prior)
        for ii in trange(new_prior.shape[0]):
            for jj in range(new_prior.shape[1]):
                if chip:
                    new_prior[ii][jj] = (
                        prior[ii][jj]
                        / ((2 * jnp.pi * a_1[ii][jj] ** 2) * (2 * jnp.pi * a_2[ii][jj] ** 2))
                        * jnp.asarray(
                            joint_prior_from_isotropic_spins(
                                np.array(q[ii][jj]),
                                1.0,
                                np.array(chi_eff[ii][jj]),
                                np.array(chi_p[ii][jj]),
                            )
                        )
                    )
                else:
                    new_prior[ii][jj] = (
                        prior[ii][jj]
                        / ((2 * jnp.pi * a_1[ii][jj] ** 2) * (2 * jnp.pi * a_2[ii][jj] ** 2))
                        * jnp.asarray(
                            chi_effective_prior_from_isotropic_spins(
                                q[ii][jj],
                                1.0,
                                chi_eff[ii][jj],
                            )
                        )[0]
                    )

    new_arrays = []

    if injections:
        chi_eff_array = xr.DataArray(
            chi_eff.reshape(1, chi_eff.shape[0]),
            dims=["param", "injection"],
            coords={"param": ["chi_eff"], "injection": np.arange(dat_array.injection.shape[0])},
        )
        prior_array = xr.DataArray(
            new_prior.reshape(1, chi_eff.shape[0]),
            dims=["param", "injection"],
            coords={"param": ["prior"], "injection": np.arange(dat_array.injection.shape[0])},
        )
        new_arrays.append(chi_eff_array)
        new_arrays.append(prior_array)

        if chip:
            chip_array = xr.DataArray(
                chi_p.reshape(1, chi_p.shape[0]),
                dims=["param", "injection"],
                coords={"param": ["chi_p"], "injection": np.arange(dat_array.injection.shape[0])},
            )
            new_arrays.append(chip_array)

    else:
        chi_eff_array = xr.DataArray(
            chi_eff.reshape(chi_eff.shape[0], 1, chi_eff.shape[1]),
            dims=[
                "event",
                "param",
                "samples",
            ],
            coords={"event": dat_array.event, "param": ["chi_eff"], "samples": dat_array.samples},
        )
        prior_array = xr.DataArray(
            new_prior.reshape(prior.shape[0], 1, prior.shape[1]),
            dims=["event", "param", "samples"],
            coords={"event": dat_array.event, "param": ["prior"], "samples": dat_array.samples},
        )
        new_arrays.append(chi_eff_array)
        new_arrays.append(prior_array)

        if chip:
            chip_array = xr.DataArray(
                chi_p.reshape(1, chi_p.shape[0]),
                dims=["event", "param", "samples"],
                coords={"event": dat_array.event, "param": ["chi_p"], "samples": dat_array.samples},
            )
            new_arrays.append(chip_array)

    new_dat_array = dat_array.drop_sel(param="prior")

    for arr in new_arrays:
        new_dat_array = xr.concat([new_dat_array, arr], dim="param")

    return new_dat_array
