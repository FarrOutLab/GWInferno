"""
a module that stores helpful conversion functions between parameters
"""

import jax.numpy as jnp
import numpy as np
import xarray as xr
from tqdm import trange

from .priors import chi_effective_prior_from_isotropic_spins
from .priors import joint_prior_from_isotropic_spins


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


def chieff_from_q_component_spins(q, a1, a2, ct1, ct2):
    """
    chieff_from_q_component_spins calculates the effective spin paramer given the
        mass ratio and spin mag/tilts.

    Args:
        q (array_like): mass ratio (defined from 0 to 1)
        a1 (array_like): primary spin magnitude
        a2 (array_like): secondary spin magnitude
        ct1 (array_like): primary cos tilt angle
        ct2 (array_like): secondary cos tilt angle

    Returns:
        array_like: effective spin
    """
    return (a1 * ct1 + q * a2 * ct2) / (1.0 + q)


def chip_from_q_component_spins(q, a1, a2, ct1, ct2, bk=np):
    """
    chip_from_q_component_spins calculates the effective spin precession paramer given
        the mass ratio and spin mag/tilts.

    Args:
        q (array_like): mass ratio (defined from 0 to 1)
        a1 (array_like): primary spin magnitude
        a2 (array_like): secondary spin magnitude
        ct1 (array_like): primary cos tilt angle
        ct2 (array_like): secondary cos tilt angle

    Returns:
        array_like: effective spin precession
    """
    sint1 = bk.sqrt(1.0 - ct1**2)
    sint2 = bk.sqrt(1.0 - ct2**2)
    return bk.maximum(a1 * sint1, ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * q * a2 * sint2)


def mu_var_from_alpha_beta(alpha, beta, amax=1):
    """
    mu_var_from_alpha_beta converts the alpha/beta shape parameters of the Beta
        Distribution to the mean and variance (mu/var)

    Args:
        alpha (array_like): alpha Beta distribution shape parameter
        beta (array_like): beta Beta distribution shape parameter
        amax (array_like, optional): maximum spin mag. Defaults to 1.

    Returns:
        (array_like, array_like): returns tuple of arrays same shape as input that
            represents the mean and variances given shape parameters
    """
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax**2
    return mu, var


def alpha_beta_from_mu_var(mu, var, amax=1):
    """
    alpha_beta_from_mu_var converts the mean/variance (mu/var) parameters of the Beta
        Distribution to the shape parameters (alpha/beta)

    Args:
        mu (array_like): mean of Beta distribution
        var (array_like): variance of Beta distribution
        amax (array_like, optional): maximum spin mag. Defaults to 1.

    Returns:
        (array_like, array_like): returns tuple of arrays same shape as input that
            represents the alpha and beta shape parameters given mean and variance
    """
    mu /= amax
    var /= amax**2
    alpha = (mu**2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta
