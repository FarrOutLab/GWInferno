import numpy as np


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
