"""
a module that stores helpful conversion functions between parameters
"""

import numpy as np


def chieff_from_q_component_spins(q, a1, a2, ct1, ct2):
    r"""Calculate the effective spin from, mass ratio, component spin magnitudes
    and tilts.

    .. math::
        \chi_\mathrm{eff} = \frac{a_1 \cos t_1 + q a_2 \cos \theta_2}{1 + q}

    Parameters
    ----------
    q : array_like
        Mass ratio (defined from 0 to 1).
    a1 : array_like
        Spin magnitude of the primary component.
    a2 : array_like
        Spin magnitude of the secondary component.
    ct1 : array_like
        Cosine of the spin tilt of the primary component.
    ct2 : array_like
        Cosine of the spin tilt of the secondary component.

    Returns
    -------
    array_like
        Effective spin.
    """
    return (a1 * ct1 + q * a2 * ct2) / (1.0 + q)


def chip_from_q_component_spins(q, a1, a2, ct1, ct2, math=np):
    """Calculate the effective precessing spin, :math:`\chi_\mathrm{p}`, from
    the mass ratio, component spin magnitudes and tilts.

    Parameters
    ----------
    q : array_like
        Mass ratio (defined from 0 to 1).
    a1 : array_like
        Spin magnitude of the primary component.
    a2 : array_like
        Spin magnitude of the secondary component.
    ct1 : array_like
        Cosine of the spin tilt of the primary component.
    ct2 : array_like
        Cosine of the spin tilt of the secondary component.
    math : module, default=`numpy`
        Math module to use.

    Returns
    -------
    array_like
        Effective precessing spin.
    """
    sint1 = math.sqrt(1.0 - ct1**2)
    sint2 = math.sqrt(1.0 - ct2**2)
    return math.maximum(a1 * sint1, ((3.0 + 4.0 * q) / (4.0 + 3.0 * q)) * q * a2 * sint2)


def mu_var_from_alpha_beta(alpha, beta, xmax=1):
    """Convert α, β shape parameters of the Beta distribution to the mean and variance.

    Parameters
    ----------
    alpha : array_like
        α shape parameter.
    beta : array_like
        β shape parameter.
    xmax : array_like, default=1
        Maximum value with support.

    Returns
    -------
    array_like, array_like
        Tuple of arrays with shape matching inputs that represent the
        means and variances corresponding to the shape parameters.
    """
    mu = alpha / (alpha + beta) * xmax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * xmax**2
    return mu, var


def alpha_beta_from_mu_var(mu, var, xmax=1):
    """Convert desired mean and variance to α, β shape parameters of the Beta distribution.

    Parameters
    ----------
    mean : array_like
        Desired mean.
    var : array_like
        Desired variance.
    xmax : array_like, default=1
        Maximum value with support.

    Returns
    -------
    array_like, array_like
        Tuple of arrays with shape matching inputs that represent the
        α, β shape parameters corresponding to the desired means and variances.
    """
    mu /= xmax
    var /= xmax**2
    alpha = (mu**2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta
