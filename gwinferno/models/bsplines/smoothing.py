"""
a module that stores functions for calculating smoothing priors (i.e. P-Splines)
"""

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist


def apply_difference_prior(coefs, inv_var, degree=1):
    """computes the difference penalty for b-spline

    Args:
        coefs (array_like): b-spline coefficients
        inv_var (float): inverse of the penalty tuning parameter.
        degree (int, optional): difference order. Defaults to 1.

    Returns:
        float: log of the difference prior
    """
    delta_c = jnp.diff(coefs, n=degree)
    return -0.5 * inv_var * jnp.dot(delta_c, delta_c.T)


def apply_twod_difference_prior(coefs, inv_var_row, inv_var_col, degree_row=1, degree_col=1):
    """
    Computes the difference penalty for a 2d B-spline.
    Uses equation 4.19 from Practical Smoothing by Eilers and Marx.

    Parameters
    ----------
    coefs: array_like
        2d array of B-spline coefficients
    inv_var_row: float or int
        inverse variance along axis 0 (row)
    inv_var_col: float or int
        inverse variance along axis 1 (column)
    degree_row: int, optional
        difference order along axis 0 (row)
    degree_col: int, optional
        difference order along axis 1 (column)
    
    Returns
    -------
    float
        log of the difference prior
    """
    delta_row = jnp.diff(coefs, n=degree_row, axis=0)
    delta_col = jnp.diff(coefs, n=degree_col, axis=1)
    pen_row = -0.5 * inv_var_row * jnp.sum(jnp.square(delta_row))
    pen_col = -0.5 * inv_var_col * jnp.sum(jnp.square(delta_col))
    return pen_row + pen_col


def get_adaptive_Lambda(label, nknots, degree, omega=0.5):
    lam = numpyro.sample(f"lambda_{label}", dist.Gamma(omega, omega), sample_shape=(nknots - degree - 1,))
    li = [1.0]
    for i, la in enumerate(lam):
        li.append(li[i] * la)
    return jnp.diag(jnp.array(li))


def mixture_smoothing_parameter(label, n_mix=20, log10bmin=-5, log10bmax=5):
    bs = jnp.logspace(log10bmin, log10bmax, num=n_mix)
    ps = numpyro.sample(f"{label}_ps", dist.Dirichlet(jnp.ones(n_mix)))
    gs = numpyro.sample(f"{label}_gs", dist.Gamma(jnp.ones_like(bs), bs))
    return jnp.sum(ps * gs)
