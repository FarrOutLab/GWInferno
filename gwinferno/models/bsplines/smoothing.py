"""
a module that stores functions for calculating smoothing priors (i.e. P-Splines)
"""

import jax.numpy as jnp


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
    prior = -0.5 * inv_var * jnp.dot(delta_c, delta_c.T)
    return prior
