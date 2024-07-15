"""
a module that stores functions for calculating smoothing priors (i.e. P-Splines)
"""

import jax.numpy as jnp


def apply_difference_prior(coefs, inv_var, degree=1):
    """Computes the P-Spline difference penalty.

    Parameters
    ----------
    coefs : array_like
        B-Spline coefficients.
    inv_var : float
        Inverse of the penalty tuning parameter.  Higher values result in
        smoother functions.
    degree : int, default=1
        Difference order.

    Returns
    -------
    float
        Log difference prior.
    """
    delta_c = jnp.diff(coefs, n=degree)
    prior = -0.5 * inv_var * jnp.dot(delta_c, delta_c.T)
    return prior
