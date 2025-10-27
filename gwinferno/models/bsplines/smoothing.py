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

def apply_2d_difference_prior(coeffs, inv_var_row, inv_var_column, order=1):
    """Computes the P-spline difference penalty

    Args:
        coeffs (array-like): coefficients of the B-spline
        inv_var (float): inverse of the penalty tuning parameter. Higher values result in smoother functions
        order (int, default=1): difference order
    """
    delta_r_squared = jnp.sum(jnp.square(jnp.diff(coeffs, n=order, axis=-1)))
    delta_c_squared = jnp.sum(jnp.square(jnp.diff(coeffs, n=order, axis=-2)))
    prior = -0.5 * (inv_var_row * delta_r_squared + inv_var_column * delta_c_squared)
    return prior