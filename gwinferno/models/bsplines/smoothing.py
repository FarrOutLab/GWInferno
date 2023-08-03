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


def apply_twod_difference_prior(coefs, inv_var_c, inv_var_r, degree_c=1, degree_r=1):
    delta_c = jnp.diff(coefs, n=degree_c, axis=1)
    delta_r = jnp.diff(coefs, n=degree_r, axis=0)
    pen_c = -0.5 * inv_var_c * jnp.sum(jnp.square(delta_c))
    pen_r = -0.5 * inv_var_r * jnp.sum(jnp.square(delta_r))
    return pen_c + pen_r


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
