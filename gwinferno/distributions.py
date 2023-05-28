import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.special import gammaln

"""
=============================================
This file contains some functions copied from https://github.com/ColmTalbot/gwpopulation re-implemented with jax.numpy
=============================================
"""


def log_logistic_unit(x, x0):
    """
    log_logistic_unit soft truncate a distribution with the log logistic unit

    Args:
        x (array_like): input array to truncate
        x0 (float): value of array we want to apply a soft truncation to

    Returns:
        array_like: input array with the soft truncation at x0 applied
    """
    diff = x - x0
    return jnp.where(
        jnp.greater(diff, 0),
        -jnp.log1p(jnp.exp(-4 * diff)),
        4 * diff - jnp.log1p(jnp.exp(4 * diff)),
    )


def logistic_unit(x, x0, sgn=1, sc=4):
    """
    logistic_unit soft truncate a distribution with the logistic unit

    Args:
        x (array_like): input array to truncate
        x0 (float): value of array we want to apply a soft truncation to
        sgn (int, optional): Which side do we truncate on (1 for right, -1 for left). Defaults to 1.
        sc (int, optional): scale of truncation, where higher values is sharper. Defaults to 4.

    Returns:
        array_like: input array with the soft truncation at x0 applied
    """
    return 1.0 / (1.0 + jnp.exp(sgn * sc * (x - x0)))


def powerlaw_logit_pdf(xx, alpha, high, fall_off):
    """
    powerlaw_logit_pdf pdf of high mass soft truncation powerlaw:
        $$ p(x) \propto x^{\alpha}\Theta(x-x_\mathrm{min})\Theta(x_\mathrm{max}-x) $$

    Args:
        xx (array_like): points to evaluate pdf at
        alpha (float): power law index
        high (float): high end truncation bound
        fall_off (float): scale of logistic unit to truncate distribution

    Returns:
        array_like: pdf evaluated at xx
    """
    prob = jnp.power(xx, alpha) * logistic_unit(xx, high, sign=1.0, a=fall_off)
    return prob


def powerlaw_pdf(xx, alpha, low, high, floor=0.0):
    """
    powerlaw_pdf pdf of sharp truncated powerlaw:

    Args:
        xx (array_like): points to evaluate pdf at
        alpha (float): power law index
        low (float): low end truncation bound
        high (float): high end truncation bound
        floor (float, optional): lower bound of pdf (Defaults to 0.0)
    """
    prob = jnp.power(xx, alpha)
    return jnp.where(jnp.less(xx, low) | jnp.greater(xx, high), floor, prob)


def truncnorm_pdf(xx, mu, sig, low, high, log=False):
    """
    $$ p(x) \propto \mathcal{N}(x | \mu, \sigma)\Theta(x-x_\mathrm{min})\Theta(x_\mathrm{max}-x) $$
    """
    prob = jnp.exp(-jnp.power(xx - mu, 2) / (2 * sig**2))
    if log:
        norm_num = 2**0.5 / jnp.pi**0.5 / sig / jnp.exp(xx)
    else:
        norm_num = 2**0.5 / jnp.pi**0.5 / sig
    norm_denom = erf((high - mu) / 2**0.5 / sig) + erf((mu - low) / 2**0.5 / sig)
    norm = norm_num / norm_denom
    return jnp.where(jnp.greater(xx, high) | jnp.less(xx, low), 0, prob * norm)


"""
def truncnorm_pdf(xx, mu, sig, low, high, floor=0.0):
    truncnorm_pdf pdf of high mass truncated normal distributon:
        $$ p(x) \propto \mathcal{N}(x | \mu, \sigma)\Theta(x-x_\mathrm{min})\Theta(x_\mathrm{max}-x) $$

    Args:
        xx (array_like): points to evaluate pdf at
        mu (float): mean of normal distributon
        sig (float): standar deviation of the normal distribution
        low (float): low end truncation bound
        high (float): high end truncation bound
        floor (float, optional): lower bound of pdf (Defaults to 0.0)

    Returns:
        array_like: pdf evaluated at xx
    prob = jnp.exp(-jnp.power(xx - mu, 2) / (2 * sig**2))
    norm_num = 2**0.5 / jnp.pi**0.5 / sig
    norm_denom = erf((high - mu) / 2**0.5 / sig) + erf((mu - low) / 2**0.5 / sig)
    norm = norm_num / norm_denom
    return jnp.where(jnp.greater(xx, high) | jnp.less(xx, low), floor, prob * norm)
    """


def ln_beta_fct(alpha, beta):
    """
    ln_beta_fct evaluate log beta fct (see: )

    Args:
        alpha (float): alpha shape parameter
        beta (float): beta shape parameter

    Returns:
        float: log Beta fct
    """
    return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)


def betadist(xx, alpha, beta, scale=1.0, floor=0.0):
    """
    betadist pdf of Beta distribution evaluated at xx with optional max vale of scale:

    Args:
        xx (array_like): points to evaluate pdf at
        alpha (float): alpha shape parameter
        beta (float): beta shape parameter
        scale (float, optional): maximum value of support in Beta distribution. Defaults to 1.0.
        floor (float, optional): lower bound of pdf (Defaults to 0.0)

    Returns:
        array_like: pdf evaluated at xx
    """
    ln_beta = (alpha - 1) * jnp.log(xx) + (beta - 1) * jnp.log(scale - xx) - (alpha + beta - 1) * jnp.log(scale)
    ln_beta = ln_beta - ln_beta_fct(alpha, beta)
    return jnp.where(jnp.less_equal(xx, scale) & jnp.greater_equal(xx, 0), jnp.exp(ln_beta), floor)
