import jax.numpy as jnp
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.stats import beta
from scipy.stats import truncnorm
from scipy.stats import truncpareto

import gwinferno.distributions as dist

RTOL = 1e-5


def test_logistic_functions():
    """
    expit(x) = 1/(1+exp(-x))
             = logistic_function(x, L=1, k=1, x0=0)
             = logistic_unit(x, x0=0, sgn=-1, sc=1)
    """
    x = jnp.linspace(-10, 10, 50)
    L = 1
    k = 1
    x0 = 0
    sgn = -1
    sc = 1
    expectation = expit(x)
    assert_allclose(dist.logistic_function(x, L=L, k=k, x0=x0), expectation, rtol=RTOL)
    assert_allclose(dist.logistic_unit(x, x0=x0, sgn=sgn, sc=sc), expectation, rtol=RTOL)


def test_powerlaw_pdf():
    """
    Compare with scipy truncpareto, where its bounds are
        (scale + loc) <= x <= (c*scale + loc)
    and alpha = -(b + 1)
    """
    x = jnp.linspace(2, 55, 1000)
    alpha = -3.2
    xmin = 3.0
    xmax = 50.0
    b = -alpha - 1
    loc = 0.0
    scale = xmin - loc
    c = (xmax - loc) / scale
    expectation = truncpareto.pdf(x, b, c, loc=loc, scale=scale)
    assert_allclose(dist.powerlaw_pdf(x, alpha, xmin, xmax), expectation, rtol=RTOL)


def test_truncnorm_pdf():
    """
    Compare with scipy, noting that scipy's bounds are in sigma from mean
    """
    x = jnp.linspace(-1, 1.2, 50)
    mu = 0.3
    sigma = 1.4
    a_trunc = -0.8
    b_trunc = 1.0
    a = (a_trunc - mu) / sigma
    b = (b_trunc - mu) / sigma

    expectation = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
    assert_allclose(dist.truncnorm_pdf(x, mu, sigma, a_trunc, b_trunc, log=False), expectation, rtol=RTOL)


def test_truncnorm_pdf_log():
    """
    this is a truncated log-normal distribution
    """
    x = jnp.linspace(0.1, 10.2, 50)
    mu = 0.8
    sigma = 1.4
    log_a_trunc = -1.0
    log_b_trunc = 2.0
    a = (log_a_trunc - mu) / sigma
    b = (log_b_trunc - mu) / sigma

    expectation = truncnorm.pdf(jnp.log(x), a, b, loc=mu, scale=sigma) / x
    assert_allclose(dist.truncnorm_pdf(x, mu, sigma, jnp.exp(log_a_trunc), jnp.exp(log_b_trunc), log=True), expectation, rtol=RTOL)


def test_betadist():
    """
    beta distribution
    """
    x = jnp.linspace(0, 1, 50)
    a = 2
    b = 3
    expectation = beta.pdf(x, a, b)
    assert_allclose(dist.betadist(x, a, b), expectation, rtol=RTOL)
