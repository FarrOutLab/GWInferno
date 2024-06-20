from numpy.testing import assert_allclose, assert_array_equal
import gwinferno.distributions as dist
import jax.numpy as jnp
from scipy.special import expit


def test_logistic_functions():
    """
    expit(x) = 1/(1+exp(-x))
             = logistic_function(x, L=1, k=1, x0=0)
             = logistic_unit(x, x0=0, sgn=-1, sc=1)
    """
    x = jnp.linspace(-10, 10, 100)
    L = 1
    k = 1
    x0 = 0
    sgn = -1
    sc = 1
    expectation = expit(x)
    assert_allclose(dist.logistic_function(x, L=L, k=k, x0=x0), expectation, rtol=1e-6)
    assert_allclose(dist.logistic_unit(x, x0=x0, sgn=sgn, sc=sc), expectation, rtol=1e-6)