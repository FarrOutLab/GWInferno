"""
a module defining additional numpyro distributions
"""

import jax.numpy as jnp
from jax import lax
from jax import random
from jax import vmap
from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import validate_sample

from .cosmology import PLANCK_2018_Cosmology as cosmo
from .interpolation import NaturalCubicUnivariateSpline
from .models.bsplines.smoothing import apply_difference_prior


def cumtrapz(y, x):
    difs = jnp.diff(x)
    idxs = jnp.array([i for i in range(1, len(y))])
    res = jnp.cumsum(vmap(lambda i, d: d * (y[i] + y[i + 1]) / 2.0)(idxs, difs))
    return jnp.concatenate([jnp.array([0]), res])


class Sine(Distribution):
    arg_constraints = {
        "minimum": constraints.real,
        "maximum": constraints.real,
    }
    reparametrized_params = ["minimum", "maximum"]

    def __init__(self, minimum=0.0, maximum=jnp.pi, validate_args=None):
        self.minimum, self.maximum = promote_shapes(minimum, maximum)
        self._support = constraints.interval(minimum, maximum)
        batch_shape = lax.broadcast_shapes(jnp.shape(minimum), jnp.shape(maximum))
        super(Sine, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        return jnp.sin(value) / 2.0

    def cdf(self, value):
        cdf = jnp.atleast_1d((jnp.cos(value) - jnp.cos(self.minimum)) / (jnp.cos(self.maximum) - jnp.cos(self.minimum)))
        cdf = jnp.where(jnp.less(value, self.minimum), 0.0, cdf)
        cdf = jnp.where(jnp.greater(value, self.maximum), 1.0, cdf)
        return cdf

    def icdf(self, q):
        norm = jnp.cos(self.minimum) - jnp.cos(self.maximum)
        return jnp.arccos(jnp.cos(self.minimum) - q * norm)


class Cosine(Distribution):
    arg_constraints = {
        "minimum": constraints.real,
        "maximum": constraints.real,
    }
    reparametrized_params = ["minimum", "maximum"]

    def __init__(self, minimum=-jnp.pi / 2.0, maximum=jnp.pi / 2.0, validate_args=None):
        self.minimum, self.maximum = promote_shapes(minimum, maximum)
        self._support = constraints.interval(minimum, maximum)
        batch_shape = lax.broadcast_shapes(jnp.shape(minimum), jnp.shape(maximum))
        super(Cosine, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        return jnp.cos(value) / 2.0

    def cdf(self, value):
        cdf = jnp.atleast_1d((jnp.sin(value) - jnp.sin(self.minimum)) / (jnp.sin(self.maximum) - jnp.sin(self.minimum)))
        cdf = jnp.where(jnp.less(value, self.minimum), 0.0, cdf)
        cdf = jnp.where(jnp.greater(value, self.maximum), 1.0, cdf)
        return cdf

    def icdf(self, q):
        norm = jnp.sin(self.minimum) - jnp.sin(self.maximum)
        return jnp.arcsin(jnp.sin(self.minimum) - q * norm)


class Powerlaw(Distribution):
    arg_constraints = {
        "minimum": constraints.real,
        "maximum": constraints.real,
        "alpha": constraints.real,
    }
    reparametrized_params = ["minimum", "maximum", "alpha"]

    def __init__(self, alpha, minimum=0.0, maximum=1.0, low=0.0, high=0.0, validate_args=None):
        self.minimum, self.maximum, self.alpha = promote_shapes(minimum, maximum, alpha)
        self._support = constraints.interval(low, high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(minimum),
            jnp.shape(maximum),
            jnp.shape(alpha),
        )
        super(Powerlaw, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        logp = self.alpha * jnp.log(value)
        logp = logp + jnp.log((1.0 + self.alpha) / (self.maximum ** (1.0 + self.alpha) - self.minimum ** (1.0 + self.alpha)))
        logp_neg1 = -jnp.log(value) - jnp.log(self.maximum / self.minimum)
        return jnp.where(
            jnp.less(value, self.minimum) | jnp.greater(value, self.maximum),
            jnp.nan_to_num(-jnp.inf),
            jnp.where(jnp.equal(self.alpha, -1.0), logp_neg1, logp),
        )

    def cdf(self, value):
        cdf = jnp.atleast_1d(value ** (self.alpha + 1.0) - self.minimum ** (self.alpha + 1.0)) / (
            self.maximum ** (self.alpha + 1.0) - self.minimum ** (self.alpha + 1.0)
        )
        cdf_neg1 = jnp.log(value / self.minimum) / jnp.log(self.maximum / self.minimum)
        cdf = jnp.where(jnp.equal(self.alpha, -1.0), cdf_neg1, cdf)
        cdf = jnp.minimum(cdf, 1.0)
        cdf = jnp.maximum(cdf, 0.0)
        return cdf

    def icdf(self, q):
        icdf = (self.minimum ** (1.0 + self.alpha) + q * (self.maximum ** (1.0 + self.alpha) - self.minimum ** (1.0 + self.alpha))) ** (
            1.0 / (1.0 + self.alpha)
        )
        icdf_neg1 = self.minimum * jnp.exp(q * jnp.log(self.maximum / self.minimum))
        return jnp.where(jnp.equal(self.alpha, -1.0), icdf_neg1, icdf)


class NumericallyNormalizedDistribition(Distribution):
    arg_constraints = {
        "maximum": constraints.real,
        "minimum": constraints.real,
    }
    reparametrized_params = ["maximum", "minimum"]

    def __init__(self, minimum, maximum, Ngrid=1000, grid=None, validate_args=None):
        self.maximum, self.minimum = promote_shapes(maximum, minimum)
        self._support = constraints.real
        batch_shape = lax.broadcast_shapes(
            jnp.shape(maximum),
            jnp.shape(minimum),
        )
        super(NumericallyNormalizedDistribition, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.grid = grid if grid is not None else jnp.linspace(minimum, maximum, Ngrid)
        self.pdfs = jnp.exp(self._log_prob_nonorm(self.grid))
        self.norm = jnp.trapz(self.pdfs, self.grid)
        self.pdfs /= self.norm
        self.cdfgrid = cumtrapz(self.pdfs, self.grid)
        self.cdfgrid = self.cdfgrid.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    def _log_prob_nonorm(self, value):
        raise NotImplementedError

    @validate_sample
    def log_prob(self, value):
        return jnp.where(
            jnp.less(value, self.minimum) | jnp.greater(value, self.maximum),
            jnp.nan_to_num(-jnp.inf),
            self._log_prob_nonorm(value) - jnp.log(self.norm),
        )

    def cdf(self, value):
        return jnp.interp(value, self.grid, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.grid)


class PowerlawRedshift(NumericallyNormalizedDistribition):
    arg_constraints = {"maximum": constraints.positive, "lamb": constraints.real}
    reparametrized_params = ["maximum", "lamb"]

    def __init__(self, lamb, maximum, minimum=1e-11, Ngrid=1000, grid=None, validate_args=None):
        self.lamb = lamb
        super().__init__(minimum=minimum, maximum=maximum, Ngrid=Ngrid, grid=grid, validate_args=validate_args)
        self._support = constraints.positive

    def _log_prob_nonorm(self, value):
        return (self.lamb - 1) * jnp.log(1 + value) + cosmo.logdVcdz(value)


class LinearInterpolatedPowerlaw(NumericallyNormalizedDistribition):
    arg_constraints = {
        "yinterps": constraints.real_vector,
        "xinterps": constraints.ordered_vector,
        "maximum": constraints.real,
        "minimum": constraints.real,
        "alpha": constraints.real,
    }
    reparametrized_params = ["yinterps", "xinterps", "maximum", "minimum", "alpha"]

    def __init__(self, xinterps, yinterps, minimum, maximum, alpha, Ngrid=1000, validate_args=None):
        self.alpha = alpha
        self.xinterps = xinterps
        self.yinterps = yinterps
        super(LinearInterpolatedPowerlaw, self).__init__(minimum, maximum, Ngrid=Ngrid, validate_args=validate_args)

    def _log_prob_nonorm(self, value):
        return self.alpha * jnp.log(value) + jnp.interp(value, self.xinterps, self.yinterps)


class CubicInterpolatedPowerlaw(LinearInterpolatedPowerlaw):
    def __init__(self, xinterps, yinterps, minimum, maximum, alpha, Ngrid=1000, validate_args=None):
        self.interpolator = NaturalCubicUnivariateSpline(xinterps, yinterps)
        super(CubicInterpolatedPowerlaw, self).__init__(xinterps, yinterps, minimum, maximum, alpha, Ngrid=Ngrid, validate_args=validate_args)

    def _log_prob_nonorm(self, value):
        return self.alpha * jnp.log(value) + self.interpolator(value)


class BSplineDistribution(Distribution):
    arg_constraints = {
        "maximum": constraints.real,
        "minimum": constraints.real,
        "cs": constraints.real_vector,
    }
    reparametrized_params = ["maximum", "minimum", "cs"]

    def __init__(self, minimum, maximum, cs, grid, grid_dmat, validate_args=None):
        self.maximum, self.minimum, self.cs = promote_shapes(maximum, minimum, cs)
        self._support = constraints.interval(minimum, maximum)
        batch_shape = lax.broadcast_shapes(jnp.shape(maximum), jnp.shape(minimum), jnp.shape(cs))
        super(BSplineDistribution, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.grid = grid
        self.lpdfs = jnp.einsum("i,i...->...", self.cs, grid_dmat)
        self.pdfs = jnp.exp(self.lpdfs)
        self.norm = jnp.trapz(self.pdfs, self.grid)
        self.pdfs /= self.norm
        self.cdfgrid = cumtrapz(self.pdfs, self.grid)
        self.cdfgrid = self.cdfgrid.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    def _log_prob_nonorm(self, value):
        return jnp.interp(value, self.grid, self.lpdfs)

    @validate_sample
    def log_prob(self, value):
        return self._log_prob_nonorm(value) - jnp.log(self.norm)

    def cdf(self, value):
        return jnp.interp(value, self.grid, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.grid)


class PSplineCoeficientPrior(Distribution):
    arg_constraints = {"inv_var": constraints.positive}
    reparametrized_params = ["inv_var"]

    def __init__(self, N, inv_var, diff_order=2, validate_args=None):
        (self.inv_var,) = promote_shapes(inv_var)
        self._support = constraints.real_vector
        batch_shape = lax.broadcast_shapes(jnp.shape(inv_var))
        super(PSplineCoeficientPrior, self).__init__(batch_shape=batch_shape, validate_args=validate_args, event_shape=(N,))
        self.diff_order = diff_order
        self.N = N

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jnp.ones(shape=sample_shape + self.batch_shape)

    @validate_sample
    def log_prob(self, value):
        assert value.shape == (self.N,)
        return apply_difference_prior(value, self.inv_var, self.diff_order)
