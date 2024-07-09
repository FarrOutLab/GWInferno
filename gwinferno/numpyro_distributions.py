"""
a module defining additional numpyro distributions
"""

import jax.numpy as jnp
from jax import lax
from jax import random
from jax import vmap
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import validate_sample

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
        lp = jnp.log(jnp.sin(value) / 2.0)
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

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
        lp = jnp.log(jnp.cos(value) / 2.0)
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

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

    def __init__(self, alpha, minimum=0.0, maximum=1.0, low=0.0, high=1.0, validate_args=None):
        self.minimum, self.maximum, self.alpha = promote_shapes(minimum, maximum, alpha)
        self._support = constraints.interval(low, high)
        batch_shape = broadcast_shapes(
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


class PowerlawRedshift(Distribution):
    arg_constraints = {
        "maximum": constraints.positive,
        "lamb": constraints.real,
    }
    reparametrized_params = ["maximum", "lamb"]

    def __init__(self, lamb, maximum, zgrid, dVcdz, low=0.0, high=1000.0, validate_args=None):
        self.maximum, self.lamb = promote_shapes(maximum, lamb)
        self._support = constraints.interval(low, high)
        batch_shape = broadcast_shapes(
            jnp.shape(maximum),
            jnp.shape(lamb),
        )
        super(PowerlawRedshift, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.zs = zgrid
        self.dVdc_ = dVcdz
        self.pdfs = self.dVdc_ * (1 + self.zs) ** (lamb - 1)
        self.norm = trapezoid(self.pdfs, self.zs)
        self.pdfs /= self.norm
        self.cdfgrid = cumtrapz(self.pdfs, self.zs)
        self.cdfgrid = self.cdfgrid.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value, dVdc=None):
        if dVdc is None:
            dVdc = jnp.interp(value, self.zs, self.dVdc_)
        return jnp.where(
            jnp.less_equal(value, self.maximum),
            jnp.log(dVdc) + (self.lamb - 1.0) * jnp.log(1.0 + value) - jnp.log(self.norm),
            jnp.nan_to_num(-jnp.inf),
        )

    def cdf(self, value):
        return jnp.interp(value, self.zs, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.zs)


class PowerlawSmoothedPowerlaw(Distribution):
    arg_constraints = {
        "minimum": constraints.positive,
        "maximum": constraints.positive,
        "alpha": constraints.real,
        "alpha_max": constraints.positive,
        "alpha_min": constraints.positive,
    }
    reparametrized_params = ["minimum", "maximum", "alpha", "alpha_max", "alpha_min"]

    def __init__(self, alpha, minimum, maximum, alpha_max, alpha_min, low, high, validate_args=None):
        self.minimum, self.maximum, self.alpha, self.alpha_max, self.alpha_min = promote_shapes(minimum, maximum, alpha, alpha_max, alpha_min)
        self.alpha_max = -self.alpha_max
        self._support = constraints.interval(low, high)
        self.low, self.high = low, high
        batch_shape = broadcast_shapes(jnp.shape(maximum), jnp.shape(minimum), jnp.shape(alpha), jnp.shape(alpha_max), jnp.shape(alpha_min))
        super(PowerlawSmoothedPowerlaw, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        gamma = (self.alpha_min + 1) / (self.minimum ** (self.alpha_min + 1) - self.low ** (self.alpha_min + 1))
        self.k1 = -gamma / (
            1
            + gamma
            / (self.alpha + 1)
            * self.minimum ** (self.alpha_min - self.alpha)
            * (self.minimum ** (self.alpha + 1) - self.maximum ** (self.alpha + 1))
            + gamma
            / (self.alpha_max + 1)
            * self.minimum ** (self.alpha_min - self.alpha)
            * self.maximum ** (self.alpha - self.alpha_max)
            * (self.maximum ** (self.alpha_max + 1) - self.high ** (self.alpha_max + 1))
        )
        self.k2 = self.k1 * self.minimum ** (self.alpha_min - self.alpha)
        self.k3 = self.k2 * self.maximum ** (self.alpha - self.alpha_max)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape
        return jnp.ones(shape)

    @validate_sample
    def log_prob(self, value):
        low_pl = jnp.where(jnp.less(value, self.minimum), jnp.log(self.k1) + jnp.log(value) * self.alpha_min, 0.0)
        high_pl = jnp.where(jnp.greater(value, self.maximum), jnp.log(self.k3) + jnp.log(value) * self.alpha_max, 0.0)
        mid_pl = jnp.where(
            jnp.greater_equal(value, self.minimum),
            jnp.where(jnp.less_equal(value, self.maximum), jnp.log(self.k2) + jnp.log(value) * self.alpha, 0.0),
            0.0,
        )
        return low_pl + mid_pl + high_pl


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
        # grid_dmat will contain nan's where the grid is outside the support
        self.lpdfs = jnp.nan_to_num(jnp.einsum("i,i...->...", self.cs, grid_dmat), nan=-jnp.inf)
        self.pdfs = jnp.exp(self.lpdfs)
        self.norm = trapezoid(self.pdfs, self.grid)
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
