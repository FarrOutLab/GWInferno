import jax.numpy as jnp
from jax import lax
from jax import random
from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import validate_sample


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

    def __init__(self, alpha, minimum=0.0, maximum=1.0, validate_args=None):
        self.minimum, self.maximum, self.alpha = promote_shapes(minimum, maximum, alpha)
        self._support = constraints.interval(minimum, maximum)
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
        return jnp.where(jnp.equal(self.alpha, -1.0), logp_neg1, logp)

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
