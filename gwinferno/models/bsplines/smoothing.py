import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist


def apply_difference_prior(coefs, inv_var, degree=1):
    D = jnp.diff(jnp.identity(len(coefs)), n=degree)
    delta_c = jnp.dot(coefs, D)
    for i in range(degree):
        old = delta_c.at[i].get()
        delta_c.at[i].set(2 * (degree - i) * old)
        idx = -1 - i
        old = delta_c.at[idx].get()
        delta_c.at[idx].set(2 * (degree - i) * old)
    return -0.5 * inv_var * jnp.dot(delta_c, delta_c.T)


def apply_twod_difference_prior(coefs, inv_var, degree=1):
    D = jnp.diff(jnp.eye(len(coefs)), n=degree)
    delta_c = jnp.dot(coefs, D)
    return -0.5 * inv_var * jnp.sum(jnp.dot(delta_c, delta_c.T).flatten())


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
