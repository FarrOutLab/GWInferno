import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from gwinferno.numpyro_distributions import Powerlaw
from gwinferno.numpyro_distributions import PowerlawRedshift
from gwinferno.pipeline.analysis import hierarchical_likelihood


def model(samps, injs, Ninj, Nobs, Tobs):
    alpha = numpyro.sample("alpha", dist.Normal(0, 3))
    beta = numpyro.sample("beta", dist.Normal(0, 3))
    mmin = numpyro.sample("mmin", dist.Uniform(2, 10))
    mmax = numpyro.sample("mmax", dist.Uniform(50, 100))
    lamb = numpyro.sample("lamb", dist.Normal(0, 3))

    m1dist = Powerlaw(alpha, minimum=mmin, maximum=mmax)
    qdist = Powerlaw(beta, minimum=0.02, maximum=1.0)
    zdist = PowerlawRedshift(lamb, maximum=2.3)

    inj_weights = m1dist.log_prob(injs["mass_1"]) + qdist.log_prob(injs["mass_ratio"]) + zdist.log_prob(injs["redshift"]) - jnp.log(injs["prior"])
    pe_weights = m1dist.log_prob(samps["mass_1"]) + qdist.log_prob(samps["mass_ratio"]) + zdist.log_prob(samps["redshift"]) - jnp.log(samps["prior"])

    def shvf():
        return zdist["redshift"].norm

    hierarchical_likelihood(
        pe_weights,
        inj_weights,
        total_inj=Ninj,
        Nobs=Nobs,
        Tobs=Tobs,
        surv_hypervolume_fct=shvf,
        vtfct_kwargs={},
        marginalize_selection=False,
        min_neff_cut=True,
        posterior_predictive_check=True,
        pedata=samps,
        injdata=injs,
        param_names=[
            "mass_1",
            "mass_ratio",
            "redshift",
        ],
        m1min=mmin,
        m2min=mmin,
        mmax=mmax,
    )
