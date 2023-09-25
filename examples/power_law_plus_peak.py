from argparse import ArgumentParser

import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from jax import jit
from jax import random
from jax.scipy.stats import norm
from jax.scipy.special import erf
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from tqdm import trange

from gwinferno.models.gwpopulation.gwpopulation import PowerlawRedshiftModel
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.postprocess.plotting import plot_mass_dist
from gwinferno.postprocess.plotting import plot_rofz
from gwinferno.preprocess.data_collection import load_injections
from gwinferno.preprocess.data_collection import load_posterior_samples

az.style.use("arviz-darkgrid")


def truncated_powerlaw(xx, alpha, low, high):
    prob = jnp.power(xx, alpha)
    prob_normed = prob*(alpha+1.)/(jnp.power(high,alpha+1.) - jnp.power(low,alpha+1.))
    return jnp.where(jnp.less(xx, low) | jnp.greater(xx, high), 0.0, prob_normed)

def truncated_normal(xx,mu,sigma,low,high):
    prob = (1./(sigma*jnp.sqrt(2.*jnp.pi))) * jnp.exp((-1./2.)*jnp.power((xx-mu)/sigma,2))
    norm_factor = norm.cdf(high,loc=mu,scale=sigma) -  norm.cdf(low,loc=mu,scale=sigma)
    return jnp.where(jnp.less(xx, low) | jnp.greater(xx, high), 0.0, prob/norm_factor)

def truncated_powerlaw_plus_peak(xx,alpha, mmin, mmax, mu,sigma,gamma):
    return gamma*truncated_powerlaw(xx, alpha=-alpha, low=mmin, high=mmax) + (1.-gamma)*truncated_normal(xx,mu,sigma,low=mmin,high=mmax)

def power_law_plus_peak_full(m1, q, alpha, beta, mmin, mmax, mu,sigma,gamma):
    p_m1 = truncated_powerlaw_plus_peak(m1,alpha, mmin, mmax, mu, sigma, gamma) 
    p_q = truncated_powerlaw(q, alpha=beta, low=mmin / m1, high=1)
    return p_m1 * p_q / norm_mass_model(alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, mu=mu, sigma=sigma, gamma=gamma)

def norm_mass_model(alpha, beta, mmin, mmax, mu,sigma,gamma):
    ms = jnp.linspace(3, 100, 750)
    qs = jnp.linspace(0.01, 1, 500)
    mm, qq = jnp.meshgrid(ms, qs)
    p_mq = truncated_powerlaw_plus_peak(ms, alpha, mmin, mmax, mu,sigma,gamma) * truncated_powerlaw(qq, alpha=beta, low=mmin / mm, high=1)
    return jnp.trapz(jnp.trapz(p_mq, qs, axis=0), ms)


def load_parser():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/bruce.edelman/projects/GWTC3_allevents/")
    parser.add_argument(
        "--inj-file",
        type=str,
        default="/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5",
    )
    parser.add_argument("--outdir", type=str, default="test")
    parser.add_argument("--mmin", type=float, default=4.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    return parser.parse_args()


def setup_redshift_model(injdata, pedata, pmap):
    z_pe = pedata[pmap["redshift"]]
    z_inj = injdata[pmap["redshift"]]
    model = PowerlawRedshiftModel(z_pe, z_inj)
    return model


def setup(args):
    injections = load_injections(args.inj_file, spin=False)
    pe_samples, names = load_posterior_samples(args.data_dir, spin=False)
    param_names = [
        "mass_1",
        "mass_ratio",
        "redshift",
        "prior",
    ]
    param_map = {p: i for i, p in enumerate(param_names)}
    injdata = jnp.array([injections[k] for k in param_names])
    pedata = jnp.array([[pe_samples[e][p] for e in names] for p in param_names])
    nObs = pedata.shape[1]
    total_inj = injections["total_generated"]
    obs_time = injections["analysis_time"]

    z_model = setup_redshift_model(injdata, pedata, param_map)
    injdict = {k: injdata[param_map[k]] for k in param_names}
    pedict = {k: pedata[param_map[k]] for k in param_names}

    print(f"{len(injdict['redshift'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {pedict['redshift'].shape[1]} samples, over an observing time of {obs_time} yrs")

    return (
        z_model,
        pedict,
        injdict,
        total_inj,
        nObs,
        obs_time,
    )


def model(
    z_model,
    pedict,
    injdict,
    total_inj,
    Nobs,
    Tobs,
    sample_prior=False,
):
    alpha = numpyro.sample("alpha", dist.Normal(0, 3))
    beta = numpyro.sample("beta", dist.Normal(0, 3))
    mmin = 5.0  # numpyro.sample("mmin", dist.Uniform(4, 9))
    mmax = 85.0  # numpyro.sample("mmax", dist.Uniform(50, 100))
    lamb = numpyro.sample("lamb", dist.Normal(0, 3))
    gamma = numpyro.sample("gamma",dist.Uniform(0,1))
    mu = numpyro.sample("mu",dist.Uniform(mmin,mmax))
    sigma = numpyro.sample("sigma",dist.LogUniform(0.01,1.5))

    if not sample_prior:

        def get_weights(m1, q, z, prior):
            p_m1q = power_law_plus_peak_full(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, mu=mu,sigma=sigma,gamma=gamma)
            #p_m1q = truncated(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
            p_z = z_model(z, lamb)
            wts = p_m1q * p_z / prior
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

        peweights = get_weights(pedict["mass_1"], pedict["mass_ratio"], pedict["redshift"], pedict["prior"])
        injweights = get_weights(injdict["mass_1"], injdict["mass_ratio"], injdict["redshift"], injdict["prior"])
        hierarchical_likelihood(
            peweights,
            injweights,
            total_inj=total_inj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=z_model.normalization,
            vtfct_kwargs=dict(lamb=lamb),
            marginalize_selection=False,
            min_neff_cut=True,
            posterior_predictive_check=True,
            pedata=pedict,
            injdata=injdict,
            param_names=[
                "mass_1",
                "mass_ratio",
                "redshift",
            ],
            m1min=mmin,
            m2min=mmin,
            mmax=mmax,
        )


def calculate_m1q(alpha, beta, mmin, mmax, mu, sigma, gamma, rate=None):
    ms = np.linspace(3.0, 100, 800)
    qs = np.linspace(3.0 / 100, 1, 600)
    mm, qq = np.meshgrid(ms, qs)

    mpdfs = np.zeros((alpha.shape[0], len(ms)))
    qpdfs = np.zeros((beta.shape[0], len(qs)))
    if rate is None:
        rate = jnp.ones(beta.shape[0])

    def calc_pdf(a, b, mi, ma, mu_x,sigma_x,gamma_x,r):
        p_mq = power_law_plus_peak_full(mm, qq, alpha=a, beta=b, mmin=mi, mmax=ma, mu=mu_x, sigma=sigma_x, gamma=gamma_x)
        p_mq = jnp.where(jnp.less(mm, mmin) | jnp.less(mm * qq, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return r * p_m / jnp.trapz(p_m, ms), r * p_q / jnp.trapz(p_q, qs)

    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(alpha[0], beta[0], mmin, mmax, mu[0], sigma[0], gamma[0], rate[0])
    # loop through hyperposterior samples
    for ii in trange(alpha.shape[0]):
        mpdfs[ii], qpdfs[ii] = calc_pdf(alpha[ii], beta[ii], mmin, mmax, mu[ii], sigma[ii], gamma[ii], rate[ii])
    return mpdfs, qpdfs, ms, qs


def calculate_rate_of_z_ppds(lamb, rate, model):
    zs = model.zs
    rs = np.zeros((len(lamb), len(zs)))

    def calc_rz(la, r):
        return r * jnp.power(1.0 + zs, la)

    calc_rz = jit(calc_rz)
    _ = calc_rz(lamb[0], rate[0])
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(lamb[ii], rate[ii])
    return rs, zs


def main():
    args = load_parser()
    label = f"{args.outdir}/simple_example"
    z, pedict, injdict, total_inj, nObs, obs_time = setup(args)
    if not args.skip_inference:
        RNG = random.PRNGKey(0)
        MCMC_RNG, PRIOR_RNG, _RNG = random.split(RNG, num=3)
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            thinning=args.thinning,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
        )
        print("running mcmc: sampling prior...")
        mcmc.run(
            PRIOR_RNG,
            z,
            pedict,
            injdict,
            float(total_inj),
            nObs,
            obs_time,
            sample_prior=True,
        )
        prior = mcmc.get_samples()
        dd.io.save(f"{label}_prior_samples.h5", prior)

        kernel = NUTS(model, dense_mass=True)
        mcmc = MCMC(
            kernel,
            thinning=args.thinning,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
        )
        print("running mcmc: sampling posterior...")
        mcmc.run(
            MCMC_RNG,
            z,
            pedict,
            injdict,
            float(total_inj),
            nObs,
            obs_time,
            sample_prior=False,
        )
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        dd.io.save(f"{label}_posterior_samples.h5", posterior)
        plot_params = [
            "alpha",
            "beta",
            "mu",
            "sigma",
            "gamma",
            "detection_efficency",
            "lamb",
            "log_nEff_inj",
            "log_nEffs",
            "logBFs",
            "log_l",
            "rate",
            "surveyed_hypervolume",
        ]
        fig = az.plot_trace(az.from_numpyro(mcmc), var_names=plot_params)
        plt.savefig(f"{label}_trace_plot.png")
        del fig, mcmc, pedict, injdict, total_inj, obs_time
    else:
        print(f"loading prior and posterior samples from run with label: {label}...")
        prior = dd.io.load(f"{label}_prior_samples.h5")
        posterior = dd.io.load(f"{label}_posterior_samples.h5")

    print("calculating mass prior ppds...")
    prior_pm1s, prior_pqs, ms, qs = calculate_m1q(
        prior["alpha"],
        prior["beta"],
        mmin=5.0,  # prior["mmin"],
        mmax=85.0,  # prior["mmax"],
        mu=prior["mu"],
        sigma=prior["sigma"],
        gamma=prior["gamma"],
    )
    print("calculating mass posterior ppds...")
    pm1s, pqs, ms, qs = calculate_m1q(
        posterior["alpha"],
        posterior["beta"],
        mmin=5.0,  # posterior['mmin'],
        mmax=85.0,  # posterior['mmax'],
        mu=posterior["mu"],
        sigma=posterior["sigma"],
        gamma=posterior["gamma"],
    )
    print("calculating rate prior ppds...")
    prior_Rofz, zs = calculate_rate_of_z_ppds(prior["lamb"], jnp.ones_like(prior["lamb"]), z)
    print("calculating rate posterior ppds...")
    Rofz, zs = calculate_rate_of_z_ppds(posterior["lamb"], posterior["rate"], z)

    ppd_dict = {
        "dRdm1": pm1s,
        "dRdq": pqs,
        "m1s": ms,
        "qs": qs,
        "Rofz": Rofz,
        "zs": zs,
    }
    dd.io.save(f"{label}_ppds.h5", ppd_dict)
    prior_ppd_dict = {
        "pm1": prior_pm1s,
        "pq": prior_pqs,
        "m1s": ms,
        "qs": qs,
        "Rofz": prior_Rofz,
        "zs": zs,
    }
    dd.io.save(f"{label}_prior_ppds.h5", prior_ppd_dict)
    del ppd_dict, prior_ppd_dict

    print("plotting mass distribution...")
    fig = plot_mass_dist(
        pm1s,
        pqs,
        ms,
        qs,
        mmin=args.mmin,
        mmax=args.mmax,
        priors={"m1": prior_pm1s, "q": prior_pqs},
    )
    plt.savefig(f"{label}_mass_distribution.png")
    del fig

    print("plotting R(z)...")
    fig = plot_rofz(Rofz, zs, prior=prior_Rofz)
    plt.savefig(f"{label}_rate_vs_z.png")
    del fig
    fig = plot_rofz(Rofz, zs, logx=True, prior=prior_Rofz)
    plt.savefig(f"{label}_rate_vs_z_logscale.png")
    del fig


if __name__ == "__main__":
    main()
