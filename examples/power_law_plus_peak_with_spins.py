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
from jax.scipy.special import logit
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
    norm_factor = 1.- norm.cdf(low,loc=mu,scale=sigma) + (1.-norm.cdf(high,loc=mu,scale=sigma))
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

def logit_normal(xx,mu,sigma,low=0.,high=1.):
    xx_new = (xx-low)/(high-low)
    prob = (1./(sigma*jnp.sqrt(2.*jnp.pi))) * jnp.exp((-1./2.)*jnp.power((logit(xx_new)-mu)/sigma,2)) / ((xx_new*(1.-xx_new)))
    return jnp.where(jnp.less(xx_new,0) | jnp.greater(xx_new,1),0.0,prob/(high-low))

def PL_peak_w_spins(m1,q,a1,cost1,a2,cost2,alpha,beta,mmin,mmax,mu_m1,sigma_m1,gamma_m1,mu_a1,sigma_a1,mu_a2,sigma_a2,mu_cost1,sigma_cost1,mu_cost2,sigma_cost2):
    pm1q = power_law_plus_peak_full(m1, q, alpha, beta, mmin, mmax, mu_m1,sigma_m1,gamma_m1) 
    pa1 = logit_normal(a1,mu_a1,sigma_a1)
    pa2 = logit_normal(a2,mu_a2,sigma_a2)
    pcost1 = logit_normal(cost1,mu_cost1,sigma_cost1,low=-1., high=1.)
    pcost2 = logit_normal(cost2,mu_cost2,sigma_cost2,low=-1., high=1.)
    return pm1q*pa1*pa2*pcost1*pcost2 

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
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    return parser.parse_args()


def setup_redshift_model(injdata, pedata, pmap):
    z_pe = pedata[pmap["redshift"]]
    z_inj = injdata[pmap["redshift"]]
    model = PowerlawRedshiftModel(z_pe, z_inj)
    return model


def setup(args):
    injections = load_injections(args.inj_file, spin=True)
    pe_samples, names = load_posterior_samples(args.data_dir, spin=True)
    param_names = [
        "mass_1",
        "mass_ratio",
        "redshift",
        "a_1",
        "a_2",
        "cos_tilt_1",
        "cos_tilt_2",
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
    gamma_m1 = numpyro.sample("gamma_m1",dist.Uniform(0,1))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(mmin,mmax))
    sigma_m1 = numpyro.sample("sigma_m1",dist.Uniform(3,20))
    mu_a1 = numpyro.sample("mu_a1",dist.Uniform(-2,2))
    mu_a2 = numpyro.sample("mu_a2",dist.Uniform(-2,2))
    sigma_a1 = numpyro.sample("sigma_a1",dist.Uniform(0.3,3))
    sigma_a2 = numpyro.sample("sigma_a2",dist.Uniform(0.3,3))
    mu_cost1 = numpyro.sample("mu_cost1",dist.Uniform(-2,2))
    mu_cost2 = numpyro.sample("mu_cost2",dist.Uniform(-2,2))
    sigma_cost1 = numpyro.sample("sigma_cost1",dist.Uniform(0.3,3))
    sigma_cost2 = numpyro.sample("sigma_cost2",dist.Uniform(0.3,3))
    
    if not sample_prior:

        def get_weights(m1, q, z, a1, cost1, a2, cost2, prior):
            p_m1qa = PL_peak_w_spins(m1,q,a1,cost1,a2,cost2,alpha,beta,mmin,mmax,mu_m1,sigma_m1,gamma_m1,mu_a1,sigma_a1,mu_a2,sigma_a2,mu_cost1,sigma_cost1,mu_cost2,sigma_cost2)
            #p_m1q = power_law_plus_peak_full(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, mu=mu,sigma=sigma,gamma=gamma)
            #p_m1q = truncated(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
            p_z = z_model(z, lamb)
            wts = p_m1qa * p_z / prior
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

        peweights = get_weights(pedict["mass_1"], pedict["mass_ratio"], pedict["redshift"], pedict["a_1"], pedict["cos_tilt_1"], pedict["a_2"], pedict["cos_tilt_2"],pedict["prior"])
        injweights = get_weights(injdict["mass_1"], injdict["mass_ratio"], injdict["redshift"], injdict["a_1"], injdict["cos_tilt_1"], injdict["a_2"], injdict["cos_tilt_2"],injdict["prior"])
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
                "a_1",
                "cos_tilt_1",
                "a_2",
                "cos_tilt_2",
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

def calculate_spin_ppds(mu_a1,sigma_a1,mu_a2,sigma_a2,mu_cost1,sigma_cost1,mu_cost2,sigma_cost2,rate=None):
    a_grid = np.linspace(0,1,600)
    cost_grid = np.linspace(-1,1,600)
    a1_pdfs = np.zeros((mu_a1.shape[0],len(a_grid)))
    a2_pdfs = np.zeros((mu_a2.shape[0],len(a_grid)))
    cost1_pdfs = np.zeros((mu_cost1.shape[0],len(cost_grid)))
    cost2_pdfs = np.zeros((mu_cost2.shape[0],len(cost_grid)))
    if rate is None:
        rate = jnp.ones(mu_a1.shape[0])

    def calc_pdf(ma1,sa1,ma2,sa2,mc1,sc1,mc2,sc2,r):
        p_a1 = logit_normal(a_grid,ma1,sa1)
        p_a2 = logit_normal(a_grid,ma2,sa2)
        p_cost1 = logit_normal(cost_grid,mc1,sc1,low=0.,high=1.)
        p_cost2 = logit_normal(cost_grid,mc2,sc2,low=0.,high=1.)
        return r*p_a1, r*p_a2, r*p_cost1, r*p_cost2

    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(mu_a1[0],sigma_a1[0],mu_a2[0],sigma_a2[0],mu_cost1[0],sigma_cost1[0],mu_cost2[0],sigma_cost2[0],rate[0])
    # loop through hyperposterior samples
    for ii in trange(mu_a1.shape[0]):
        a1_pdfs[ii], a2_pdfs[ii], cost1_pdfs[ii], cost2_pdfs[ii] = calc_pdf(mu_a1[ii],sigma_a1[ii],mu_a2[ii],sigma_a2[ii],mu_cost1[ii],sigma_cost1[ii],mu_cost2[ii],sigma_cost2[ii],rate[ii])
    return a1_pdfs, a2_pdfs, cost1_pdfs, cost2_pdfs, a_grid, cost_grid

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
            "mu_a1",
            "sigma_a1",
            "mu_cost1",
            "sigma_cost1",
            "mu_a2",
            "sigma_a2",
            "mu_cost2",
            "sigma_cost2"
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
   
    print("calculating spin prior ppds")
    a1_prior, a2_prior, cost1_prior, cost2_prior, a_grid, cost_grid = calculate_spin_ppds(prior["mu_a1"],prior["sigma_a1"],prior["mu_a2"],prior["sigma_a2"],prior["mu_cost1"],prior["sigma_cost1"],prior["mu_cost2"],prior["sigma_cost2"])
    print("calculating spin posterior ppds")
    a1_posts, a2_posts, cost1_posts, cost2_posts, a_grid, cost_grid = calculate_spin_ppds(posterior["mu_a1"],posterior["sigma_a1"],posterior["mu_a2"],posterior["sigma_a2"],posterior["mu_cost1"],posterior["sigma_cost1"],posterior["mu_cost2"],posterior["sigma_cost2"])

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
        "dRda1" : a1_posts,
        "dRda2" : a2_posts,
        "as":a_grid,
        "dRdcost1" : cost1_posts,
        "dRdcost2" : cost2_posts,
        "costs" : cost_grid,
    }
    dd.io.save(f"{label}_ppds.h5", ppd_dict)
    prior_ppd_dict = {
        "pm1": prior_pm1s,
        "pq": prior_pqs,
        "m1s": ms,
        "qs": qs,
        "Rofz": prior_Rofz,
        "zs": zs,
        "dRda1" : a1_prior,
        "dRda2" : a2_prior,
        "as":a_grid,
        "dRdcost1" : cost1_prior,
        "dRdcost2" : cost2_prior,
        "costs" : cost_grid,       
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
    fig = plot_spin_dist(a1_posts,a2_posts,cost1_posts,cost2_posts,a_grid,cost_grid)
    plt.savefig(f"spin_ppds.png")
    del fig

    
def plot_spin_dist(pa1,pa2,pcost1,pcost2,a_grid,cost_grid):
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    a1_median, a1_lows, a1_highs, a2_median, a2_lows, a2_high= np.median(pa1, axis=0), np.percentile(pa1,5,axis=0), np.percentile(pa1,95,axis=0), np.median(pa2, axis=0),  np.percentile(pa2,5,axis=0), np.percentile(pa2,95,axis=0)
    cost1_median, cost1_lows, cost1_highs, cost2_median, cost2_lows, cost2_high= np.median(pcost1,axis=0), np.percentile(pcost1,5,axis=0), np.percentile(pcost1,95,axis=0), np.median(pcost2,axis=0), np.percentile(pcost2,5,axis=0), np.percentile(pcost2,95,axis=0)

    ax[0].fill_between(a_grid,a1_lows,a1_highs,alpha=0.4,color='green')
    ax[0].fill_between(a_grid,a2_lows,a2_highs,alpha=0.4,color='blue')
    ax[0].plot(a_grid,a1_median,color='green',lw=3,label='1')
    ax[0].plot(a_grid,a2_median,color='blue',lw=3,label='2')
    ax[0].set_xlabel('spin')
    ax[0].legend()

    ax[1].fill_between(cost_grid,cost1_lows,cost1_highs,alpha=0.4,color='green')
    ax[1].fill_between(cost_grid,cost2_lows,cost2_highs,alpha=0.4,color='blue')
    ax[1].plot(cost_grid,cost1_median,color='green',lw=3,label='1')
    ax[1].plot(cost_grid,cost2_median,color='blue',lw=3,label='2')
    ax[1].set_xlabel('cos tilt')
    ax[1].legend()
    return fig

if __name__ == "__main__":
    main()
