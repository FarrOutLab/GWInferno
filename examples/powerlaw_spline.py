#!/usr/bin/env python
from argparse import ArgumentParser

import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from jax import config
from jax import random
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.interpolation import LogXBSpline
from gwinferno.models.bsplines.smoothing import apply_difference_prior
from gwinferno.models.parametric.parametric import PowerlawRedshiftModel
from gwinferno.models.spline_perturbation import PowerlawBasisSplinePrimaryPowerlawRatio
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.postprocess.calculate_ppds import calculate_m1q_ppds_plbspline_model
from gwinferno.postprocess.calculate_ppds import calculate_powerlaw_rate_of_z_ppds
from gwinferno.postprocess.plotting import plot_mass_dist
from gwinferno.preprocess.data_collection import load_injection_dataset
from gwinferno.preprocess.data_collection import load_posterior_samples

config.update("jax_enable_x64", True)
az.style.use("arviz-darkgrid")


def setup_redshift_model(injdata, pedata, pmap):
    z_pe = pedata[pmap["redshift"]]
    z_inj = injdata[pmap["redshift"]]
    model = PowerlawRedshiftModel(z_pe, z_inj)
    return model


def setup(args):
    injections = load_injection_dataset(args.inj_file, spin=False)
    pe_samples, names = load_posterior_samples(args.data_dir, spin=False)
    param_names = [
        "mass_1",
        "mass_ratio",
        "redshift",
        "prior",
    ]
    param_map = {p: i for i, p in enumerate(param_names)}
    injdata = jnp.array([injections[k] for k in param_names], dtype=jnp.float64)
    pedata = jnp.array([[pe_samples[e][p] for e in names] for p in param_names], dtype=jnp.float64)
    nObs = pedata.shape[1]
    total_inj = injections["total_generated"]
    obs_time = injections["analysis_time"]

    z_model = setup_redshift_model(injdata, pedata, param_map)

    inj_dict = {k: injdata[param_map[k]] for k in param_names}
    pe_dict = {k: pedata[param_map[k]] for k in param_names}

    print(f"{len(inj_dict['redshift'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {pe_dict['redshift'].shape[1]} samples, over an observing time of {obs_time} yrs")
    return z_model, pe_dict, inj_dict, total_inj, nObs, obs_time


def model(mass_model, z_model, pe_dict, inj_dict, total_inj, Nobs, Tobs, args):
    beta = numpyro.sample("beta", dist.Normal(0, 3))
    alpha = numpyro.sample("alpha", dist.Normal(0, 3))
    mmax = numpyro.sample("mmax", dist.Uniform(60, args.mmax))
    mmin = args.mmin

    fs = numpyro.sample("mass_cs", dist.Normal(), sample_shape=(mass_model.nknots,))
    tau = numpyro.deterministic("mass_tau", 1)
    numpyro.factor("mass_log_smoothing_penalty", apply_difference_prior(fs, tau, degree=2))

    lamb = numpyro.sample("lamb", dist.Normal(0, 3))

    if not args.sample_prior:

        def get_weights(m1, q, z, prior):
            p_m1q = mass_model(m1, q, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, cs=fs)
            p_z = z_model(z, lamb)
            wts = p_m1q * p_z / prior
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

        peweights = get_weights(
            pe_dict["mass_1"],
            pe_dict["mass_ratio"],
            pe_dict["redshift"],
            pe_dict["prior"],
        )
        injweights = get_weights(
            inj_dict["mass_1"],
            inj_dict["mass_ratio"],
            inj_dict["redshift"],
            inj_dict["prior"],
        )

        hierarchical_likelihood(
            peweights,
            injweights,
            total_inj=total_inj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=z_model.normalization,
            vtfct_kwargs={"lamb": lamb},
            min_neff_cut=True,
            posterior_predictive_check=True,
            pedata=pe_dict,
            injdata=inj_dict,
            marginalize_selection=False,
            param_names=[
                "mass_1",
                "mass_ratio",
                "redshift",
            ],
        )


def load_parser():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/bruce.edelman/projects/GWTC3_allevents/")
    parser.add_argument(
        "--inj-file",
        type=str,
        default="/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5",
    )
    parser.add_argument("--outdir", type=str, default="test")
    parser.add_argument("--mmin", type=float, default=5.0)
    parser.add_argument("--m2min", type=float, default=3.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--mass-knots", type=int, default=48)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = load_parser()

    label = f"{args.outdir}/plbspline_{args.mass_knots}n_mspline_{args.chains}chains"
    RNG = random.PRNGKey(0)
    MCMC_RNG, PRIOR_RNG, RNG = random.split(RNG, num=3)
    z, pe, inj, total_inj, nObs, obs_time = setup(args)

    plspline = PowerlawBasisSplinePrimaryPowerlawRatio(
        args.mass_knots, pe["mass_1"], inj["mass_1"], mmin=args.mmin, m2min=args.m2min, mmax=args.mmax, basis=LogXBSpline, normalize=False
    )
    print("running mcmc: sampling prior...")
    args.sample_prior = True
    thinning = 1
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=args.warmup // 3,
        num_samples=args.samples // 3,
        num_chains=args.chains,
    )
    mcmc.run(PRIOR_RNG, plspline, z, pe, inj, float(total_inj), nObs, obs_time, args)
    prior = mcmc.get_samples()
    dd.io.save(f"{label}_prior_samples.h5", prior)

    print("running mcmc: sampling posterior...")
    args.sample_prior = False
    kernel = NUTS(model)
    plspline = PowerlawBasisSplinePrimaryPowerlawRatio(
        args.mass_knots, pe["mass_1"], inj["mass_1"], mmin=args.mmin, mmax=args.mmax, basis=LogXBSpline, normalize=False
    )
    mcmc = MCMC(
        kernel,
        thinning=thinning,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
    )
    mcmc.run(MCMC_RNG, plspline, z, pe, inj, float(total_inj), nObs, obs_time, args)
    mcmc.print_summary()

    print("plotting trace plot...")
    plot_params = [
        "alpha",
        "beta",
        "lamb",
        "log_nEff_inj",
        "log_nEffs",
        "log_l",
        "mass_cs",
        "rate",
    ]
    fig = az.plot_trace(az.from_numpyro(mcmc), var_names=plot_params)
    plt.savefig(f"{label}_trace_plot.png")
    del fig

    posterior = mcmc.get_samples()
    dd.io.save(f"{label}_posterior_samples.h5", posterior)
    del mcmc, pe, inj, total_inj, obs_time

    print("calculating mass posterior ppds...")
    pm1s, pqs, ms, qs = calculate_m1q_ppds_plbspline_model(
        posterior, PowerlawBasisSplinePrimaryPowerlawRatio, args.mass_knots, mmin=args.mmin, m2min=args.m2min, mmax=args.mmax
    )
    print("calculating mass prior ppds...")
    prior_pm1s, prior_pqs, _, _ = calculate_m1q_ppds_plbspline_model(
        prior, PowerlawBasisSplinePrimaryPowerlawRatio, args.mass_knots, mmin=args.mmin, m2min=args.m2min, mmax=args.mmax
    )

    print("calculating rate posterior ppds...")
    try:
        Rofz, zs = calculate_powerlaw_rate_of_z_ppds(posterior["lamb"], posterior["rate"], z)
    except KeyError:
        Rofz, zs = calculate_powerlaw_rate_of_z_ppds(2.7 * jnp.ones_like(posterior["rate"]), posterior["rate"], z)
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


if __name__ == "__main__":
    main()
