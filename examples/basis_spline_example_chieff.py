import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from jax import random
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.bsplines.single import BSplineChiEffective
from gwinferno.models.bsplines.smoothing import apply_difference_prior
from gwinferno.models.gwpopulation.gwpopulation import PowerlawRedshiftModel
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.parser import load_base_parser
from gwinferno.postprocess.calculate_ppds import calculate_m1q_bspline_ppds
from gwinferno.postprocess.calculate_ppds import calculate_powerlaw_rate_of_z_ppds
from gwinferno.postprocess.calculate_ppds import calculate_chieff_bspline_ppds
from gwinferno.postprocess.plotting import plot_m1_vs_z_ppc
from gwinferno.postprocess.plotting import plot_mass_dist
from gwinferno.postprocess.plotting import plot_rofz
from gwinferno.postprocess.plotting import plot_chieff_dist
from gwinferno.preprocess.data_collection import load_injections
from gwinferno.preprocess.data_collection import load_posterior_samples

az.style.use("arviz-darkgrid")


def load_parser():
    parser = load_base_parser()
    parser.add_argument("--mass-knots", type=int, default=100)
    parser.add_argument("--mag-knots", type=int, default=30)
    parser.add_argument("--q-knots", type=int, default=30)
    parser.add_argument("--tilt-knots", type=int, default=25)
    parser.add_argument("--z-knots", type=int, default=20)
    parser.add_argument("--chieff-nsplines", type=int, default=30)
    parser.add_argument("--skip-prior", action="store_true", default=True)
    return parser.parse_args()


def setup_mass_BSpline_model(injdata, pedata, pmap, nknots, qknots, mmin=3.0, mmax=100.0):
    print(f"Basis Spline model in m1 w/ {nknots} number of bases. Knots are logspaced from {mmin} to {mmax}...")
    print(f"Basis Spline model in q w/ {qknots} number of bases. Knots are linspaced from {mmin/mmax} to 1...")

    model = BSplinePrimaryBSplineRatio(
        nknots,
        qknots,
        pedata[pmap["mass_1"]],
        injdata[pmap["mass_1"]],
        pedata[pmap["mass_ratio"]],
        injdata[pmap["mass_ratio"]],
        m1min=mmin,
        m2min=mmin,
        mmax=mmax,
        basis_m=LogXLogYBSpline,
        basis_q=LogYBSpline,
    )
    return model

def setup_chieff_BSpline_model(nsplines, injdata, pedata, pmap):
    print(f"Basis spline model in chieff w/ {nsplines} bases. Knots are linearly spaced.")
    model = BSplineChiEffective(
        n_splines=nsplines,
        chieff=pedata[pmap['chi_eff']],
        chieff_inj=injdata[pmap['chi_eff']],
        basis=LogYBSpline,
    )
    return model


def setup_redshift_model(injdata, pedata, pmap):
    print(f"Powerlaw redshift model set up.")
    z_pe = pedata[pmap["redshift"]]
    z_inj = injdata[pmap["redshift"]]
    model = PowerlawRedshiftModel(z_pe, z_inj)
    return model


def setup(args):
    df = dd.io.load("./saved-pe-and-injs/posterior_samples_and_injections_chi_effective.h5")
    pedata = df['pedata']
    injdata = df['injdata']
    param_map = df['param_map']
    param_names = [
        "mass_1", "mass_ratio", "redshift", "chi_eff", "prior"
    ]
    param_map = {p: i for i, p in enumerate(param_names)}
    injdict = {k: injdata[param_map[k]] for k in param_names}
    pedict = {k: pedata[param_map[k]] for k in param_names}
    nObs = pedata.shape[1]
    total_inj = df["total_generated"]
    obs_time = df["analysis_time"]

    mass_model = setup_mass_BSpline_model(
        injdata,
        pedata,
        param_map,
        args.mass_knots,
        args.q_knots,
        mmin=args.mmin,
        mmax=args.mmax,
    )
    z_model = setup_redshift_model(injdata, pedata, param_map)
    chieff_model = setup_chieff_BSpline_model(args.chieff_nsplines, injdata, pedata, param_map)
    injdict = {k: injdata[param_map[k]] for k in param_names}
    pedict = {k: pedata[param_map[k]] for k in param_names}

    print(f"{len(injdict['redshift'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {pedict['redshift'].shape[1]} samples, over an observing time of {obs_time} yrs")

    return (
        mass_model,
        chieff_model,
        z_model,
        pedict,
        injdict,
        total_inj,
        nObs,
        obs_time,
    )


def model(
    mass_model,
    chieff_model,
    z_model,
    pedict,
    injdict,
    total_inj,
    Nobs,
    Tobs,
    sample_prior=False,
):
    mass_knots = mass_model.primary_model.n_splines
    q_knots = mass_model.ratio_model.n_splines
    chieff_nsplines = chieff_model.n_splines

    mass_cs = numpyro.sample("mass_cs", dist.Normal(0, 6), sample_shape=(mass_knots,))
    mass_tau_squared = numpyro.sample("mass_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.01), low = 0, high = 1))
    mass_lambda = numpyro.deterministic("mass_lambda", 1/mass_tau_squared)
    numpyro.factor("mass_log_smoothing_prior", apply_difference_prior(mass_cs, mass_lambda, degree=2))

    q_cs = numpyro.sample("q_cs", dist.Normal(0, 4), sample_shape=(q_knots,))
    q_tau_squared = numpyro.sample("q_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.1), low = 0, high = 1))
    q_lambda = numpyro.deterministic("q_lambda", 1/q_tau_squared)
    numpyro.factor("q_log_smoothing_prior", apply_difference_prior(q_cs, q_lambda, degree=2))

    chieff_cs = numpyro.sample("chieff_cs", dist.Normal(0,4), sample_shape=(chieff_nsplines,))
    chieff_tau_squared = numpyro.sample("chieff_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.1), low = 0, high = 1))
    chieff_lambda = numpyro.deterministic("chieff_lambda", 1/chieff_tau_squared)
    numpyro.factor("chieff_log_smoothing_prior", apply_difference_prior(chieff_cs, chieff_lambda, degree=2))

    lamb = numpyro.sample("lamb", dist.Normal(0, 3))

    if not sample_prior:

        def get_weights(z, prior, pe_samples = True):
            p_m1q = mass_model(mass_cs, q_cs, pe_samples)
            p_chieff = chieff_model(chieff_cs, pe_samples)
            p_z = z_model(z, lamb)
            wts = p_m1q * p_chieff * p_z / prior
            
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

        peweights = get_weights(pedict["redshift"], pedict["prior"])
        injweights = get_weights(injdict["redshift"], injdict["prior"], pe_samples=False)
        hierarchical_likelihood(
            peweights,
            injweights,
            total_inj=total_inj,
            Nobs=Nobs,
            Tobs=Tobs,
            vtfct_kwargs=dict(lamb=lamb),
            marginalize_selection=False,
            min_neff_cut=False,
            posterior_predictive_check=True,
            pedata=pedict,
            injdata=injdict,
            param_names=[
                "mass_1",
                "mass_ratio",
                "redshift",
                "chi_eff",
            ],
        )


def main():
    args = load_parser()
    label = f"{args.outdir}/bsplines_{args.chieff_nsplines}chieff_{args.mass_knots}m1_{args.q_knots}q_z"
    mass, chieff, z, pedict, injdict, total_inj, nObs, obs_time = setup(args)
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
            mass,
            chieff,
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

        kernel = NUTS(model)
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
            mass,
            chieff,
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
            "detection_efficency",
            "lamb",
            "log_nEff_inj",
            "log_nEffs",
            "logBFs",
            "log_l",
            "chieff_cs",
            "chieff_lambda",
            "mass_cs",
            "mass_lambda",
            "q_cs",
            "q_lambda",
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
    prior_pm1s, prior_pqs, ms, qs = calculate_m1q_bspline_ppds(
        prior["mass_cs"],
        prior["q_cs"],
        BSplinePrimaryBSplineRatio,
        args.mass_knots,
        args.q_knots,
        mmin=args.mmin,
        m1mmin=args.mmin,
        mmax=args.mmax,
        basis_m=LogXLogYBSpline,
        basis_q=LogYBSpline,
    )
    print("calculating mass posterior ppds...")
    pm1s, pqs, ms, qs = calculate_m1q_bspline_ppds(
        posterior["mass_cs"],
        posterior["q_cs"],
        BSplinePrimaryBSplineRatio,
        args.mass_knots,
        args.q_knots,
        mmin=args.mmin,
        m1mmin=args.mmin,
        mmax=args.mmax,
        basis_m=LogXLogYBSpline,
        basis_q=LogYBSpline,
    )

    if not args.skip_prior:
        print("calculating rate prior ppds...")
        prior_Rofz, zs = calculate_powerlaw_rate_of_z_ppds(prior["lamb"], jnp.ones_like(prior["lamb"]), z)
    print("calculating rate posterior ppds...")
    Rofz, zs = calculate_powerlaw_rate_of_z_ppds(posterior["lamb"], posterior["rate"], z)

    if not args.skip_prior:
        print("calculating chieff prior ppds...")
        prior_pchieff, xs = calculate_chieff_bspline_ppds(
            coefs=prior["chieff_cs"],
            model=chieff,
            nknots=args.chieff_nsplines,
            basis=LogYBSpline,
        )

    print("calculating chieff posterior ppds...")
    pchieff, xs = calculate_chieff_bspline_ppds(
        coefs=posterior["chieff_cs"],
        model=BSplineChiEffective,
        nknots=args.chieff_nsplines,
        basis=LogYBSpline,
    )

    print("plotting mass distribution...")
    priors = None if args.skip_prior else {"m1": prior_pm1s, "q": prior_pqs}
    fig = plot_mass_dist(
        pm1s,
        pqs,
        ms,
        qs,
        mmin=5.0,
        mmax=args.mmax,
        priors=priors,
    )
    plt.savefig(f"{label}_mass_distribution.png")
    del fig

    print("plotting chieff distribution...")
    prior = None if args.skip_prior else prior_pchieff
    fig = plot_chieff_dist(pchieff, xs, prior=prior)
    plt.savefig(f"{label}_chieff_distribution.png")
    del fig

    print("plotting R(z)...")
    prior = None if args.skip_prior else prior_Rofz
    fig = plot_rofz(Rofz, zs, prior=prior)
    plt.savefig(f"{label}_rate_vs_z.png")
    del fig
    prior = None if args.skip_prior else prior_Rofz
    fig = plot_rofz(Rofz, zs, logx=True, prior=prior)
    plt.savefig(f"{label}_rate_vs_z_logscale.png")
    del fig

    print("plotting m1/z PPC...")
    fig = plot_m1_vs_z_ppc(posterior, nObs, 5.0, args.mmax, z.zmax)
    plt.savefig(f"{label}_m1_vs_z_ppc.png")
    del fig

if __name__ == "__main__":
    main()