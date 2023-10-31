import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from jax import random
from numpyro import distributions as dist
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.models.bsplines.separable import BSplineIIDSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIIDSpinTilts
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.bsplines.smoothing import apply_difference_prior
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel
from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.parser import load_base_parser
from gwinferno.postprocess.calculate_ppds import calculate_iid_spin_bspline_ppds
from gwinferno.postprocess.calculate_ppds import calculate_m1q_bspline_ppds
from gwinferno.postprocess.calculate_ppds import calculate_powerbspline_rate_of_z_ppds
from gwinferno.postprocess.plotting import plot_iid_spin_dist
from gwinferno.postprocess.plotting import plot_m1_vs_z_ppc
from gwinferno.postprocess.plotting import plot_mass_dist
from gwinferno.postprocess.plotting import plot_ppc_brontosaurus
from gwinferno.postprocess.plotting import plot_rofz
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


def setup_spin_BSpline_model(injdata, pedata, pmap, magnknots, tiltnknots):
    magmodel = BSplineIIDSpinMagnitudes(
        magnknots,
        pedata[pmap["a_1"]],
        pedata[pmap["a_2"]],
        injdata[pmap["a_1"]],
        injdata[pmap["a_2"]],
        basis=LogYBSpline,
        normalize=True,
    )
    tiltmodel = BSplineIIDSpinTilts(
        tiltnknots,
        pedata[pmap["cos_tilt_1"]],
        pedata[pmap["cos_tilt_2"]],
        injdata[pmap["cos_tilt_1"]],
        injdata[pmap["cos_tilt_2"]],
        basis=LogYBSpline,
        normalize=True,
    )

    return {"mag": magmodel, "tilt": tiltmodel}


def setup_redshift_model(z_knots, injdata, pedata, pmap):
    z_pe = pedata[pmap["redshift"]]
    z_inj = injdata[pmap["redshift"]]
    model = PowerlawSplineRedshiftModel(z_knots, z_pe, z_inj, basis=LogXBSpline)
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

    mass_model = setup_mass_BSpline_model(
        injdata,
        pedata,
        param_map,
        args.mass_knots,
        args.q_knots,
        mmin=args.mmin,
        mmax=args.mmax,
    )
    z_model = setup_redshift_model(args.z_knots, injdata, pedata, param_map)
    spin_models = setup_spin_BSpline_model(injdata, pedata, param_map, args.mag_knots, args.tilt_knots)
    injdict = {k: injdata[param_map[k]] for k in param_names}
    pedict = {k: pedata[param_map[k]] for k in param_names}

    print(f"{len(injdict['redshift'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {pedict['redshift'].shape[1]} samples, over an observing time of {obs_time} yrs")

    return (
        mass_model,
        spin_models,
        z_model,
        pedict,
        injdict,
        total_inj,
        nObs,
        obs_time,
    )


def model(
    mass_model,
    spin_models,
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
    mag_model = spin_models["mag"]
    tilt_model = spin_models["tilt"]
    mag_knots = mag_model.primary_model.n_splines
    tilt_knots = tilt_model.primary_model.n_splines
    z_knots = z_model.nknots

    mass_cs = numpyro.sample("mass_cs", dist.Normal(0, 6), sample_shape=(mass_knots,))
    mass_tau_squared = numpyro.sample("mass_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.01), low = 0, high = 1))
    mass_lambda = numpyro.deterministic("mass_lambda", 1/mass_tau_squared)
    numpyro.factor("mass_log_smoothing_prior", apply_difference_prior(mass_cs, mass_lambda, degree=2))

    q_cs = numpyro.sample("q_cs", dist.Normal(0, 4), sample_shape=(q_knots,))
    q_tau_squared = numpyro.sample("q_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.1), low = 0, high = 1))
    q_lambda = numpyro.deterministic("q_lambda", 1/q_tau_squared)
    numpyro.factor("q_log_smoothing_prior", apply_difference_prior(q_cs, q_lambda, degree=2))

    mag_cs = numpyro.sample("mag_cs", dist.Normal(0, 2), sample_shape=(mag_knots,))
    mag_tau_squared = numpyro.sample("mag_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.1), low = 0, high = 1))
    mag_lambda = numpyro.deterministic("mag_lambda", 1/mag_tau_squared)
    numpyro.factor("mag_log_smoothing_prior", apply_difference_prior(mag_cs, mag_lambda, degree=2))

    tilt_cs = numpyro.sample("tilt_cs", dist.Normal(0, 2), sample_shape=(tilt_knots,))
    tilt_tau_squared = numpyro.sample("tilt_tau_squared", dist.TruncatedDistribution(dist.Normal(scale = 0.1), low = 0, high = 1))
    tilt_lambda = numpyro.deterministic("tilt_lambda", 1/tilt_tau_squared)
    numpyro.factor("tilt_log_smoothing_prior", apply_difference_prior(tilt_cs, tilt_lambda, degree=2))

    lamb = numpyro.sample("lamb", dist.Normal(0, 3))
    z_cs = numpyro.sample("z_cs", dist.Normal(), sample_shape=(z_knots,))
    z_tau_squared = numpyro.sample("z_tau_squared", dist.Uniform(1, 10))
    z_lambda = numpyro.deterministic("z_lambda", 1/z_tau_squared)
    numpyro.factor("z_log_smoothing_prior", apply_difference_prior(z_cs, z_lambda, degree=2))

    if not sample_prior:

        def get_weights(z, prior, pe_samples = True):
            p_m1q = mass_model(mass_cs, q_cs, pe_samples)
            p_a1a2 = mag_model(mag_cs, pe_samples)
            p_ct1ct2 = tilt_model(tilt_cs, pe_samples)
            p_z = z_model(z, lamb, z_cs)
            wts = p_m1q * p_a1a2 * p_ct1ct2 * p_z / prior
            
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

        peweights = get_weights(pedict["redshift"], pedict["prior"])
        injweights = get_weights(injdict["redshift"], injdict["prior"], pe_samples=False)
        hierarchical_likelihood(
            peweights,
            injweights,
            total_inj=total_inj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=z_model.normalization,
            vtfct_kwargs=dict(lamb=lamb, cs=z_cs),
            marginalize_selection=False,
            min_neff_cut=False,
            posterior_predictive_check=True,
            pedata=pedict,
            injdata=injdict,
            param_names=[
                "mass_1",
                "mass_ratio",
                "a_1",
                "a_2",
                "cos_tilt_1",
                "cos_tilt_2",
                "redshift",
            ],
        )


def main():
    args = load_parser()
    label = f"{args.outdir}/bsplines_{args.mass_knots}m1_{args.q_knots}q_iid{args.mag_knots}mag_iid{args.tilt_knots}tilt_pl{args.z_knots}z"
    mass, spin, z, pedict, injdict, total_inj, nObs, obs_time = setup(args)
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
            spin,
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
            spin,
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
            "mag_cs",
            "mag_lambda",
            "mass_cs",
            "mass_lambda",
            "q_cs",
            "q_lambda",
            "rate",
            "surveyed_hypervolume",
            "tilt_cs",
            "tilt_lambda",
            "z_cs",
            "z_lambda",
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
        print("calculating mag prior ppds...")
        prior_pmags, mags = calculate_iid_spin_bspline_ppds(prior["mag_cs"], BSplineIIDSpinMagnitudes, args.mag_knots, xmin=0, xmax=1, basis=LogYBSpline)
    print("calculating mag posterior ppds...")
    pmags, mags = calculate_iid_spin_bspline_ppds(posterior["mag_cs"], BSplineIIDSpinMagnitudes, args.mag_knots, xmin=0, xmax=1, basis=LogYBSpline)

    if not args.skip_prior:
        print("calculating tilt prior ppds...")
        prior_ptilts, tilts = calculate_iid_spin_bspline_ppds(prior["tilt_cs"], BSplineIIDSpinTilts, args.tilt_knots, xmin=-1, xmax=1, basis=LogYBSpline)
    print("calculating tilt posterior ppds...")
    ptilts, tilts = calculate_iid_spin_bspline_ppds(posterior["tilt_cs"], BSplineIIDSpinTilts, args.tilt_knots, xmin=-1, xmax=1, basis=LogYBSpline)

    if not args.skip_prior:
        print("calculating rate prior ppds...")
        prior_Rofz, zs = calculate_powerbspline_rate_of_z_ppds(prior["lamb"], prior["z_cs"], jnp.ones_like(prior["lamb"]), z)
    print("calculating rate posterior ppds...")
    Rofz, zs = calculate_powerbspline_rate_of_z_ppds(posterior["lamb"], posterior["z_cs"], posterior["rate"], z)

# Lines (357-383) are commented out due to deepdish errors
    # if not args.skip_prior:
    #     prior_ppd_dict = {
    #     "pm1": prior_pm1s,
    #     "pq": prior_pqs,
    #     "pa": prior_pmags,
    #     "pct": prior_ptilts,
    #     "m1s": ms,
    #     "qs": qs,
    #     "mags": mags,
    #     "tilts": tilts,
    #     "Rofz": prior_Rofz,
    #     "zs": zs,
    #     }

    #     dd.io.save(f"{label}_prior_ppds.h5", prior_ppd_dict)
    #     del prior_ppd_dict

    # ppd_dict = {
    #     "dRdm1": pm1s,
    #     "dRdq": pqs,
    #     "m1s": ms,
    #     "qs": qs,
    #     "dRda": pmags,
    #     "mags": mags,
    #     "dRdct": ptilts,
    #     "tilts": tilts,
    #     "Rofz": Rofz,
    #     "zs": zs,
    # }
    # dd.io.save(f"{label}_ppds.h5", ppd_dict)
    # del ppd_dict

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

    print("plotting spin distributions...")
    priors = None if args.skip_prior else {"mags": prior_pmags, "tilts": prior_ptilts}
    fig = plot_iid_spin_dist(pmags, ptilts, mags, tilts, priors=priors)
    plt.savefig(f"{label}_iid_component_spin_distribution.png")
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

    print("plotting m1 brontasaurus PPC...")
    fig = plot_ppc_brontosaurus(posterior, nObs, 5.0, args.mmax, z.zmax)
    plt.savefig(f"{label}_m1_ppc_brontosaurus.png")
    del fig


if __name__ == "__main__":
    main()