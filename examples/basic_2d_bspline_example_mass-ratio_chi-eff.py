import os

import numpyro
import numpyro.distributions as dist
import xarray as xr
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import run_massratio_chieff_bspline_analysis_2d
from utils import setup_result_dir

from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.utils import bspline_mass_prior
from gwinferno.pipeline.utils import bspline_redshift_prior
from gwinferno.pipeline.utils import bspline_massratio_chieff_prior_2d
from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import load_pe_and_injections_as_dict
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.pipeline.utils import posterior_dict_to_xarray
from gwinferno.postprocess.calculations import postprocess_min_neff_cut
from gwinferno.postprocess.calculations import calculate_bspline_primary_chiq_ppds
from gwinferno.postprocess.calculations import calculate_powerlaw_spline_rate_of_z_ppds
from gwinferno.postprocess.plot import plot_primary_chiq_pdfs
from gwinferno.postprocess.plot import plot_primary_pdfs
from gwinferno.postprocess.plot import plot_chiq_pdfs_stats
from gwinferno.postprocess.plot import plot_chiq_pdfs_samples
from gwinferno.postprocess.plot import plot_chiq_pdfs_rng_stats
from gwinferno.postprocess.plot import plot_rate_of_z_pdfs

def model(pedict, injdict, Nobs, Tobs, Ninj, mass_model, chiq_model, z_model, mmin, mmax, nspline_dict, param_names, hyper_params_dict):
    """Numpyro model

    Args:
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injection data
        Nobs (int): Number of CBC events
        Tobs (float): analysis time
        Ninj (int): total number of generated injections
        m_chiq_models (list of objs): list containing initialized b-splines for primary mass and effective spin-mass ratio
        z_model (obj): initialized b-spline-powerlaw for redshift
        mmin (float): minimum mass
        mmax (float): maximum mass
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        param_names (list of str): list of parameters
    """

    #### Priors ####

    mass_cs = bspline_mass_prior(m_nsplines=nspline_dict["m1"], m_tau=1)

    chiq_cs = bspline_massratio_chieff_prior_2d(chi_eff_nsplines=nspline_dict["chi_eff"], q_nsplines=nspline_dict["q"], tau=, order=1)

    z_cs = bspline_redshift_prior(z_nsplines=nspline_dict["redshift"], z_tau=1)
    lamb = numpyro.sample("lamb", dist.Normal(0,3))

    #### Calculate weights ####

    def get_weights(datadict, pe_samples=True):

        p_m = mass_model(mass_cs, pe_samples=pe_samples)
        p_chiq = chiq_model(chiq_cs, pe_samples=pe_samples)

        p_z = z_model(datadict["redshift"], lamb, z_cs)

        weights_1 = p_m * p_chiq * p_z / datadict["prior"]

        return weights_1
    
    pe_weights = get_weights(pedict, pe_samples=True)
    inj_weights = get_weights(injdict, pe_samples=False)
    
    #### Likelihood ####
    hierarchical_likelihood(
        pe_weights,
        inj_weights,
        float(Ninj),
        Nobs,
        Tobs,
        z_model.normalization(lamb=lamb, cs=z_cs),
        param_names=param_names,
        pedata=pedict,
        injdata=injdict,
        m2min=mmin,
        m1min=mmin,
        mmax=mmax,
        min_neff_cut=False,
    )

def main():

    """
    Load argument parser (used when running script from command line)
    """

    base_parser = load_base_parser()

    ### Example of function that adds additional arguments to the base parser
    def add_args(parser):
        parser.add_argument("--example", type=str)
        return parser
    
    parser = add_args(base_parser)
    args = parser.parse_args()

    nspline_dict = {
        "m1": args.m_nsplines,
        "chi_eff": args.chi_eff_nsplines,
        "q": args.q_nsplines,
        "redshift": args.z_nsplines
    }
    hyper_params_dict = {
        "chiq_tau_row": args.chiq_tau_r,
        "chiq_tau_column": args.chiq_tau_c,
        "chiq_diff_order": args.chiq_diff
    }

    """
    Load PE and injections as dictionaries, along constants like # of observations, 
    injection analysis time, etc., and a list of the parameter names being modeled.
    """

    pedict, injdict, constants, param_names = load_pe_and_injections_as_dict(args.pe_inj_file)

    """
    Setup directory where results will be stored.
    """
    label, result_dir = setup_result_dir(args)
    dof_label = f'n-chi-eff-{nspline_dict['chi_eff']}_n-q-{nspline_dict['q']}_chiq-tau-{args.chiq_tau}_chiq-diff-{args.chiq_diff}'
    full_dir = f'{result_dir}/{dof_label}'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    """
    Run inference and save posterior samples to file. If flag --skip-inference present, then don't perform inference and load posterior samples from existing file.
    """

    if args.skip_inference:
        z_model = run_massratio_chieff_bspline_analysis_2d(model, pedict, injdict, constants, param_names, nspline_dict, args, skip_inference=True)
        print(f"loading posterior file: {result_dir}/{dof_label}/{label}_posterior_samples.h5")
        mcmc_posterior = xr.load_dataset(result_dir + f"/{dof_label}/{label}_posterior_samples.h5")

    else:
        mcmc, posterior_dict, z_model, trace_plots = run_massratio_chieff_bspline_analysis_2d(model, pedict, injdict, constants, param_names, nspline_dict, args)
        fig = plt.gcf()
        fig.tight_layout()
        print(f"posteriors file saved: {result_dir}/{dof_label}/{label}_posterior_samples.h5")
        mcmc_posterior = az.from_numpyro(mcmc)['posterior']
        mcmc_posterior.to_netcdf(result_dir + f"/{dof_label}/{label}_posterior_samples.h5")
        # posterior = posterior_dict_to_xarray(posterior_dict)
        plt.savefig(result_dir + f"/{dof_label}/{label}_trace_plots.png")


    """
    Create list of population labels and corresponding colors. In this analysis, we are fitting the entire population with one model, so we only have 1 element. 
    
        Example of model with 2 Subpopulations: 
            names = ['Population A', 'Population B']
            colors = ['red', 'blue']
    """
    names = ["B-Spline"]
    colors = ["tab:blue"]

    """
    Remove samples that are below the minimum effective cut
    """
    print("Imposing cuts on injection and PE samples...")
    posterior = postprocess_min_neff_cut(mcmc_posterior, Nobs_cut=False).sel(chain=0)
    posterior.to_netcdf(result_dir + f"/{dof_label}/{label}_posterior_samples_cut.h5")
    """
    Calculate primary and effective spin-mass ratio pdfs
    """

    print("calculating primary and effective spin-mass ratio ppds:")
    mass_pdfs = []
    chiq_pdfs = []
    mpdfs, m1s, chiqpdfs, chi_effs, qs = calculate_bspline_primary_chiq_ppds(m_cs=posterior[f"mass_cs"].values, chiq_cs=posterior[f"chi_eff_q_cs"].values, nspline_dict=nspline_dict, mmin=args.mmin, mmax=args.mmax)
    mass_pdfs.append(mpdfs)
    chiq_pdfs.append(chiqpdfs)
    mass_pdfs = jnp.array(mass_pdfs)
    chiq_pdfs = jnp.array(chiq_pdfs)

    """
    Calculate rate as a funciton of redshift
    """
    print("calculating rate(z) ppds:")
    r_of_z, zs = calculate_powerlaw_spline_rate_of_z_ppds(posterior["lamb"].values, posterior["z_cs"].values, posterior["rate"].values, z_model)

    """
    Save PDF plots of each parameter
    """

    print("plotting primary mass and effective spin-mass ratio distributions:")
    # plot_primary_chiq_pdfs(mass_pdfs, chiq_pdfs, m1s, chi_effs, qs, names, label, result_dir, save=args.save_plots)
    plot_primary_pdfs(mass_pdfs, m1s, names, label, full_dir, save=args.save_plots)
    plot_chiq_pdfs_samples(chiq_pdfs, chi_effs, qs, label, full_dir, save=args.save_plots)
    plot_chiq_pdfs_stats(chiq_pdfs, chi_effs, qs, label, full_dir, save=args.save_plots)
    plot_chiq_pdfs_rng_stats(chiq_pdfs, chi_effs, qs, label, full_dir, save=args.save_plots)


    print("plotting redshift distributions:")
    plot_rate_of_z_pdfs(r_of_z, zs, label, full_dir, save=args.save_plots)

    """
    Convert dictionary of pdfs and params to an xarray Dataset
    """
    pdf_dict = {
        "mass_1": mass_pdfs[0],
        "chiq": chiq_pdfs[0],
        "redshift": r_of_z,
    }
    ## TODO: modify or define new function that takes pdf dict to xarray

if __name__ == "__main__":
    main()