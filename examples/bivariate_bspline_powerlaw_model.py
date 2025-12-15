import os

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt

from utils import run_powerlaw_peak_bbspline_chiq_analysis
from utils import setup_result_dir

from  gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.utils import bspline_massratio_chieff_prior_2d

from gwinferno.models.parametric.parametric import plpeak_primary_pdf

from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import load_pe_and_injections_as_dict
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.pipeline.utils import posterior_dict_to_xarray
from gwinferno.postprocess.calculations import postprocess_min_neff_cut
from gwinferno.postprocess.calculations import postprocess_max_variance_cut

from gwinferno.postprocess.calculations import calculate_plpluspeak_primary_bbspline_chiq_ppds
from gwinferno.postprocess.calculations import calculate_powerlaw_rate_of_z_ppds

from gwinferno.postprocess.plot import plot_primary_pdfs
from gwinferno.postprocess.plot import plot_chiq_pdfs_percentiles
from gwinferno.postprocess.plot import plot_chiq_pdfs_samples
from gwinferno.postprocess.plot import plot_chiq_pdfs_stats
from gwinferno.postprocess.plot import plot_marginal_chi_q_pdfs
from gwinferno.postprocess.plot import plot_chiq_slices
from gwinferno.postprocess.plot import plot_rate_of_z_pdfs

def model(pedict, injdict, Nobs, Tobs, Ninj, chiq_model, z_model, mmin, mmax, nspline_dict, hyper_params_dict, param_names):
    """Numpyro model

    Args:
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injection data
        Nobs (int): Number of CBC events
        Tobs (float): analysis time
        Ninj (int): total number of generated injections
        chiq_model (objs): list containing initialized b-splines for effective spin-mass ratio
        z_model (obj): initialized b-spline-powerlaw for redshift
        mmin (float): minimum mass
        mmax (float): maximum mass
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        hyper_params_dict (dict): dictionary containing hyperparameters 
        param_names (list of str): list of parameters
    """

    #### Priors ####

    # Primary mass
    primary_index = numpyro.sample('primary_index', dist.Uniform(-4,12))
    primary_mmin = numpyro.sample('primary_mmin', dist.Uniform(2,10))
    primary_mmax = numpyro.sample('primary_mmax', dist.Uniform(30,100))
    primary_mu = numpyro.sample('primary_mu', dist.Uniform(20,50))
    primary_sig = numpyro.sample('primary_sig', dist.Uniform(1,10))
    primary_lamb = numpyro.sample('primary_lamb', dist.Uniform(0,1))
    # primary_delta = numpyro.sample('primary_delta', dist.Uniform(0,10))

    # Effective-spin mass-ratio
    chiq_cs = bspline_massratio_chieff_prior_2d(chi_eff_nsplines=nspline_dict['chi_eff'], q_nsplines=nspline_dict['q'],
                    tau_row=hyper_params_dict['chiq_tau_row'], tau_column=hyper_params_dict['chiq_tau_column'], order=hyper_params_dict['chiq_diff_order'])
    
    # Redshift
    lamb = numpyro.sample('lamb', dist.Normal(0,3))

    #### Calculate weights ####

    def get_weights(datadict, pe_samples):
        
        p_m = plpeak_primary_pdf(datadict['mass_1'], primary_index, primary_mmin, primary_mmax, primary_mu, primary_sig, primary_lamb, delta=3)
        p_chiq = chiq_model(chiq_cs, pe_samples=pe_samples)
        p_z = z_model(datadict['redshift'], lamb)

        weights = p_m * p_chiq * p_z / datadict['prior']
        return weights

    pe_weights = get_weights(pedict, pe_samples=True)
    inj_weights = get_weights(injdict, pe_samples=False)

    #### Likelihood ####
    hierarchical_likelihood(
        pe_weights, inj_weights, float(Ninj), Nobs, Tobs,
        z_model.normalization(lamb=lamb),
        param_names=param_names, pedata=pedict, injdata=injdict,
        m2min=mmin, m1min=mmin, mmax=mmax,
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
        'chi_eff': args.chi_eff_nsplines,
        'q': args.q_nsplines
    }
    hyper_params_dict = {
        'chiq_order': tuple(args.chiq_order),
        'chiq_tau_row': args.chiq_tau_row,
        'chiq_tau_column': args.chiq_tau_column,
        'chiq_diff_order': args.chiq_diff
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
    dof_label = f'plpeak_plus_n-chi-eff-{nspline_dict['chi_eff']}_n-q-{nspline_dict['q']}_chiq-tau-r-{hyper_params_dict['chiq_tau_row']}_chiq-tau-c-{hyper_params_dict['chiq_tau_column']}_chiq-diff-{hyper_params_dict['chiq_diff_order']}'
    full_dir = f'{result_dir}/{dof_label}/full_posterior'
    cut_dir = f'{result_dir}/{dof_label}/cut_posterior'
    dirs_dict = {'full_posterior':full_dir, 'cut_posterior':cut_dir}
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        os.makedirs(cut_dir)

    """
    Run inference and save posterior samples to file. If flag --skip-inference present, then don't perform inference and load posterior samples from existing file.
    """

    if args.skip_inference:
        z_model = run_powerlaw_peak_bbspline_chiq_analysis(model, pedict, injdict, constants, param_names, nspline_dict, hyper_params_dict, args, skip_inference=True)
        print(f'loading posterior file: {full_dir}/{label}_posterior_samples.h5')
        full_posterior = xr.load_dataset(f'{full_dir}/{label}_posterior_samples.h5')

    else:
        mcmc, posterior_dict, z_model, trace_plots = run_powerlaw_peak_bbspline_chiq_analysis(model, pedict, injdict, constants, param_names, nspline_dict, hyper_params_dict, args)
        fig = plt.gcf()
        fig.tight_layout()
        print(f'posteriors file saved: {full_dir}/{label}_posterior_samples.h5')
        full_posterior = az.from_numpyro(mcmc)['posterior']
        full_posterior.to_netcdf(f'{full_dir}/{label}_posterior_samples.h5')
        # posterior = posterior_dict_to_xarray(posterior_dict)
        plt.savefig(f'{full_dir}/{label}_trace_plots.png')

    """
    Create list of population labels and corresponding colors. In this analysis, we are fitting the entire population with one model, so we only have 1 element. 
    
        Example of model with 2 Subpopulations: 
            names = ['Population A', 'Population B']
            colors = ['red', 'blue']
    """
    names = ['B-Spline']
    colors = ['tab:blue']

    """
    Remove samples that are below the minimum effective cut
    """
    if args.remove_threshold:
        full_posterior = full_posterior.sel(chain=0)
    print('Imposing cuts on injection and PE samples...')
    if args.threshold == 'min_N_eff':
        cut_posterior = postprocess_min_neff_cut(full_posterior, Nobs_cut=False)
    elif args.threshold == 'max_variance':
        cut_posterior = postprocess_max_variance_cut(full_posterior)
    else:
        raise AssertionError('The threshold must be of `min_N_eff` cut or `max_variance` cut.')
    # TODO: defining posterior with `.sel(chain=0)` when len(posterior.draw) == 0 already gives error, assertion not needed?
    # assert len(posterior.draw) > 0, 'Imposed threshold(s) returned 0 draws of the MCMC. Choose either a different cut or remove the threshold.'
    cut_posterior.to_netcdf(f'{cut_dir}/{label}_posterior_samples_{args.threshold}_cut.h5')
    posteriors_dict = {'full_posterior':full_posterior, 'cut_posterior':cut_posterior}

    """
    Calculate primary and effective spin-mass ratio pdfs
    """

    for key, posterior in posteriors_dict.items():
        dir = dirs_dict[key]

        print(f'calculating {key} primary and effective spin-mass ratio ppds:')
        # TODO: there's gotta be a better way to make these arrays

        mass_pdfs = []
        chiq_pdfs = []
        chieff_pdfs = []
        q_pdfs = []
        mpdfs, m1s, chiqpdfs, chieffpdfs, qpdfs, chi_effs, qs, lims_dict = calculate_plpluspeak_primary_bbspline_chiq_ppds(
            posterior['primary_index'].values, posterior['primary_mmin'].values, posterior['primary_mmax'].values,
            posterior['primary_mu'].values, posterior['primary_sig'].values, posterior['primary_lamb'].values,
            3, posterior['chi_eff_q_cs'].values, nspline_dict=nspline_dict, hyper_params_dict=hyper_params_dict,
            mmin=args.mmin, mmax=args.mmax,
        )
        CHIs, Qs = jnp.meshgrid(chi_effs, qs, indexing = 'ij')
        mass_pdfs.append(mpdfs)
        chiq_pdfs.append(chiqpdfs)
        chieff_pdfs.append(chieffpdfs)
        q_pdfs.append(qpdfs)
        mass_pdfs = jnp.array(mass_pdfs)
        chiq_pdfs = jnp.array(chiq_pdfs)
        chieff_pdfs = jnp.array(chieff_pdfs)
        q_pdfs = jnp.array(q_pdfs)

        """
        Calculate rate as a funciton of redshift
        """
        print(f'calculating {key} rate(z) ppds:')
        r_of_z, zs = calculate_powerlaw_rate_of_z_ppds(posterior['lamb'].values, posterior['rate'].values, z_model)

        """
        Save PDF plots of each parameter
        """

        print(f'plotting {key} primary mass and effective spin-mass ratio distributions:')
        plot_primary_pdfs(mass_pdfs, m1s, names, label, dir, save=args.save_plots)
        plot_marginal_chi_q_pdfs(chieff_pdfs, q_pdfs, chi_effs, qs, lims_dict['chieff_lims'], lims_dict['qmin'], label, dir, save=args.save_plots)
        plot_chiq_slices(chiq_pdfs, chi_effs, qs, label, dir, save=args.save_plots)
        plot_chiq_pdfs_percentiles(chiq_pdfs, CHIs, Qs, label, dir, save=args.save_plots)
        plot_chiq_pdfs_samples(chiq_pdfs, CHIs, Qs, label, dir, save=args.save_plots)
        sample_frac = 0.1 if key == 'full_posterior' else 1.0
        plot_chiq_pdfs_stats(chiq_pdfs, CHIs, Qs, label, dir, sample_frac=sample_frac, save=args.save_plots)

        print(f'plotting {key} redshift distributions:')
        plot_rate_of_z_pdfs(r_of_z, zs, label, dir, save=args.save_plots)

if __name__ == "__main__":
    main()