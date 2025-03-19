import os

import numpyro
import numpyro.distributions as dist
import xarray as xr

from utils import setup_result_dir
from utils import run_powerlawpeak_analysis

from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import load_pe_and_injections_as_dict
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.pipeline.utils import posterior_dict_to_xarray

from gwinferno.models.parametric.parametric import plpeak_primary_ratio_pdf
from gwinferno.models.parametric.parametric import independent_spin_tilt
from gwinferno.models.parametric.parametric import independent_spin_magnitude_beta_dist

from gwinferno.postprocess.calculations import calculate_powerlaw_peak_mass_ppds
from gwinferno.postprocess.calculations import calculate_beta_spin_mag
from gwinferno.postprocess.calculations import calculate_mixture_iso_aligned_spin_tilt
from gwinferno.postprocess.calculations import calculate_powerlaw_rate_of_z_ppds
from gwinferno.postprocess.plot import plot_mass_pdfs
from gwinferno.postprocess.plot import plot_rate_of_z_pdfs
from gwinferno.postprocess.plot import plot_spin_pdfs


def model(pedict, injdict, Nobs, Tobs, Ninj, z_model, mmin, mmax, param_names):
    """Numpyro model

    Args:
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injection data
        Nobs (int): Number of CBC events
        Tobs (float): analysis time
        Ninj (int): total number of generated injectiosn
        mass_models (list of objs): list containing initialized b-splines for primary mass and mass ratio
        mag_model (obj): initialized b-spline for spin magnitude
        tilt_model (obj): initialized b-spline for cos_tilt
        z_model (obj): initialized b-spline-powerlaw for redshift
        mmin (float): minimum mass
        mmax (float): maximum mass
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        param_names (list of str): list of parameters

    """

    #### Priors ####

    # Mass
    beta = numpyro.sample('beta', dist.Normal(0,5))
    alpha = numpyro.sample('alpha', dist.Normal(0,5))
    mu_peak = numpyro.sample('mu_peak', dist.Uniform(mmin, mmax))
    sig_peak = numpyro.sample('sig_peak', dist.HalfNormal(10))
    lambda_m = numpyro.sample('lambda_m', dist.Uniform(0,1))

    # Spin Magnitude (Independent)
    mu_a1 = numpyro.sample('mu_a1', dist.Uniform(0,1))
    var_a1 = numpyro.sample('var_a1', dist.Uniform(0.005, 0.25))
    mu_a2 = numpyro.sample('mu_a2', dist.Uniform(0,1))
    var_a2 = numpyro.sample('var_a2', dist.Uniform(0.005, 0.25))

    alpha_a1 = numpyro.deterministic('alpha_a1', mu_a1 * var_a1)
    alpha_a2 = numpyro.deterministic('alpha_a2', mu_a2 * var_a2)
    beta_a1 = numpyro.deterministic('beta_a1', (1 - mu_a1) * var_a1)
    beta_a2 = numpyro.deterministic('beta_a2', (1 - mu_a2) * var_a2)

    # Spin Tilt (Independent)

    lambda_ct1 = numpyro.sample('lambda_ct1', dist.Uniform(0,1))
    lambda_ct2 = numpyro.sample('lambda_ct2', dist.Uniform(0,1))
    sig_ct1 = numpyro.sample('sig_ct1', dist.Uniform(0.1, 4))
    sig_ct2 = numpyro.sample('sig_ct2', dist.Uniform(0.1, 4))

    # Redshift
    lamb = numpyro.sample("lamb", dist.Normal(0, 5))


    #### Calcualte weights ####

    def get_weights(datadict):

        p_m1q = plpeak_primary_ratio_pdf(datadict['mass_1'], datadict['mass_ratio'], alpha, beta, mmin, mmax, mu_peak, sig_peak, lambda_m)
        p_a = independent_spin_magnitude_beta_dist(datadict['a_1'], datadict['a_2'], alpha_a1, beta_a1, alpha_a2, beta_a2)
        p_ct = independent_spin_tilt(datadict['cos_tilt_1'], datadict['cos_tilt_2'], lambda_ct1, lambda_ct2, sig_ct1, sig_ct2)
        p_z = z_model(datadict["redshift"], lamb)

        weights_1 = p_m1q * p_a * p_ct * p_z / datadict["prior"]

        return weights_1

    pe_weights = get_weights(pedict)
    inj_weights = get_weights(injdict)

    #### Likelihood ####

    hierarchical_likelihood(
        pe_weights,
        inj_weights,
        float(Ninj),
        Nobs,
        Tobs,
        surveyed_hypervolume=z_model.normalization,
        vtfct_kwargs=dict(lamb=lamb),
        param_names=param_names,
        posterior_predictive_check=True,
        pedata=pedict,
        injdata=injdict,
        m2min=mmin,
        m1min=mmin,
        mmax=mmax,
    )


def main():

    """
    load argument parser (used when running script from command line)
    """

    base_parser = load_base_parser()

    ### example of function that adds additional arguments to the base parser.
    def add_args(parser):
        parser.add_argument("--example", type=str)
        return parser

    parser = add_args(base_parser)
    args = parser.parse_args()

    """
    Load PE and injections as dictionaries, along constants like # of observations, 
    injection analysis time, etc., and a list of the parameter names being modeled.
    """

    pedict, injdict, constants, param_names = load_pe_and_injections_as_dict(args.pe_inj_file)

    """
    Setup directory where results will be stored.
    """
    label, result_dir = setup_result_dir(args)

    """
    Run inference and save posterior samples to file. If flag --skip-inference present, then don't perform inference and load posterior samples from existing file.
    """

    if args.skip_inference:
        z_model = run_powerlawpeak_analysis(model, pedict, injdict, constants, param_names, args, skip_inference=True)
        print(f"loading posterior file: {result_dir}/{label}_posterior_samples.h5")
        posterior = xr.load_dataset(result_dir + f"/{label}_posterior_samples.h5")

    else:
        posterior_dict, z_model = run_powerlawpeak_analysis(model, pedict, injdict, constants, param_names, args)
        print(f"posteriors file saved: {result_dir}/{label}_posterior_samples.h5")
        posterior = posterior_dict_to_xarray(posterior_dict)
        posterior.to_netcdf(result_dir + f"/{label}_posterior_samples.h5")

    """
    Create list of population labels and corresponding colors. In this analysis, we are fitting the entire population with one model, so we only have 1 element. 
    
        Example of model with 2 Subpopulations: 
            names = ['Population A', 'Population B']
            colors = ['red', 'blue']
    """
    names = ["PowerlawPeak"]
    colors = ["tab:blue"]

    """
    Calculate Mass pdfs (for loop necessary for multiple subpopulations)
    """

    print("calculating mass ppds:")
    mass_pdfs = []
    q_pdfs = []
    for i in range(len(names)):
        mass, m1s, mass_ratio, qs = calculate_powerlaw_peak_mass_ppds(
            posterior["alpha"].values, posterior["beta"].values, posterior["mu_peak"].values, posterior["sig_peak"].values, posterior["lambda_m"].values, args.mmin, args.mmax
        )
        mass_pdfs.append(mass)
        q_pdfs.append(mass_ratio)

    """
    Calculate Mass pdfs (for loop necessary for multiple subpopulations)
    """
    print("calculating spin ppds:")
    mag1_pdfs = []
    mag2_pdfs = []
    tilt1_pdfs = []
    tilt2_pdfs = []
    for i in range(len(names)):
        mag1, _ = calculate_beta_spin_mag(
            posterior[f"alpha_a1"].values,
            posterior[f"beta_a1"].values,
        )
        mag2, mags = calculate_beta_spin_mag(
            posterior[f"alpha_a2"].values,
            posterior[f"beta_a2"].values,
        )
        tilt1, _ = calculate_mixture_iso_aligned_spin_tilt(
            posterior['sig_ct1'].values,
            posterior['lambda_ct1'].values
            )
        tilt2, tilts = calculate_mixture_iso_aligned_spin_tilt(
            posterior['sig_ct2'].values,
            posterior['lambda_ct2'].values
            )

        mag1_pdfs.append(mag1)
        mag2_pdfs.append(mag2)
        tilt1_pdfs.append(tilt1)
        tilt2_pdfs.append(tilt2)

    """
    Calculate rate as a funciton of redshift
    """
    print("calculating rate(z) ppds:")
    r_of_z, zs = calculate_powerlaw_rate_of_z_ppds(posterior["lamb"].values, posterior["rate"].values, z_model)

    """
    Save PDF plots of each parameter
    """
    print("plotting mass distributions:")
    plot_mass_pdfs(mass_pdfs, q_pdfs, m1s, qs, names, label, result_dir, save=args.save_plots, colors=colors)

    print("plotting primary spin distributions:")
    plot_spin_pdfs(mag1_pdfs, tilt1_pdfs, mags, tilts, names, label, result_dir, save=args.save_plots, colors=colors)

    print("plotting secondary spin distributions:")
    plot_spin_pdfs(mag2_pdfs, tilt2_pdfs, mags, tilts, names, label, result_dir, save=args.save_plots, colors=colors, secondary=True)

    print("plotting mass distributions:")
    plot_rate_of_z_pdfs(r_of_z, zs, label, result_dir, save=args.save_plots)

    """
    Convert dictionary of pdfs and params to an xarray Dataset
    """
    pdf_dict = {
        "a1": mag1_pdfs[0],
        "cos_tilt1": tilt1_pdfs[0],
        "a2": mag2_pdfs[0],
        "cos_tilt2": tilt2_pdfs[0],
        "mass_1": mass_pdfs[0],
        "mass_ratio": q_pdfs[0],
        "redshift": r_of_z,
    }
    param_dict = {"a1": mags, "a2": mags, "cos_tilt1": tilts, "cos_tilt2": tilts, "mass_1": m1s, "redshift": zs, "mass_ratio": qs}
    pdf_dataset = pdf_dict_to_xarray(pdf_dict, param_dict, args.samples)

    """
    Save dataset
    """
    pdf_dataset.to_netcdf(result_dir + f"/{label}_pdfs.h5")


if __name__ == "__main__":
    main()
