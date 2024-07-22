import os

import numpyro
import numpyro.distributions as dist
import xarray as xr
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from utils import run_analysis
from utils import setup_result_dir

from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.utils import bspline_mass_prior
from gwinferno.pipeline.utils import bspline_redshift_prior
from gwinferno.pipeline.utils import bspline_spin_prior
from gwinferno.pipeline.utils import load_base_parser
from gwinferno.pipeline.utils import load_pe_and_injections_as_dict
from gwinferno.pipeline.utils import pdf_dict_to_xarray
from gwinferno.pipeline.utils import posterior_dict_to_xarray
from gwinferno.postprocess.calculations import calculate_bspline_mass_ppds
from gwinferno.postprocess.calculations import calculate_bspline_spin_ppds
from gwinferno.postprocess.calculations import calculate_powerlaw_spline_rate_of_z_ppds
from gwinferno.postprocess.plot import plot_mass_pdfs
from gwinferno.postprocess.plot import plot_rate_of_z_pdfs
from gwinferno.postprocess.plot import plot_spin_pdfs


def model(pedict, injdict, Nobs, Tobs, Ninj, mass_models, mag_model, tilt_model, z_model, mmin, mmax, nspline_dict, param_names):
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

    mass_cs, q_cs = bspline_mass_prior(m_nsplines=nspline_dict["m1"], q_nsplines=nspline_dict["q"], m_tau=1, q_tau=1)

    a1_cs, tilt1_cs, a2_cs, tilt2_cs = bspline_spin_prior(
        a_nsplines=nspline_dict["a1"], ct_nsplines=nspline_dict["tilt1"], a_tau=25, ct_tau=25, IID=False
    )

    z_cs = bspline_redshift_prior(z_nsplines=nspline_dict["redshift"], z_tau=1)
    lamb = numpyro.sample("lamb", dist.Normal(0, 3))

    #### Calcualte weights ####

    def get_weights(datadict, pe_samples=True):

        p_m1q = mass_models(mass_cs, q_cs, pe_samples=pe_samples)
        p_a = mag_model(a1_cs, a2_cs, pe_samples=pe_samples)
        p_ct = tilt_model(tilt1_cs, tilt2_cs, pe_samples=pe_samples)

        p_z = z_model(datadict["redshift"], lamb, z_cs)

        weights_1 = p_m1q * p_a * p_ct * p_z / datadict["prior"]

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
        surv_hypervolume_fct=z_model.normalization,
        vtfct_kwargs=dict(lamb=lamb, cs=z_cs),
        param_names=param_names,
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

    nspline_dict = {
        "m1": args.m_nsplines,
        "q": args.q_nsplines,
        "a1": args.a_nsplines,
        "tilt1": args.tilt_nsplines,
        "a2": args.a_nsplines,
        "tilt2": args.tilt_nsplines,
        "redshift": args.z_nsplines,
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

    """
    Run inference and save posterior samples to file. If flag --skip-inference present, then don't perform inference and load posterior samples from existing file.
    """

    if args.skip_inference:
        z_model = run_analysis(model, pedict, injdict, constants, param_names, nspline_dict, args, skip_inference=True)
        print(f"loading posterior file: {result_dir}/{label}_posterior_samples.h5")
        posterior = xr.load_dataset(result_dir + f"/{label}_posterior_samples.h5")

    else:
        posterior_dict, z_model = run_analysis(model, pedict, injdict, constants, param_names, nspline_dict, args)
        print(f"posteriors file saved: {result_dir}/{label}_posterior_samples.h5")
        posterior = posterior_dict_to_xarray(posterior_dict)
        posterior.to_netcdf(result_dir + f"/{label}_posterior_samples.h5")

    """
    Create list of population labels and corresponding colors. In this analysis, we are fitting the entire population with one model, so we only have 1 element. 
    
        Example of model with 2 Subpopulations: 
            names = ['Population A', 'Population B']
            colors = ['red', 'blue']
    """
    names = ["B-Spline"]
    colors = ["tab:blue"]

    """
    Calculate Mass pdfs (for loop necessary for multiple subpopulations)
    """

    print("calculating mass ppds:")
    mass_pdfs = []
    q_pdfs = []
    for i in range(len(names)):
        mass, m1s, mass_ratio, qs = calculate_bspline_mass_ppds(
            posterior[f"mass_cs"].values, posterior[f"q_cs"].values, nspline_dict, args.mmin, args.mmax
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
        mag1, mag2, mags, tilt1, tilt2, tilts = calculate_bspline_spin_ppds(
            posterior[f"a1_cs"].values,
            posterior[f"tilt1_cs"].values,
            nspline_dict,
            a2_cs=posterior[f"a2_cs"].values,
            tilt2_cs=posterior[f"tilt2_cs"].values,
        )
        mag1_pdfs.append(mag1)
        mag2_pdfs.append(mag2)
        tilt1_pdfs.append(tilt1)
        tilt2_pdfs.append(tilt2)

    """
    Calculate rate as a funciton of redshift
    """
    print("calculating rate(z) ppds:")
    r_of_z, zs = calculate_powerlaw_spline_rate_of_z_ppds(posterior["lamb"].values, posterior["z_cs"].values, posterior["rate"].values, z_model)

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
