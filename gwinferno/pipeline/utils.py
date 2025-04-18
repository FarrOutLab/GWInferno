from argparse import ArgumentParser

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr

from gwinferno.interpolation import LogXBSpline
from gwinferno.interpolation import LogXLogYBSpline
from gwinferno.interpolation import LogYBSpline
from gwinferno.models.bsplines.separable import BSplineIIDSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIIDSpinTilts
from gwinferno.models.bsplines.separable import BSplineIndependentSpinMagnitudes
from gwinferno.models.bsplines.separable import BSplineIndependentSpinTilts
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio
from gwinferno.models.bsplines.smoothing import apply_difference_prior
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel


def load_base_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--pe-inj-file",
        type=str,
    )
    parser.add_argument("--run-label", type=str)
    parser.add_argument("--result-dir", type=str)
    parser.add_argument("--m-nsplines", type=int, default=50)
    parser.add_argument("--q-nsplines", type=int, default=30)
    parser.add_argument("--a-nsplines", type=int, default=16)
    parser.add_argument("--tilt-nsplines", type=int, default=16)
    parser.add_argument("--z-nsplines", type=int, default=20)
    parser.add_argument("--mmin", type=float, default=3.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    parser.add_argument("--rngkey", type=int, default=1)
    parser.add_argument("--save-plots", type=bool, default=True)
    return parser


"""
Load data
"""


def load_pe_and_injections_as_dict(file, ignore=[]):
    """Load PE and injection file created by `gwinferno.preprocess.data_collection.save_posterior_samples_and_injection_datasets_as_idata()`.

    Parameters
    ----------
    file : str
        Path to NetCDF file containing parameter estimation samples and injection data.

    Returns
    -------
    pedict : dict
        Dictionary of parameter estimation samples.
    injdict : dict
        Dictionary of injection data.
    constants : dict
        Dictionary of constants, e.g., total number of generated injections.
    param_names : List[str]
        List of parameter names.

    See Also
    --------
    gwinferno.preprocess.data_collection.save_posterior_samples_and_injection_datasets_as_idata
    """
    data = az.from_netcdf(file)
    print(f"data file {file} loaded")

    if ignore:
        sel = np.zeros(data.pe_data["event"].values.shape, dtype=bool)
        for gw in ignore:
            sel += data.pe_data["event"] == gw
        sel = ~sel
        pedict = {k: jnp.asarray(data.pe_data.posteriors.sel(param=k).values[sel]) for k in data.pe_data.param.values}
    else:
        pedict = {k: jnp.asarray(data.pe_data.posteriors.sel(param=k).values) for k in data.pe_data.param.values}

    injdict = {k: jnp.asarray(data.inj_data.injections.sel(param=k).values) for k in data.inj_data.param.values}

    param_names = list(data.pe_data.param.values)

    total_inj = data.inj_data.attrs["total_generated"]
    obs_time = data.inj_data.attrs["analysis_time"]
    nObs = data.pe_data.posteriors.shape[0]
    constants = {"total_inj": total_inj, "obs_time": obs_time, "nObs": nObs}

    return pedict, injdict, constants, param_names


"""
Setup B-Spline models
"""


def setup_bspline_mass_models(pedict, injdict, m_nsplines, q_nsplines, mmin, mmax):
    print("initializing spline mass design matrices")
    return BSplinePrimaryBSplineRatio(
        m_nsplines,
        q_nsplines,
        pedict["mass_1"],
        injdict["mass_1"],
        pedict["mass_ratio"],
        injdict["mass_ratio"],
        m1min=mmin,
        m2min=mmin,
        mmax=mmax,
        kwargs_m={"basis": LogXLogYBSpline},
        kwargs_q={"basis": LogYBSpline},
    )


def setup_bspline_spin_models(pedict, injdict, a1_nsplines, ct1_nsplines, IID=False, a2_nsplines=None, ct2_nsplines=None):
    print("initializing spline spin design matrices")

    if IID:
        tilt_model = BSplineIIDSpinTilts(
            ct1_nsplines, pedict["cos_tilt_1"], pedict["cos_tilt_2"], injdict["cos_tilt_1"], injdict["cos_tilt_2"], normalize=True
        )

        mag_model = BSplineIIDSpinMagnitudes(a1_nsplines, pedict["a_1"], pedict["a_2"], injdict["a_1"], injdict["a_2"], normalize=True)

    else:
        tilt_model = BSplineIndependentSpinTilts(
            ct1_nsplines,
            ct2_nsplines,
            pedict["cos_tilt_1"],
            pedict["cos_tilt_2"],
            injdict["cos_tilt_1"],
            injdict["cos_tilt_2"],
            normalize=True,
        )

        mag_model = BSplineIndependentSpinMagnitudes(
            a1_nsplines, a2_nsplines, pedict["a_1"], pedict["a_2"], injdict["a_1"], injdict["a_2"], normalize=True
        )

    return mag_model, tilt_model


def setup_powerlaw_spline_redshift_model(pedict, injdict, z_nsplines, basis=LogXBSpline):
    print("initializing redshift model")
    return PowerlawSplineRedshiftModel(z_nsplines, pedict["redshift"], injdict["redshift"], basis=basis)


"""
Setup B-Spline Priors
"""


def bspline_mass_prior(m_nsplines=None, q_nsplines=None, m_tau=1, q_tau=1, name=None, m_cs_sig=15, q_cs_sig=5, m_deg=1, q_deg=1):

    name = "_" + name if name is not None else ""

    if m_nsplines is not None:
        mass_cs = numpyro.sample("mass_cs" + name, dist.Normal(0, m_cs_sig), sample_shape=(m_nsplines,))
        numpyro.factor("mass_smoothing_prior" + name, apply_difference_prior(mass_cs, m_tau, degree=m_deg))

    if q_nsplines is not None:
        q_cs = numpyro.sample("q_cs" + name, dist.Normal(0, q_cs_sig), sample_shape=(q_nsplines,))
        numpyro.factor("q_smoothing_prior" + name, apply_difference_prior(q_cs, q_tau, degree=q_deg))

    if m_nsplines is not None and q_nsplines is None:
        return mass_cs
    if m_nsplines is None and q_nsplines is not None:
        return q_cs
    if m_nsplines is None and q_nsplines is None:
        raise AssertionError("number of mass splines or q splines must be specified.")
    else:
        return mass_cs, q_cs


def bspline_spin_prior(a_nsplines=None, ct_nsplines=None, a_tau=None, ct_tau=None, name=None, IID=False, a_cs_sig=5, ct_cs_sig=5, a_deg=2, ct_deg=2):

    name = "_" + name if name is not None else ""

    if IID:
        a_cs = numpyro.sample("a_cs" + name, dist.Normal(0, a_cs_sig), sample_shape=(a_nsplines,))
        numpyro.factor("a_smoothing_prior" + name, apply_difference_prior(a_cs, a_tau, degree=a_deg))

        ct_cs = numpyro.sample("tilt_cs" + name, dist.Normal(0, ct_cs_sig), sample_shape=(ct_nsplines,))
        numpyro.factor("ct_smoothing_prior" + name, apply_difference_prior(ct_cs, ct_tau, degree=ct_deg))
        return a_cs, ct_cs

    else:
        a1_cs = numpyro.sample("a1_cs" + name, dist.Normal(0, a_cs_sig), sample_shape=(a_nsplines,))
        numpyro.factor("a1_smoothing_prior" + name, apply_difference_prior(a1_cs, a_tau, degree=a_deg))
        a2_cs = numpyro.sample("a2_cs" + name, dist.Normal(0, a_cs_sig), sample_shape=(a_nsplines,))
        numpyro.factor("a2_smoothing_prior" + name, apply_difference_prior(a2_cs, a_tau, degree=a_deg))

        ct1_cs = numpyro.sample("tilt1_cs" + name, dist.Normal(0, ct_cs_sig), sample_shape=(ct_nsplines,))
        numpyro.factor("ct1_smoothing_prior" + name, apply_difference_prior(ct1_cs, ct_tau, degree=ct_deg))
        ct2_cs = numpyro.sample("tilt2_cs" + name, dist.Normal(0, ct_cs_sig), sample_shape=(ct_nsplines,))
        numpyro.factor("ct2_smoothing_prior" + name, apply_difference_prior(ct2_cs, ct_tau, degree=ct_deg))

        return a1_cs, ct1_cs, a2_cs, ct2_cs


def bspline_redshift_prior(z_nsplines=None, z_tau=None, name=None, z_cs_sig=1, z_deg=2):
    name = "_" + name if name is not None else ""
    z_cs = numpyro.sample("z_cs" + name, dist.Normal(0, z_cs_sig), sample_shape=(z_nsplines,))
    numpyro.factor("z_smoothing_prior" + name, apply_difference_prior(z_cs, z_tau, degree=z_deg))
    return z_cs


def posterior_dict_to_xarray(posteriors):

    for key in posteriors.keys():
        n_samples = posteriors[key].shape[0]
        posteriors[key] = {"dims": "draw", "data": posteriors[key]}

        if posteriors[key]["data"].shape != (n_samples,):
            new_dims = ["draw"] + [f"{key}_dim{i + 2}" for i in range(len(posteriors[key]["data"].shape) - 1)]
            posteriors[key]["dims"] = new_dims

    return xr.Dataset.from_dict(posteriors)


def pdf_dict_to_xarray(pdf_dict, param_dict, n_samples, subpop_names=None):
    xr_dict = {}
    if subpop_names is None:
        pdfs = {f"{key}_pdfs": (["draw", key], item) for key, item in pdf_dict.items()}
        xr_dict = xr_dict | pdfs
    else:
        z = {"redshift_pdfs": (["draw", "redshift"], pdf_dict["redshift"])}
        xr_dict = xr_dict | z
        del pdf_dict["redshift"]
        for i, nm in enumerate(subpop_names):
            single = {f"{nm}_{key}_pdfs": (["draw", key], item[i]) for key, item in pdf_dict.items()}
            xr_dict = xr_dict | single

    coords = {key: ([key], item) for key, item in param_dict.items()}
    coords = coords | {"draw": (["draw"], jnp.arange(n_samples))}

    pdf_dataset = xr.Dataset(xr_dict, coords=coords)

    return pdf_dataset
