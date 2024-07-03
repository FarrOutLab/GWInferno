"""
a module that stores the meat of the calculations for hierarchical population inference
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy.cosmology import Planck15
from jax import jit
from jax import random
from jax.scipy.integrate import trapezoid
from jax.scipy.special import logsumexp
from numpyro.infer import SVI
from numpyro.infer import Trace_ELBO
from numpyro.infer import autoguide
from numpyro.optim import Adam

from .parser import PopMixtureModel
from .parser import PopModel

NP_KERNEL_MAP = {"NUTS": numpyro.infer.NUTS, "HMC": numpyro.infer.HMC}


def find_map(rng_key, numpyro_model, *model_args, Niter=100, lr=0.01):
    """Find the MAP estimate for a given NumPyro model using SVI with Adam optimizing the ELBO

    Parameters
    ----------
        rng_key : jax.random.PRNGKey
            RNG Key to be passed to SVI.run().
        numpyro_model : callable
            Python callable containing `numpyro.primitives`.
        Niter : int, optional
            Number of iterations to run variational inference. Defaults to 100.
        lr : float, optional
            learning rate used for Adam optimizer. Defaults to 0.01.

    Returns
    -------
        SVIRunResult.params : dict
            parameters of the result of MAP optimization
    """
    guide = autoguide.AutoDelta(numpyro_model)
    optimizer = Adam(lr)
    svi = SVI(numpyro_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(rng_key, Niter, *model_args)
    return svi_results.params


@partial(jit, static_argnames=["log"])
def per_event_log_bayes_factors(weights, log=False):
    r"""Calculates per-event log Bayes factors via importance sampling

    .. math::

        \mathrm{BF}_i = \int p(d_i|\theta)p(\theta|\Lambda) d\theta \approx
        \frac{1}{N_s}\sum_{j=1}^{N_s} \frac{p(\theta|\Lambda)}{p(\theta|\Lambda_\emptyset)}

    Parameters
    ----------
    weights : jax.DeviceArray
        JAX array of weights to integrate over. Expected size of `(N_events,N_samples)`.
    log : bool, optional
        Flag to perform calculations in log probability. Interprets weights as log weights.
        This is slower but more numerically stable. Defaults to False.

    Returns
    -------
    jax.DeviceArray
        Array of per-event log bayes factors.
    jax.DeviceArray
        Array of per-event log effective samples sizes from Monte Carlo integrals.
    """
    if log:
        logweights = weights
        logBFs = logsumexp(logweights, axis=1)
        logn_effs = 2 * logBFs - logsumexp(2 * logweights, axis=1)
        logBFs -= jnp.log(logweights.shape[1])
    else:
        BFs = jnp.sum(weights, axis=1)
        n_effs = BFs**2 / jnp.sum(weights**2, axis=1)
        BFs /= weights.shape[1]
        logBFs = jnp.log(BFs)
        logn_effs = jnp.log(n_effs)
    return logBFs, logn_effs


@partial(jit, static_argnames=["log"])
def detection_efficiency(weights, Ninj, log=False):
    r"""Calculates the detection efficiency -- the expected fraction of sources detected from a population
    parameterized by :math:`\Lambda` -- estimated by importance sampling over the found injections from
    a fiducial population parameterized by :math:`\Lambda_\emptyset`:

    .. math::

        \mu = \int P(\mathrm{det}|\theta)p(\theta|\Lambda) d\theta \approx
        \frac{1}{N_\mathrm{found}}\sum_{i=1}^{N_\mathrm{found}} \frac{p(\theta_i|\Lambda)}{p(\theta_i | \Lambda_\emptyset)}

    with Monte Carlo integration over found
    injections, along with the effective sample size of the Monte Carlo integral.

    Parameters
    ----------
    weights : jax.DeviceArray
        JAX array of weights to integrate over. Expected size of (N_found_injections,).
    Ninj : int
        Total number of injections.
    log : bool, optional
        Flag to perform calculations in log probability. Interprets weights as log weights.
        This is slower but more numerically stable. Defaults to False.

    Returns
    -------
    jax.DeviceArray
        Array of log detection efficiency.
    jax.DeviceArray
        Array of log N_eff from Monte Carlo Integral.
    """
    if log:
        logweights = weights
        logmu = logsumexp(logweights) - jnp.log(Ninj)
        mu = jnp.exp(logmu)
        var = jnp.sum(jnp.exp(logweights) ** 2) / Ninj**2 - mu**2 / Ninj
        logn_eff = 2 * logmu - jnp.log(var)
    else:
        mu = jnp.sum(weights) / Ninj
        var = jnp.sum(weights**2) / Ninj**2 - mu**2 / Ninj
        logmu = jnp.log(mu)
        logn_eff = 2 * logmu - jnp.log(var)
    return logmu, logn_eff


class TotalVTCalculator(object):
    """
    TotalVTCalculator Object to calculate surveyed hyper-volume out to a maximum redshift
    """

    def __init__(self, maxz=1.9):
        """
        Args:
            maxz (float, optional): maximum redshift to integrate out to. Defaults to 1.9.
        """
        self.zmax = maxz
        self.zs = jnp.linspace(1e-4, maxz, 1000)
        self.dVdcs = jnp.array(Planck15.differential_comoving_volume(np.array(self.zs)).value * 4.0 * np.pi)

    def __call__(self, lamb=0):
        """Perform trapezoidal integration to get total hypervolume

        Parameters
        ----------
        lamb : int, optional
            Exponent of power-law rate evolution. Defaults to `0` (uniform with co-moving volume).

        Returns
        -------
        float
            Total hypervolume out to `z=zmax`. In units of `Gpc^3*yr`.
        """
        return trapezoid(self.dVdcs * jnp.power(1 + self.zs, lamb - 1), self.zs)


def hierarchical_likelihood(
    pe_weights,
    inj_weights,
    total_inj,
    Nobs,
    Tobs,
    categorical=False,
    marginal_qs=False,
    indv_weights=None,
    rngkey=None,
    pop_frac=None,
    surv_hypervolume_fct=TotalVTCalculator(),
    vtfct_kwargs={"lamb": 0},
    marginalize_selection=False,
    reconstruct_rate=True,
    min_neff_cut=True,
    posterior_predictive_check=False,
    param_names=None,
    pedata=None,
    injdata=None,
    m2min=3.0,
    m1min=5.0,
    mmax=100.0,
    log=False,
):
    """Performs the hierarchical likelihood calculation using importance sampling
    over injections and PE samples from each event's posterior samples assuming
    a fiducial prior density.

    Parameters
    ----------
    pe_weights : jax.DeviceArray
        Array of weights evaluated at PE samples to integrate over.
        If `log=True` this is expected to be the **log** of the weights.
        Expected size of `(N_events,N_samples)`.
    inj_weights : jax.DeviceArray
        Array of weights evaluated at found injections to integrate over.
        If `log=True` this is expected to be the **log** of the weights.
        Expected size of `(N_found_injections,)`.
    total_inj : int
        Total number of generated injections before cutting on found.
    Nobs : int
        Total number of observed events analyzing.
    Tobs : float
        Time spent observing to produce catalog (in yrs).
    categorical : bool, optional
        If `True` use latent categorical parameters to assign
        each event to one of many subpopulations. Defaults to `False`.
    marginal_qs : bool, optional
        TODO: add description!
    indv_weights : jax.DeviceArray
        TODO: add description!
    rngkey : jax.random.PRNGKey, optional
        RNG Key to be passed to sample categorical variable.
        Needed if `categorical=True`. Defaults to `None`.
    pop_frac : tuple of floats, optional
        Tuple of true astrophysical population fractions.
        Shape is number of categorical subpopulations, needs to sum to 1, and is
        needed if `categorical=True`. Defaults to `None`.
    surv_hypervolume_fct : callable, optional
        Callable function to calculate total VT (normalization of the redshift model).
        Defaults to `TotalVTCalculator()`.
    vtfct_kwargs : dict, optional
        Diction of args needed to call `surv_hypervolume_fct()`.
        Defaults to `{"lamb": 0}`.
    marginalize_selection : bool, optional
        Flag to marginalize over uncertainty in selection monte carlo integral.
        Defaults to `True`.
    reconstruct_rate : bool, optional
        Flag to reconstruct marginalize merger rate. Defaults to `True`.
    min_neff_cut : bool, optional
        Flag to use the `min_neff` cut on the likelihood ensuring Monte Carlo
        integrals converge. Defaults to `True`.
    posterior_predictive_check : bool, optional
        Flag to sample from the PE/injection data to perform posterior predictive check.
        Defaults to `False`.
    param_names : iterable, optional
        Parameters to sample for PPCs. Defaults to `None`.
    pedata : dict, optional
        Dictionary of PE data needed to perform PPCs. Defaults to `None`.
    injdata : dict, optional
        Dictionary of found injection data needed to perform PPCs. Defaults to `None`.
    m2min : float, optional
        Minimum mass for secondary components (solar masses). Defaults to `3.0`.
    m1min : float, optional
        Minimum mass for primary components (solar masses). Defaults to `5.0`.
    mmax : float, optional
        Maximum mass for primary components (solar masses). Defaults to `100.0`.
    log : bool, optional
        Flag to perform calculations in log space. Interprets weights as log weights.
        This is slower but more numerically stable. Defaults to `False`.

    Returns
    -------
    float
        Marginalized merger rate in units of `Gpc^-3 yr^-1`.
    """
    rate = None
    if categorical:
        with numpyro.plate("nObs", Nobs):
            Qs = numpyro.sample(
                "Qs",
                dist.Categorical(probs=jnp.stack(pop_frac, axis=-1)),
                rng_key=rngkey,
            ).reshape((-1, 1))
            selector = [jnp.equal(Qs, ii) for ii in range(len(pop_frac))]
            mix_pe_weights = jnp.select(selector, pe_weights)
            logBFs, logn_effs = per_event_log_bayes_factors(mix_pe_weights, log=log)
            if log:
                pe_weights = logsumexp(pe_weights, b=pop_frac, axis=0)
            else:
                pe_weights = jnp.sum(pop_frac * pe_weights, axis=0)
    else:
        logBFs, logn_effs = per_event_log_bayes_factors(pe_weights, log=log)

    log_det_eff, logn_eff_inj = detection_efficiency(inj_weights, total_inj, log=log)
    numpyro.deterministic("log_nEff_inj", logn_eff_inj)
    numpyro.deterministic("log_nEffs", logn_effs)
    numpyro.deterministic("logBFs", logBFs)
    numpyro.deterministic("detection_efficiency", jnp.exp(log_det_eff))
    if reconstruct_rate:
        total_vt = numpyro.deterministic("surveyed_hypervolume", surv_hypervolume_fct(**vtfct_kwargs) / 1.0e9 * Tobs)
        unscaled_rate = numpyro.sample("unscaled_rate", dist.Gamma(Nobs))
        rate = numpyro.deterministic("rate", unscaled_rate / jnp.exp(log_det_eff) / total_vt)
    if marginalize_selection:
        log_det_eff = jnp.where(
            jnp.greater_equal(logn_eff_inj, jnp.log(4 * Nobs)),
            log_det_eff - (3 + Nobs) / (2 * jnp.exp(logn_eff_inj)),
            jnp.inf,
        )
    sel = numpyro.deterministic(
        "selection_factor",
        jnp.where(jnp.isinf(log_det_eff), jnp.nan_to_num(-jnp.inf), -Nobs * log_det_eff),
    )
    sumlogBFs = numpyro.deterministic("sum_logBFs", jnp.sum(logBFs))
    log_l = sel + sumlogBFs
    log_l = numpyro.deterministic(
        "log_l",
        jnp.where(
            jnp.isnan(log_l),
            jnp.nan_to_num(-jnp.inf),
            jnp.nan_to_num(log_l),
        ),
    )

    # TODO: clean this up, make value of min_neff a function kwarg
    if min_neff_cut:
        min_n_effs = jnp.exp(jnp.min(jnp.nan_to_num(logn_effs)))
        log_l = numpyro.deterministic(
            "neff_less_Nobs",
            jnp.where(
                jnp.less_equal(min_n_effs, Nobs),
                jnp.nan_to_num(-jnp.inf),
                log_l,
            ),
        )

    numpyro.factor("log_likelihood", log_l)

    if posterior_predictive_check:
        if param_names is not None and injdata is not None and pedata is not None:
            if log:
                pe_weights = jnp.exp(pe_weights)
                inj_weights = jnp.exp(inj_weights)
            cond = jnp.less(pedata["mass_1"], m1min) | jnp.greater(pedata["mass_1"], mmax)
            pe_weights = jnp.where(
                cond | jnp.less(pedata["mass_1"] * pedata["mass_ratio"], m2min),
                0,
                pe_weights,
            )
            inj_weights = jnp.where(
                jnp.less(injdata["mass_1"], m1min)
                | jnp.greater(injdata["mass_1"], mmax)
                | jnp.less(injdata["mass_1"] * injdata["mass_ratio"], m2min),
                0,
                inj_weights,
            )
            for ev in range(Nobs):
                k = random.PRNGKey(ev)
                k1, k2 = random.split(k)
                obs_idx = random.choice(
                    k1,
                    pe_weights.shape[1],
                    p=pe_weights[ev, :] / jnp.sum(pe_weights[ev, :]),
                )

                if marginal_qs:
                    for i in range(len(indv_weights)):
                        numpyro.deterministic(f"cat_frac_subpop_{i + 1}_event_{ev}", indv_weights[i][ev, obs_idx] / pe_weights[ev, obs_idx])

                pred_idx = random.choice(k2, inj_weights.shape[0], p=inj_weights / jnp.sum(inj_weights))
                for p in param_names:
                    numpyro.deterministic(f"{p}_obs_event_{ev}", pedata[p][ev, obs_idx])
                    numpyro.deterministic(f"{p}_pred_event_{ev}", injdata[p][pred_idx])
    return rate


def construct_hierarchical_model(model_dict, prior_dict, min_neff_cut=True, marginalize_selection=False, posterior_predictive_check=True):
    source_param_names = [k for k in model_dict.keys()]
    hyper_params = {k: None for k in prior_dict.keys()}
    pop_models = {k: None for k in model_dict.keys()}

    if "redshift" in pop_models.keys():
        z_grid = jnp.linspace(1e-9, prior_dict["redshift_maximum"], 1000)

    def model(samps, injs, Ninj, Nobs, Tobs):
        for k, v in prior_dict.items():
            try:
                hyper_params[k] = numpyro.sample(k, v.dist(**v.params))
            except AttributeError:
                hyper_params[k] = v
        iid_mapping = {}
        for k, v in model_dict.items():
            if isinstance(v, PopMixtureModel):
                components = [
                    v.components[i](**{p: hyper_params[f"{k}_component_{i + 1}_{p}"] for p in v.component_params[i]})
                    for i in range(len(v.components))
                ]
                mixing_dist = v.mixing_dist(**{p: hyper_params[f"{k}_mixture_dist_{p}"] for p in v.mixing_params})
                pop_models[k] = v.model(mixing_dist, components)
            elif isinstance(v, PopModel):
                hps = {p: hyper_params[f"{k}_{p}"] for p in v.params}
                if k == "redshift":
                    hps["grid"] = z_grid
                pop_models[k] = v.model(**hps)
            elif isinstance(v, str):
                iid_mapping[v] = k
            else:
                raise ValueError(f"Unknown model type: {type(v)}:{v}")
        for shared_param, param in iid_mapping.items():
            pop_models[shared_param] = pop_models[param]

        inj_weights = jnp.sum(jnp.array([pop_models[k].log_prob(injs[k]) for k in source_param_names]), axis=0) - jnp.log(injs["prior"])
        pe_weights = jnp.sum(jnp.array([pop_models[k].log_prob(samps[k]) for k in source_param_names]), axis=0) - jnp.log(samps["prior"])

        hierarchical_likelihood(
            pe_weights,
            inj_weights,
            total_inj=Ninj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=lambda *_: pop_models["redshift"].norm,
            vtfct_kwargs={},
            marginalize_selection=marginalize_selection,
            min_neff_cut=min_neff_cut,
            posterior_predictive_check=posterior_predictive_check,
            pedata=samps,
            injdata=injs,
            param_names=source_param_names,
            m1min=2.0,
            m2min=2.0,
            mmax=100.0,
            log=True,
        )

    return model
