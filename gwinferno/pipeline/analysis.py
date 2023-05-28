import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy.cosmology import Planck15
from jax import jit
from jax import random
from jax.scipy.special import logsumexp
from numpyro.infer import SVI
from numpyro.infer import Trace_ELBO
from numpyro.infer import autoguide
from numpyro.optim import Adam


def find_map(rng_key, numpyro_model, *model_args, Niter=100, lr=0.01):
    """
    find_map Find the MAP estimate for a given NumPyro model using SVI with Adam optimizing the ELBO

    Args:
        rng_key (jax.random.PRNGKey): RNG Key to be passed to SVI.run().
        numpyro_model (callable): Python callable containing `numpyro.primitives`/
        Niter (int, optional): Number of iterations to run variational inference. Defaults to 100.
        lr (float, optional): learning rate used for Adam optimizer. Defaults to 0.01.

    Returns:
        SVIRunResult.params: parameters of the result of MAP optimization
    """
    guide = autoguide.AutoDelta(numpyro_model)
    optimizer = Adam(lr)
    svi = SVI(numpyro_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(rng_key, Niter, *model_args)
    return svi_results.params


@jit
def per_event_log_bayes_factor_log_neffs(weights):
    """
    per_event_log_bayes_factor_log_neffs calculate per event log BFs via montecarlo integrating.
    Also return effective samples size for convergence checks.

    Args:
        weights (jax.DeviceArray): JAX array of weights to integrate over. Expected size of (N_events,N_samples)

    Returns:
        (jax.DeviceArray,jax.DeviceArray): array of per event logBF, array of per event N_eff from Monte Carlo Integral
    """
    BFs = jnp.sum(weights, axis=1)
    n_effs = BFs**2 / jnp.sum(weights**2, axis=1)
    BFs /= weights.shape[1]
    return jnp.log(BFs), jnp.log(n_effs)


@jit
def mu_neff_injections(weights, Ninj):
    """
    mu_neff_injections calculates detection efficeny (mu) with monte carlo integration over found
    injections along with the integrals effective sample size

    Args:
        weights (jax.DeviceArray): JAX array of weights to integrate over. Expected size of (N_found_injections,)
        Ninj (int): total number of injections

    Returns:
        (jax.DeviceArray,jax.DeviceArray):  array of detection efficiency, array of N_eff from Monte Carlo Integral
    """
    mu = jnp.sum(weights) / Ninj
    var = jnp.sum(weights**2) / Ninj**2 - mu**2 / Ninj
    n_eff = mu**2 / var
    return mu, n_eff


@jit
def per_event_log_bayes_factor_log_neffs_log(logweights):
    """
    per_event_log_bayes_factor_log_neffs_log calculate per event log BFs via montecarlo integrating.
    Also return effective samples size for convergence checks. Performs calculation in log prob.

    Args:
        logweights (jax.DeviceArray): JAX array of logweights to integrate over. Expected size of (N_events,N_samples)

    Returns:
        (jax.DeviceArray,jax.DeviceArray): array of per event logBF, array of per event N_eff from Monte Carlo Integral
    """
    logBFs = logsumexp(logweights, axis=1)
    logn_effs = 2 * logBFs - logsumexp(2 * logweights, axis=1)
    logBFs -= jnp.log(logweights.shape[1])
    return logBFs, logn_effs


@jit
def logmu_logneff_injections_log(logweights, Ninj):
    """
    logmu_logneff_injections_log calculates log detection efficeny (log_mu) with monte carlo
    integration over found injections along with the integrals effective sample size.
    Performs calculation in log prob.

    Args:
        logweights (jax.DeviceArray): JAX array of log weights to integrate over. Expected size of (N_found_injections,)
        Ninj (int): total number of injections

    Returns:
        (jax.DeviceArray,jax.DeviceArray):  array of log detection efficiency, array of log N_eff from Monte Carlo Integral
    """
    logmu = logsumexp(logweights) - jnp.log(Ninj)
    logvar = logsumexp(
        jnp.array(
            [
                logsumexp(2 * logweights) - 2 * jnp.log(Ninj) ** 2,
                2 * logmu - jnp.log(Ninj),
            ]
        ),
        b=jnp.array([1.0, -1]),
    )
    logn_eff = 2 * logmu - logvar
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
        """
        __call__ perfrom trapezoidal integration to get total hypervolume

        Args:
            lamb (int, optional): exponent of power-law rate evolution.
            Defaults to 0 (uniform with co-moving volume).

        Returns:
            (float): total hypervolume out to z=zmax. In units of Gpc^3*yr
        """
        return jnp.trapz(self.dVdcs * jnp.power(1 + self.zs, lamb - 1), self.zs)


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
):
    """
    hierarchical_likelihood performs the hierarchical likeihood calculation be
        resampling over injections and pe samples from each event's indiviudally done analayses. f
        or reference see:
    Args:
        pe_weights (jax.DeviceArray): JAX array of weights evaluated at pe samples to integrate over.
            Expected size of (N_events,N_samples)
        inj_weights (jax.DeviceArray): JAX array of weights evaluated at found injections to integrate over.
            Expected size of (N_found_injections,)
        total_inj (int): total number of generated injections before cutting on found.
        Nobs (int): Total number of observed events analyzing
        Tobs (float): Time spent observing to produce catalog (in yrs)
        categorical (bool, optional): set to True if using categorical model. Defaults to False.
        marginal_qs (bool, optional):
        indv_weights (jax.DeviceArray):
        rngkey (jax.random.PRNGKey, optional): RNG Key to be passed to sample categorical variable.
            Needed if categorical=True. Defaults to None.
        pop_frac (tuple of floats, optional): Tuple of true astrophysical population fractions.
            Shape is number of categorical subpopulatons and needs to sum to 1 and is needed if categorical=True.
            Defaults to None.
        surv_hypervolume_fct (callable, optional): callable function to calculate total VT
            (normalization of the redshift model). Defaults to TotalVTCalculator().
        vtfct_kwargs (dict, optional): diction of args needed to call surv_hypervolume_fct().
            Defaults to {"lamb": 0}.
        marginalize_selection (bool, optional): Flag to marginalize over uncertainty in
            selection monte carlo integral. Defaults to True.
        reconstruct_rate (bool, optional): Flag to reconstruct marginalize merger rate. Defaults to True.
        min_neff_cut (bool, optional): flag to use the min_neff cut on the likelihood
            ensuring monte carlo integrals converge. Defaults to True.
        posterior_predictive_check (bool, optional): Flag to sample from the PE/injection data to p
            erform posterior predictive check. Defaults to False.
        param_names (iterable, optional): parameters to sample for PPCs. Defaults to None.
        pedata (dict, optional): diction of PE data needed to perform PPCs. Defaults to None.
        injdata (dict, optional): diction of found injection data needed to perform PPCs. Defaults to None.
        m2min (float, optional): mininmum mass for secondary components (solar masses). Defaults to 3.0.
        m1min (float, optional): mininmum mass for primary components (solar masses). Defaults to 6.5.
        mmax (float, optional): maximum mass for primary components (solar masses). Defaults to 100.0.
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
            logBFs, logn_effs = per_event_log_bayes_factor_log_neffs(mix_pe_weights)
            pe_weights = sum([p * pew for p, pew in zip(pop_frac, pe_weights)])
    else:
        logBFs, logn_effs = per_event_log_bayes_factor_log_neffs(pe_weights)

    vt_factor, n_eff_inj = mu_neff_injections(inj_weights, total_inj)
    numpyro.deterministic("log_nEff_inj", jnp.log(n_eff_inj))
    numpyro.deterministic("log_nEffs", logn_effs)
    numpyro.deterministic("logBFs", logBFs)
    numpyro.deterministic("detection_efficency", vt_factor)
    if reconstruct_rate:
        total_vt = numpyro.deterministic("surveyed_hypervolume", surv_hypervolume_fct(**vtfct_kwargs) / 1.0e9 * Tobs)
        unscaled_rate = numpyro.sample("unscaled_rate", dist.Gamma(Nobs))
        rate = numpyro.deterministic("rate", unscaled_rate / vt_factor / total_vt)
    if marginalize_selection:
        vt_factor = jnp.where(
            jnp.greater_equal(n_eff_inj, 4 * Nobs),
            vt_factor / jnp.exp((3 + Nobs) / 2 / n_eff_inj),
            jnp.inf,
        )
    sel = numpyro.deterministic(
        "selection_factor",
        jnp.where(jnp.isinf(vt_factor), jnp.nan_to_num(-jnp.inf), -Nobs * jnp.log(vt_factor)),
    )
    sumlogBFs = numpyro.deterministic("sum_logBFs", jnp.sum(logBFs))
    log_l = numpyro.deterministic("log_l", sel + sumlogBFs)
    if min_neff_cut:
        numpyro.factor(
            "log_likelihood",
            jnp.where(
                jnp.isnan(log_l) | jnp.less_equal(jnp.exp(jnp.min(logn_effs)), Nobs),
                jnp.nan_to_num(-jnp.inf),
                jnp.nan_to_num(log_l),
            ),
        )
    else:
        numpyro.factor(
            "log_likelihood",
            jnp.where(jnp.isnan(log_l), jnp.nan_to_num(-jnp.inf), jnp.nan_to_num(log_l)),
        )
    if posterior_predictive_check:
        if param_names is not None and injdata is not None and pedata is not None:
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
                        numpyro.deterministic(f"cat_frac_subpop_{i+1}_event_{ev}", indv_weights[i][ev, obs_idx] / pe_weights[ev, obs_idx])

                pred_idx = random.choice(k2, inj_weights.shape[0], p=inj_weights / jnp.sum(inj_weights))
                for p in param_names:
                    numpyro.deterministic(f"{p}_obs_event_{ev}", pedata[p][ev, obs_idx])
                    numpyro.deterministic(f"{p}_pred_event_{ev}", injdata[p][pred_idx])
    return rate


def hierarchical_likelihood_in_log(
    logpe_weights,
    loginj_weights,
    total_inj,
    Nobs,
    Tobs,
    surv_hypervolume_fct=TotalVTCalculator(),
    vtfct_kwargs={"lamb": 0},
    marginalize_selection=True,
    reconstruct_rate=True,
    min_neff_cut=True,
    posterior_predictive_check=False,
    param_names=None,
    pedata=None,
    injdata=None,
    m2min=3.0,
    m1min=6.5,
    mmax=100.0,
):
    """
    hierarchical_likelihood_log performs the hierarchical likeihood calculation in log probability
        by resampling over injections and pe samples from each event's indiviudally done analayses.
        for reference see:
    Args:
        logpe_weights (jax.DeviceArray): JAX array of log weights evaluated at pe samples to integrate over.
            Expected size of (N_events,N_samples)
        loginj_weights (jax.DeviceArray): JAX array of log weights evaluated at found injections to
            integrate over. Expected size of (N_found_injections,)
        total_inj (int): total number of generated injections before cutting on found.
        Nobs (int): Total number of observed events analyzing
        Tobs (float): Time spent observing to produce catalog (in yrs)
        surv_hypervolume_fct (callable, optional): callable function to calculate total
            VT (normalization of the redshift model). Defaults to TotalVTCalculator().
        vtfct_kwargs (dict, optional): diction of args needed to call
            surv_hypervolume_fct(). Defaults to {"lamb": 0}.
        marginalize_selection (bool, optional): Flag to marginalize over uncertainty in
            selection monte carlo integral. Defaults to True.
        reconstruct_rate (bool, optional): Flag to reconstruct marginalize
            merger rate. Defaults to True.
        min_neff_cut (bool, optional): flag to use the min_neff cut on the likelihood
            ensuring monte carlo integrals converge. Defaults to True.
        posterior_predictive_check (bool, optional): Flag to sample from the PE/injection data
            to perform posterior predictive check. Defaults to False.
        param_names (iterable, optional): parameters to sample for PPCs. Defaults to None.
        pedata (dict, optional): diction of PE data needed to perform PPCs. Defaults to None.
        injdata (dict, optional): diction of found injection data needed to perform PPCs. Defaults to None.
        m2min (float, optional): mininmum mass for secondary components (solar masses). Defaults to 3.0.
        m1min (float, optional): mininmum mass for primary components (solar masses). Defaults to 6.5.
        mmax (float, optional): maximum mass for primary components (solar masses). Defaults to 100.0.
    """
    logBFs, logn_effs = per_event_log_bayes_factor_log_neffs_log(logpe_weights)
    log_vt_factor, logn_eff_inj = logmu_logneff_injections_log(loginj_weights, total_inj)
    numpyro.deterministic("log_nEff_inj", logn_eff_inj)
    numpyro.deterministic("log_nEffs", logn_effs)
    numpyro.deterministic("logBFs", logBFs)
    vt_factor = numpyro.deterministic("detection_efficency", jnp.exp(log_vt_factor))
    if reconstruct_rate:
        total_vt = numpyro.deterministic("surveyed_hypervolume", surv_hypervolume_fct(**vtfct_kwargs) / 1.0e9 * Tobs)
        unscaled_rate = numpyro.sample("unscaled_rate", dist.Gamma(Nobs))
        numpyro.deterministic("rate", unscaled_rate / vt_factor / total_vt)
    if marginalize_selection:
        log_vt_factor = jnp.where(
            jnp.greater_equal(logn_eff_inj, jnp.log(4 * Nobs)),
            log_vt_factor - (3 + Nobs) / (2 * jnp.exp(logn_eff_inj)),
            jnp.inf,
        )
    sel = numpyro.deterministic(
        "selection_factor",
        jnp.where(jnp.isinf(log_vt_factor), jnp.nan_to_num(-jnp.inf), -Nobs * log_vt_factor),
    )
    sumlogBFs = numpyro.deterministic("sum_logBFs", jnp.sum(logBFs))
    log_l = numpyro.deterministic("log_l", sel + sumlogBFs)
    if min_neff_cut:
        numpyro.factor(
            "log_likelihood",
            jnp.where(
                jnp.isnan(log_l) | jnp.less_equal(jnp.exp(jnp.min(logn_effs)), 10),
                jnp.nan_to_num(-jnp.inf),
                jnp.nan_to_num(log_l),
            ),
        )
    else:
        numpyro.factor(
            "log_likelihood",
            jnp.where(jnp.isnan(log_l), jnp.nan_to_num(-jnp.inf), jnp.nan_to_num(log_l)),
        )
    if posterior_predictive_check:
        if param_names is not None and injdata is not None and pedata is not None:
            pe_weights = jnp.exp(logpe_weights)
            inj_weights = jnp.exp(loginj_weights)
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
                pred_idx = random.choice(k2, inj_weights.shape[0], p=inj_weights / jnp.sum(inj_weights))
                for p in param_names:
                    numpyro.deterministic(f"{p}_obs_event_{ev}", pedata[p][ev, obs_idx])
                    numpyro.deterministic(f"{p}_pred_event_{ev}", injdata[p][pred_idx])
