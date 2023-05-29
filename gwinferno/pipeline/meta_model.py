import jax.numpy as jnp
import numpyro

from .analysis import hierarchical_likelihood_in_log

NP_KERNEL_MAP = {"NUTS": numpyro.infer.NUTS, "HMC": numpyro.infer.HMC}


def construct_hierarchical_model(model_dict, prior_dict):
    source_param_names = [k for k in model_dict.keys()]
    hyper_params = {k: None for k in prior_dict.keys()}
    pop_models = {k: None for k in model_dict.keys()}

    def model(samps, injs, Ninj, Nobs, Tobs, z_grid, dVcdz_grid):
        for k, v in prior_dict.items():
            try:
                hyper_params[k] = numpyro.sample(k, v.dist(**v.params))
            except AttributeError:
                hyper_params[k] = v

        for k, v in model_dict.items():
            hps = {p: hyper_params[f"{k}_{p}"] for p in v.params}
            if k == "redshift":
                pop_models[k] = v.model(**hps, zgrid=z_grid, dVcdz=dVcdz_grid)
            else:
                pop_models[k] = v.model(**hps)

        inj_weights = jnp.sum(jnp.array([pop_models[k].log_prob(injs[k]) for k in source_param_names]), axis=0) - jnp.log(injs["prior"])
        pe_weights = jnp.sum(jnp.array([pop_models[k].log_prob(samps[k]) for k in source_param_names]), axis=0) - jnp.log(samps["prior"])

        def shvf(lamb):
            return pop_models["redshift"].norm

        hierarchical_likelihood_in_log(
            pe_weights,
            inj_weights,
            total_inj=Ninj,
            Nobs=Nobs,
            Tobs=Tobs,
            surv_hypervolume_fct=shvf,
            vtfct_kwargs={"lamb": hyper_params["redshift_lamb"]},
            marginalize_selection=False,
            min_neff_cut=True,
            posterior_predictive_check=True,
            pedata=samps,
            injdata=injs,
            param_names=source_param_names,
            m1min=hyper_params["mass_1_minimum"],
            m2min=2.0,
            mmax=hyper_params["mass_1_maximum"],
        )

    return model
