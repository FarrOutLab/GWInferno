=====================================================================
Quick Start
=====================================================================


Given a catalog of GW Posterior samples in standardized PESummary format, defined by catalog.json file to run population inference with GWInferno one must write a yaml file defining the desired model, hyperparameters, and other auxiliary configuration arguments. 

For example a simple config.yml file that defines a Truncated Powerlaw population model over primary masses (i.e. mass_1) we have:

.. code-block:: yaml

        # Run Label
        label: Truncated_Powerlaw_mass_1

        # Population Parameters, Models, HyperParameters, and Priors
        models:
            mass_1:
                model: gwinferno.numpyro_distributions.Powerlaw
                hyper_params:
                    alpha:
                    prior: numpyro.distributions.Normal
                    prior_params:
                        loc: 0.0
                        scale: 3.0
                    minimum:
                    prior: numpyro.distributions.Uniform
                    prior_params:
                        low: 2.0
                        high: 10.0
                    maximum:
                    prior: numpyro.distributions.Uniform
                    prior_params:
                        low: 50.0
                        high: 100.0

        # Sampler Configuration Args
        sampler_args:
            kernel: NUTS
            kernel_kwargs:
                dense_mass: true
        mcmc_kwargs:
            num_warmup: 500
            num_samples: 1500
            num_chains: 1

        # Data Configuration Args
        data_args:
            catalog_summary_json: /path/to/catalog/summary/file/catalog.json

With this file written and ready to go run to perform inference with a single command!

.. code-block:: bash

        gwinferno_run_from_config.py config.yml


Now defining models in this way may not be able to handle complex model declarations, that have correlated population distributions or mixtures 
models over components of different parameters. 

Currently, All 1D model distributions either single or mixture models (in one parameter, i.e. Powerlaw+Peak) are supported along with defining two 
population parameters to share a model when assuming they are IID (i.e. spin magnitudes or tilts usually assumed to be IID between binary components).

If you want to define a more complex model you can pass a path to a python file where this probabilistic numpyro model is defined and can be imported as `model`. 

For example one defines a python file declared model in the `config.yml` file as follows:

.. code-block:: yaml

        # Run Label
        label: Truncated_Powerlaw_mass_1

        # Population Parameters, Models, HyperParameters, and Priors
        models:
            python_file: gwinferno/pipeline/config_files/example_python_model.py

The requirements for this model definition within the python file include:
- Must at least take in PEsamples, Found injections, Number of Observations, Total number of injections, and total observing time (in yrs) as the initial 5 positional arguments of the model. 
- Define hyper-parameters and priors along with population distributions then interface with GWInferno's likleihood calculation with `gwinferno.pipeline.analysis.hierarchical_likelihood_in_log`
- Main input to this likelihood function is the weights calcaulated at all of the PE samples and Found injections
- Input weights need to have shapes of (Nobs, N_PEsamples) for PE samples weights where Nobs is the number of events in catalog and N_PEsamples is the nubmber of posteror samples each event contains
- Input weights for found injections need to be of shape (N_found,) where N_found is the total number of found injections remaining.

For example here is how one would define a numpyro probabilistic model in a python file to be passed to GWInferno's infrastructure

.. code-block:: python

        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        from gwinferno.numpyro_distributions import Powerlaw
        from gwinferno.numpyro_distributions import PowerlawRedshift
        from gwinferno.pipeline.analysis import hierarchical_likelihood_in_log


        def model(samps, injs, Ninj, Nobs, Tobs):
            alpha = numpyro.sample("alpha", dist.Normal(0, 3))
            beta = numpyro.sample("beta", dist.Normal(0, 3))
            mmin = numpyro.sample("mmin", dist.Uniform(2, 10))
            mmax = numpyro.sample("mmax", dist.Uniform(50, 100))
            lamb = numpyro.sample("lamb", dist.Normal(0, 3))

            m1dist = Powerlaw(alpha, minimum=mmin, maximum=mmax)
            qdist = Powerlaw(beta, minimum=0.02, maximum=1.0)
            zdist = PowerlawRedshift(lamb, maximum=2.3)

            inj_weights = m1dist.log_prob(injs["mass_1"]) + qdist.log_prob(injs["mass_ratio"]) + zdist.log_prob(injs["redshift"]) - jnp.log(injs["prior"])
            pe_weights = m1dist.log_prob(samps["mass_1"]) + qdist.log_prob(samps["mass_ratio"]) + zdist.log_prob(samps["redshift"]) - jnp.log(samps["prior"])

            def shvf():
                return zdist["redshift"].norm

            hierarchical_likelihood_in_log(
                pe_weights,
                inj_weights,
                total_inj=Ninj,
                Nobs=Nobs,
                Tobs=Tobs,
                surv_hypervolume_fct=shvf,
                vtfct_kwargs={},
                marginalize_selection=False,
                min_neff_cut=True,
                posterior_predictive_check=True,
                pedata=samps,
                injdata=injs,
                param_names=[
                    "mass_1",
                    "mass_ratio",
                    "redshift",
                ],
                m1min=mmin,
                m2min=mmin,
                mmax=mmax,
            )
