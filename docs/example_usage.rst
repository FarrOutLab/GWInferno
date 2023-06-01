=====================================================================
Quick Start
=====================================================================


Given a catalog of GW Posterior samples in standardized PESummary format, defined by catalog.json file to run population inference with GWInferno one must write a yaml file defining the desired model, hyperparameters, and other auxiliary configuration arguments. 

For example a simple config.yml file that defines a Truncated Powerlaw population model over primary masses (i.e. mass_1) we have:

.. code-block:: yaml

        # Run Label
        label: Truncated_Powerlaw_mass_1

        # Population Parameters, Models, HyperParameters, and Priors
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