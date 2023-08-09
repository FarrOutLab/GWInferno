=====================================================================
Saving Data in `Popsummary <https://git.ligo.org/zoheyr-doctor/popsummary/-/tree/main/>`_ Standard Output
=====================================================================

After using GWInferno, you may wish to save your results in a form easy for others to read and access. This can be done quickly with `gwifnerno.postprocess.postprocess.PopSummaryWriteOut`. Before saving your results in this manner, make sure your posterior samples are located in a dictionary. This can easily be done by using `get_samples()` after sampling, as in the code snippet below:

.. code-block:: python
        kernel = NUTS(model)
        mcmc = MCMC(kernel)
        rng_key = random.PRNGKey(5)
        mcmc.run(rng_key)
        posteriors = mcmc.get_samples()

Rates :math:`\frac{dN}{d\theta}` (or population PDFs :math:`p(\theta|\Lambda)`) should also be located in a dictionary, along with whatever grids they were calculated over. 

In the below example, the posteriors and rates were already saved in h5 files using the `deepdish` package and are loaded in. 

.. code-block:: python
        from gwinferno.postprocess.postprocess import PopSummaryWriteOut

        path_to_rate_file = 'path-to-rate-file.h5'
        path_to_posterior_file = 'path-to-posterior-file.h5'

        posteriors = dd.io.load(path_to_posterior_file)
        print('posteriors loaded')

        ppds = dd.io.load(path_to_posterior_file)
        print('ppds loaded')

        #name of file to write data to
        file_path = 'name-of-popsummary-file.h5'

        #Metadata
        model_names = ['Power Law Primary Mass', 'Power Law Mass Ratio']
        hyperparameters = ['alpha', 'beta']
        event_params = ['mass_1', 'mass_ratio']
        event_names = ['GW150914','GW151012','GW151226', 'GW170104','GW170608']
        hyperparameter_descriptions = ['Slope of primary mass powerlaw distribution', 'Slope of mass ratio powerlaw distribution']
        waveforms = ['IRPhenomV2']

        #metadata can be specified during this initialization step, or added later with popsum.set_metadata() (see popsummary source code for more info on how to do this)
        popsum = PopSummaryWriteOut(
            file_path,
            hyperparameters,
            events = event_names,
            hyperparameter_descriptions = hyperparameter_descriptions,
            event_waveforms = waveforms,
            model_names = model_names
        )

        #save hyperparameter samples. `overwrite=True` flag will overwrite if there are already hypersamples saved in the file but `overwrite=False` will cause it to fail if there are already hypersamples saved in the file. 
        popsum.save_hypersamples(path_to_posterior_file, overwrite = True)
        popsum.save_
