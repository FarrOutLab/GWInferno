import deepdish as dd
import numpy as np
from popsummary.popresult import PopulationResult


class PopSummaryWriteOut(PopulationResult):
    """class that saves datasets of hypersamples, reweighed event and injection samples, and rates calculated on grids to a results file."""

    def __init__(
        self,
        file_name,
        hyperparameter_names,
        new_hyperparameter_names=None,
        hyperparameter_descriptions=[],
        hyperparameter_latex_labels=[],
        references=[],
        model_names=[],
        events=[],
        event_parameters=[],
        event_sample_IDs=[],
        event_waveforms=[],
    ):
        """
        Args:
            file_name (str): name of h5 file
            hyperparameter_names (_type_): list of hyperparameters as named in posterior datafile
            new_hyperparameter_names (_type_, optional): list of new hyperparemter names,
                if you wish to rename them to something more clear. Defaults to None.
                hyperparameter_descriptions (list, optional): list of hyperparameter descriptions. Defaults to [].
            hyperparameter_latex_labels (list, optional): list of latex labels for
                hyperparameters. Defaults to [].
            references (list, optional): list of references pointed to. Defaults to [].
            model_names (list, optional): list of population models used. Defaults to [].
            events (list, optional): list of events used. Defaults to [].
            event_parameters (list, optional): list of event-level parameter names (e.g. m1, m2,
                chi_eff) in corresponding
            order to reweighted_event_samples or rewighted_injections. Defaults to [].
            event_sample_IDs (list, optional): event_sample_IDs. Defaults to [].
            event_waveforms (list, optional): event_waveforms. Defaults to [].
        """
        self.old_hyperparameter_names = hyperparameter_names
        self.new_hyperparameter_names = new_hyperparameter_names if new_hyperparameter_names is not None else hyperparameter_names
        self.events = events
        self.event_parameters = event_parameters

        super().__init__(
            fname=file_name,
            hyperparameters=self.new_hyperparameter_names,
            hyperparameter_descriptions=hyperparameter_descriptions,
            hyperparameter_latex_labels=hyperparameter_latex_labels,
            references=references,
            model_names=model_names,
            events=events,
            event_waveforms=event_waveforms,
            event_sample_IDs=event_sample_IDs,
            event_parameters=event_parameters,
        )

    def save_hypersamples(
        self,
        path_to_file,
        group="posterior",
        overwrite=False,
    ):
        """saves hypersamples to results file

        Args:
            path_to_file (str): path to file containing dictionary of all posterior samples
                (can include reweighed pe, injections, etc.), obtained from mcmc.get_samples()
                numpyro method
            group (str): group to save samples to ('posterior' or 'prior')
            overwrite (bool): whether to overwrite existing dataset
        """
        posteriors = dd.io.load(path_to_file)
        hyperparameter_samples = []
        names = []
        for (i, hp) in enumerate(self.old_hyperparameter_names):
            x = posteriors[hp].transpose()
            if len(x.shape) > 1:
                for j in range(len(x)):
                    names.append(self.new_hyperparameter_names[i] + f"_{j+1}")
                    hyperparameter_samples.append(x[j])
            else:
                names.append(self.new_hyperparameter_names[i])
                hyperparameter_samples.append(x)

        self.set_metadata("hyperparameters", names, overwrite=True)
        hyperparameter_samples = np.array(hyperparameter_samples).transpose()
        self.set_hyperparameter_samples(hyperparameter_samples, overwrite=overwrite, group=group)

    def save_reweighed_event_and_injection_samples(
        self, path_to_file, event_names=None, params=None, overwrite=False, group="posterior", event_samples=True, injection_samples=True
    ):
        """saves rates reweighed event and injection samples to results file

        Args:
            path_to_file (str): path to file containing dictionary of all posterior samples
                (can include reweighed pe, injections, etc.). Obtained from numpyro's get_samples method
            event_names (list): list of events used. Defaults to None. Will return an error if events
                                in file metadata is not already specified.
            event_params (list): list of event-level parameter names (e.g. m1, m2, chi_eff)
                                in corresponding order to reweighted_event_samples, rewighted_injections
            overwrite (bool, optional): whether to overwrite existing dataset. Defaults to False.
            group (str, optional): group to save draws to ('posterior' or 'prior'). Defaults to 'posterior'.
            event_samples (bool, optional): whether to save the reweighed event samples. Defaults to True.
            injection_samples (bool, optional): wether to save the reweighed injection samples. Defaults to True.
        """
        if event_samples:
            if not event_names and self.get_metadata("events").size == 0:
                raise AssertionError("Please include a list of the reweighed events used with the `event_names` kwarg.")
            elif not event_names and self.get_metadata("events").size != 0:
                event_names = self.get_metadata("events")
            elif event_names and self.get_metadata("events").size == 0:
                self.set_metadata("events", event_names, overwrite=True)
            elif event_names and self.get_metadata("events").size != 0:
                try:
                    event_names == self.get_metadata("events")
                except ValueError:
                    print("ValueError: `event_names` kwarg does not match events in file metadata")

        if not params and self.get_metadata("event_parameters").size == 0:
            raise AssertionError("Please include a list of the event parameters that have been reweighed in the `params` kwarg.")
        elif not params and self.get_metadata("events").size != 0:
            params = self.get_metadata("event_parameters")
        elif params and self.get_metadata("event_parameters").size == 0:
            self.set_metadata("event_parameters", event_names, overwrite=True)
        elif params and self.get_metadata("event_parameters").size != 0:
            try:
                params == self.get_metadata("event_parameters")
            except ValueError:
                print("`event_names` kwarg does not match events in metadata")

        posteriors = dd.io.load(path_to_file)
        reweighed_posteriors = np.zeros((len(event_names), 1, posteriors[f"{params[0]}_obs_event_0"].shape[0], len(params)))
        reweighed_injections = np.zeros_like(reweighed_posteriors)
        for (i, param) in enumerate(params):
            for event in range(len(event_names)):
                reweighed_posteriors[event, 0, :, i] = posteriors[f"{param}_obs_event_{event}"]
                reweighed_injections[event, 0, :, i] = posteriors[f"{param}_pred_event_{event}"]

        if event_samples:
            self.set_reweighted_event_samples(reweighed_posteriors, overwrite=overwrite, group=group)
        if injection_samples:
            self.set_reweighted_injections(reweighed_injections, overwrite=overwrite, group=group)

    def save_rates_on_grids(self, path_to_file, grid_params, rate_names, type="median", overwrite=False, group="posterior"):
        """
        save rates on grids to results file. This method assumes each element of `grid_params`
        ('mass_1', 'mass_ratio', etc.) corresponds to a single rate dataset in `rate_names`. Ex:
        grid_params = ['mass_1', 'mass_ratio', 'a_1']
        rate_names = ['primary_mass_rate', 'mass_ratio_rate', 'primary_spin_magnitide_rate']

        for mixture models, like powerlaw+peak in primary mass, this would look like:
        grid_params = ['mass_1', 'mass_1', 'mass_ratio', ...]
        rate_names = ['primary_mass_powerlaw_rate', 'primary_mass_peak_rate',
        'mass_ratio_rate', ...]

        the dictionary pointed to with `path_to_file` should contain both the positions the rates
        were calculated over (named in `grid_params`) and the rates themselves
        (named in `rate_names`)

        Args:
            path_to_file (str): path to file that contains a dictionary of rates
                calulated on grids.
            grid_params (list): list of parameter names for which rates are calculated on
            rate_names (list): name of each rate dataset (e.g. primary_mass_rate, a_1_rate', etc)
            type (str): whether to save 'median' or 'mean' of rates.
            overwrite (bool, optional): whether to overwrite existing dataset. Defaults to False.
            group (str, optional): group to save draws to ('posterior' or 'prior').
                Defaults to 'posterior'.
        """
        rates = dd.io.load(path_to_file)
        if len(grid_params) != len(rate_names):
            raise AssertionError("`grid_params` must be same length as `rate_names`")
        for (gp, rs) in zip(grid_params, rate_names):
            if type == "median":
                m_rates = np.median(rates[rs], axis=0)
            elif type == "mean":
                m_rates = np.mean(rates[rs], axis=0)
            else:
                raise AssertionError("type must be `mean` or `median`")
            self.set_rates_on_grids(rs, gp, rates[gp].reshape(1, len(rates[gp])), m_rates, overwrite=overwrite, group=group)