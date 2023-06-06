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
            new_hyperparameter_names (_type_, optional): list of new hyperparemter names, if you wish to rename them to something more clear. Defaults to None.
            hyperparameter_descriptions (list, optional): list of hyperparameter descriptions. Defaults to [].
            hyperparameter_latex_labels (list, optional): list of latex labels for hyperparameters. Defaults to [].
            references (list, optional): list of references pointed to. Defaults to [].
            model_names (list, optional): list of population models used. Defaults to [].
            events (list, optional): list of events used. Defaults to [].
            event_parameters (list, optional): list of event-level parameter names (e.g. m1, m2, chi_eff) in corresponding
            order to reweighted_event_samples or rewighted_injections. Defaults to [].
            event_sample_IDs (list, optional): event_sample_IDs. Defaults to [].
            event_waveforms (list, optional): event_waveforms. Defaults to [].
        """
        self.old_hyperparameter_names = hyperparameter_names
        self.new_hyperparameter_names = new_hyperparameter_names if new_hyperparameter_names is not None else hyperparameter_names

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
            path_to_file (str): path to file containing dictionary of all posterior samples (can include reweighed pe, injections, etc.), obtained from mcmc.get_samples() numpyro method
            group (str): group to save samples to ('posterior' or 'prior')
            overwrite (bool): whether to overwrite existing dataset
        """
        posteriors = dd.io.load(path_to_file)
        l = []
        hyperparameter_samples = []
        for (i, hp) in enumerate(self.old_hyperparameter_names):
            x = posteriors[hp].transpose()
            if len(x.shape) > 1:
                for j in range(len(x)):
                    l.append(x[j])
                    hyperparameter_samples.append(self.new_hyperparameter_names[i] + f"_{j+1}")
            else:
                l.append(x)
                hyperparameter_samples.append(self.new_hyperparameter_names[i])

        hyperparameter_samples = np.array(l)

        self.set_hyperparameter_samples(hyperparameter_samples, overwrite=overwrite, group=group)

    def save_reweighed_event_and_injection_samples(
        self, path_to_file, event_names, event_params, overwrite=False, group="posterior", events=True, injections=True
    ):
        """saves rates reweighed event and injection samples to results file

        Args:
            path_to_file (str): path to file containing dictionary of all posterior samples (can include reweighed pe, injections, etc.), obtained from numpyor's MCMC.get_samples method
            event_names (list): list of events used
            event_params (list): list of event-level parameter names (e.g. m1, m2, chi_eff) in corresponding order to reweighted_event_samples, rewighted_injections
            overwrite (bool, optional): whether to overwrite existing dataset. Defaults to False.
            group (str, optional): group to save draws to ('posterior' or 'prior'). Defaults to 'posterior'.
            events (bool, optional): whether to save the reweighed event samples. Defaults to True.
            injections (bool, optional): wether to save the reweighed injection samples. Defaults to True.
        """
        if self.get_metadata("events").size == 0:
            self.set_metadata("events", event_names, overwrite=True)
        if self.get_metadata("event_parameters").size == 0:
            self.set_metadata("event_parameters", event_params, overwrite=True)

        posteriors = dd.io.load(path_to_file)
        reweighed_posteriors = np.zeros((len(event_names), 1, posteriors[f"{event_params[0]}_obs_event_0"].shape[0], len(event_params)))
        reweighed_injections = np.zeros_like(reweighed_posteriors)
        for (i, param) in enumerate(event_params):
            for event in range(len(event_names)):
                reweighed_posteriors[event, 0, :, i] = posteriors[f"{param}_obs_event_{event}"]
                reweighed_injections[event, 0, :, i] = posteriors[f"{param}_pred_event_{event}"]

        if events:
            self.set_reweighted_event_samples(reweighed_posteriors, overwrite=overwrite, group=group)
        if injections:
            self.set_reweighted_injections(reweighed_injections, overwrite=overwrite, group=group)

    def save_rates_on_grids(self, path_to_file, grid_params, rate_names, overwrite=False, group="posterior"):
        """
        save rates on grids to results file. This method assumes each element of `grid_params` ('mass_1', 'mass_ratio', etc.) corresponds to a single rate dataset in `rate_names`. Ex:
        grid_params = ['mass_1', 'mass_ratio', 'a_1']
        rate_names = ['primary_mass_rate', 'mass_ratio_rate', 'primary_spin_magntide_rate']

        for mixture models, like powerlaw+peak in primary mass, this would look like:
        grid_params = ['mass_1', 'mass_1', 'mass_ratio', ...]
        rate_names = ['primary_mass_powerlaw_rate', 'primary_mass_peak_rate', 'mass_ratio_rate', ...]

        the dictionary pointed to with `path_to_file` should contain both the positions the rates were calculated over (named in `grid_params`) and the rates themselves (named in `rate_names`)

        Args:
            path_to_file (str): path to file that contains a dictionary of rates calulated on grids.
            grid_params (list): list of parameter names for which rates are calculated on
            rate_names (list): name of each rate dataset (e.g. primary_mass_rate, a_1_rate', etc)
            overwrite (bool, optional): whether to overwrite existing dataset. Defaults to False.
            group (str, optional): group to save draws to ('posterior' or 'prior'). Defaults to 'posterior'.
        """
        rates = dd.io.load(path_to_file)
        if len(grid_params) != len(rate_names):
            raise AssertionError("`grid_params` must be same length as `rate_names`")
        for (gp, rs) in zip(grid_params, rate_names):
            self.set_rates_on_grids(rs, gp, rates[gp], rates[rs], overwrite=overwrite, group=group)
