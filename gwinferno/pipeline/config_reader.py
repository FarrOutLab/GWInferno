from importlib import import_module

import yaml


class PopModel(object):
    def __init__(self, model, params):
        self.model = model
        self.params = params


class PopPrior(object):
    def __init__(self, dist, params):
        self.dist = dist
        self.params = params


class PopMixtureModel(PopModel):
    def __init__(self, model, mix_dist, mix_params, components, component_params):
        self.model = model
        self.components = components
        self.mixing_dist = mix_dist
        self.mixing_params = mix_params
        self.component_params = component_params


def load_dist_from_string(dist):
    split_d = dist.split(".")
    module = ".".join(split_d[:-1])
    function = split_d[-1]
    return getattr(import_module(module), function)


class ConfigReader(object):
    def __init__(self):
        self.models = {}
        self.priors = {}
        self.sampling_params = []
        self.label = None
        self.data_args = None
        self.sampler_args = None

    def parse(self, yml_file):
        with open(yml_file, "r") as f:
            yml = yaml.safe_load(f)
        self.label = yml.pop("label")
        self.data_args = yml.pop("data_args")
        self.sampler_args = yml.pop("sampler_args")
        self.construct_model_and_prior_dicts(yml)

    def construct_model_and_prior_dicts(self, yml):
        for param in yml:
            if "Mixture" in yml[param]["model"]:
                self.add_mixture_model(param, yml[param])
            else:
                self.add_model(param, yml[param])

    def add_prior(self, key, subd):
        if "prior" in subd and "prior_params" in subd:
            self.priors[key] = PopPrior(load_dist_from_string(subd["prior"]), subd["prior_params"])
            self.sampling_params.append(key)
        elif "value" in subd:
            self.priors[key] = subd["value"]

    def add_model(self, param, subd):
        self.models[param] = PopModel(load_dist_from_string(subd["model"]), [p for p in subd["hyper_params"]])
        for hp in subd["hyper_params"]:
            self.add_prior(f"{param}_{hp}", subd["hyper_params"][hp])
        if "iid" in subd:
            self.add_iid_model(param, subd["iid"]["shared_parameter"])

    def add_iid_model(self, param, shared_param):
        self.models[shared_param] = param

    def add_mixture_model(self, param, subd):
        model = load_dist_from_string(subd["model"])
        mix_dist = load_dist_from_string(subd["mixture_dist"]["model"])
        mix_params = [p for p in subd["mixture_dist"]["hyper_params"]]
        N = len(subd["mixture_dist"]["hyper_params"][mix_params[0]]["prior_params"]["concentration"])
        for hp in mix_params:
            self.add_prior(f"{param}_mixture_dist_{hp}", subd["mixture_dist"][hp])
        components = []
        component_params = []
        for i in range(N):
            subd = subd[f"component_{i+1}"]
            components.append(load_dist_from_string(subd["model"]))
            component_params.append([p for p in subd["hyper_params"]])
            for hp in subd["hyper_params"]:
                self.add_prior(f"{param}_component_{i+1}_{hp}", subd["hyper_params"][hp])
        self.models[param] = PopMixtureModel(model, mix_dist, mix_params, components, component_params)
        if "iid" in subd:
            self.add_iid_model(param, subd["iid"]["shared_parameter"])
