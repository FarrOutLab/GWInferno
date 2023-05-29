from importlib import import_module

import yaml


class PopModel(object):
    def __init__(self, model=None, params=[]):
        self.model = model
        self.params = params


class PopPrior(object):
    def __init__(self, dist, params={}):
        self.dist = dist
        self.params = params


def load_dist_from_string(dist):
    split_d = dist.split(".")
    module = ".".join(split_d[:-1])
    function = split_d[-1]
    return getattr(import_module(module), function)


def load_config_from_yaml(yml_file):
    with open(yml_file, "r") as f:
        yml = yaml.safe_load(f)
    model_dict = {}
    prior_dict = {}
    sampling_params = []
    for param in yml.keys():
        if param in ["data_args", "sampler_args", "label"]:
            continue
        for hp in yml[param]["hyper_params"]:
            if "prior" in yml[param]["hyper_params"][hp] and "prior_params" in yml[param]["hyper_params"][hp]:
                prior_dict[f"{param}_{hp}"] = PopPrior(
                    load_dist_from_string(yml[param]["hyper_params"][hp]["prior"]), yml[param]["hyper_params"][hp]["prior_params"]
                )
                sampling_params.append(f"{param}_{hp}")
            elif "value" in yml[param]["hyper_params"][hp]:
                prior_dict[f"{param}_{hp}"] = yml[param]["hyper_params"][hp]["value"]
        model_dict[param] = PopModel(load_dist_from_string(yml[param]["model"]), [p for p in yml[param]["hyper_params"]])
    return model_dict, prior_dict, yml["data_args"], yml["sampler_args"], sampling_params, yml["label"]
