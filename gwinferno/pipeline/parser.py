"""
a module that stores tools for parsing CLI arguments and config files for analysis pipelines
"""

import sys
from argparse import ArgumentParser
from importlib import import_module

import jax.numpy as jnp
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


def load_model_from_python_file(path):
    fn = path.split("/")[-1]
    direct = path.replace(f"/{fn}", "")
    sys.path.append(direct)
    return getattr(import_module(fn.replace(".py", "")), "model")


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
        self.outdir = None
        self.data_args = None
        self.sampler_args = None

    def parse(self, yml_file):
        with open(yml_file, "r") as f:
            yml = yaml.safe_load(f)
        self.label = yml.pop("label", "label")
        self.outdir = yml.pop("outdir", "./")
        self.data_conf = yml.pop("data", {})
        self.sampler_conf = yml.pop("sampler", {})
        self.likelihood_kwargs = yml.pop("likelihood", {})
        self.construct_model_and_prior_dicts(yml["models"])

    def construct_model_and_prior_dicts(self, yml):
        if "python_file" in yml:
            self.models["file_path"] = yml["python_file"]
        else:
            for param in yml:
                if "Mixture" in yml[param]["model"]:
                    self.add_mixture_model(param, yml[param])
                else:
                    self.add_model(param, yml[param])

    def add_prior(self, key, subd):
        if "prior" in subd and "prior_params" in subd:
            for k in subd["prior_params"]:
                if type(subd["prior_params"][k]) is list:
                    subd["prior_params"][k] = jnp.array(subd["prior_params"][k])
            self.priors[key] = PopPrior(load_dist_from_string(subd["prior"]), subd["prior_params"])
            self.sampling_params.append(key)
        elif "value" in subd:
            if type(subd["value"]) is list:
                self.priors[key] = jnp.array(subd["value"])
            else:
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
            self.add_prior(f"{param}_mixture_dist_{hp}", subd["mixture_dist"]["hyper_params"][hp])
        components = []
        component_params = []
        for i in range(N):
            name = f"component_{i + 1}"
            components.append(load_dist_from_string(subd[name]["model"]))
            component_params.append([p for p in subd[name]["hyper_params"]])
            for hp in subd[name]["hyper_params"]:
                self.add_prior(f"{param}_component_{i + 1}_{hp}", subd[name]["hyper_params"][hp])
        self.models[param] = PopMixtureModel(model, mix_dist, mix_params, components, component_params)
        if "iid" in subd[name]:
            self.add_iid_model(param, subd[name]["iid"]["shared_parameter"])


def load_base_parser():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/home/bruce.edelman/projects/GWTC3_allevents/")
    parser.add_argument(
        "--inj-file",
        type=str,
        default="/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5",
    )
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--mmin", type=float, default=3.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    return parser
