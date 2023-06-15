[![CI-Tests](https://github.com/FarrOutLab/GWInferno/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/FarrOutLab/GWInferno/actions/workflows/ci-tests.yml)
[![Docs](https://readthedocs.org/projects/gwinferno/badge/?version=latest)](https://gwinferno.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/FarrOutLab/GWInferno/branch/main/graph/badge.svg?token=PLXM4211S3)](https://codecov.io/gh/FarrOutLab/GWInferno)

# GWInferno: Gravitational-Wave Hierarchical Inference with NumPyro

- [Documentation](https://gwinferno.readthedocs.io/en/latest/)

## Installation

Clone the repository

```bash
git clone https://github.com/FarrOutLab/GWInferno.git
```

Recommended to use conda to set up your environment with python>=3.9:

For CPU usage only create an environment and install requirements and GWInferno with:

```bash
cd gwinferno
conda create -n gwinferno python=3.10
conda activate gwinferno
conda install -c conda-forge numpyro h5py 
pip install --upgrade pip
pip install -r pip_requirements.txt
python -m pip install .
```

To enable JAX access to CUDA enabled GPUs we need to specify specific versions to install (See [JAX](https://github.com/google/jax) installation instructions for more details). For a GPU enabled environment use:

```bash
cd gwinferno
conda create -n gwinferno_gpu python=3.10
conda activate gwinferno_gpu
conda install -c nvidia -c conda-forge jaxlib=*=*cuda* jax cuda-nvcc numpyro h5py
pip install --upgrade pip
pip install -r pip_requirements.txt
python -m pip install .
```

## Quick Start
Given a catalog of GW Posterior samples in standardized PESummary format, defined by catalog.json file to run population inference with GWInferno one must write a yaml file defining the desired model, hyperparameters, and other auxiliary configuration arguments. 

For example a simple config.yml file that defines a Truncated Powerlaw population model over primary masses (i.e. mass_1) we have:

```yaml
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
```

with this file written and ready to go run to perform inference!

```bash
gwinferno_run_from_config.py config.yml
```

## Citation

If GWInferno is used please cite this [paper](https://arxiv.org/abs/2210.12834)

We make use of and build upon other open source software projects, please cite them if using GWInferno.

- [jax](https://github.com/google/jax)
- [numpyro](https://github.com/pyro-ppl/numpyro)
- [gwpopulation](https://github.com/ColmTalbot/gwpopulation)

## Authors

- Bruce Edelman -- bedelman@uoregon.edu
- Jaxen Godfrey -- jaxeng@uoregon.edu
- Gino Carrillo -- gcarril3@uoregon.edu
- Ben Farr -- bfarr@uoregon.edu
