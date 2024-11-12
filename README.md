[![CI-Tests](https://github.com/FarrOutLab/GWInferno/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/FarrOutLab/GWInferno/actions/workflows/ci-tests.yml)
[![GitHub Doc Pages](https://github.com/FarrOutLab/GWInferno/actions/workflows/docs-gh-pages.yml/badge.svg)](https://github.com/FarrOutLab/GWInferno/actions/workflows/docs-gh-pages.yml)
[![codecov](https://codecov.io/gh/FarrOutLab/GWInferno/branch/main/graph/badge.svg?token=PLXM4211S3)](https://codecov.io/gh/FarrOutLab/GWInferno)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)


![logo_github_inferno](https://github.com/FarrOutLab/GWInferno/assets/80279129/39a9529e-9668-44ee-8319-2694abf6f629)


# Gravitational-Wave Hierarchical Inference with NumPyro

## Documentation

<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://farroutlab.github.io/GWInferno/)

</div>

## Installation

Clone the repository

```bash
git clone https://github.com/FarrOutLab/GWInferno.git
```

Recommended to use conda to set up your environment with python>=3.9:

### For CPU

For CPU usage only create an environment and install requirements and GWInferno with:

```bash
cd gwinferno
conda create -n gwinferno python=3.12
conda activate gwinferno
conda install -c conda-forge numpyro h5py 
pip install --upgrade pip
pip install -r pip_requirements.txt
python -m pip install .
```
### For GPU

To enable JAX access to CUDA enabled GPUs we need to specify specific versions to install. The following procedure will only work for Linux x86_64 and Linux aarch64; for other platforms see [Jax documentation](https://jax.readthedocs.io/en/latest/installation.html)

Jax recommends installing Nvidia CUDA and cuDNN with pip wheels. If you use local installations of CUDA and cuDNN, which could be the case for a remote cluster, then you'll need to install jax from the single CUDA wheel variant it offers. As of writing, this wheel is only compatible with CUDA >= 12.1 and cuDNN >= 9.1 < 10.0. See [JAX](https://github.com/google/jax) installation instructions for more details.

#### Installation process for CUDA installed via pip:

```bash
cd gwinferno
conda create -n gwinferno_gpu python=3.12
conda activate gwinferno_gpu
pip install --upgrade pip
```
Next, install CUDA and cuDNN pip wheels. See [Nvidia's CUDA quickstart guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#pip-wheels-linux) for the CUDA installation procedure and the [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#installing-cudnn-with-pip) for the cuDNN installation procedure. Once that has finished, continue with these steps:
```
pip install --upgrade "jax[cuda12]"
pip install numpyro[cuda]
pip install -r pip_requirements.txt
python -m pip install .
```

#### Installation process for locally installed CUDA and cuDNN:

Ensure [Nvidia CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html) are installed locally. 

```bash
cd gwinferno
conda create -n gwinferno_gpu python=3.12
conda activate gwinferno_gpu
pip install --upgrade pip
pip install --upgrade "jax[cuda12_local]"
pip install numpyro[cuda]
pip install -r pip_requirements.txt
python -m pip install .
```

## License 

Released under [MIT](/LICENSE.md) by [@FarrOutLab](https://github.com/FarrOutLab).

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
