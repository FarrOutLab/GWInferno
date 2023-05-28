[![Python package](https://github.com/FarrOutLab/GWInferno/actions/workflows/python-package.yml/badge.svg)](https://github.com/FarrOutLab/GWInferno/actions/workflows/python-package.yml)

# GWInferno: Gravitational-Wave Hierarchical Inference with NumPyro

- [Documentation](https://gwinferno.readthedocs.io/en/latest/)

## Installation

Make sure to install the cuda compatible versions of jax and numpyro before installation if you want to use GPU computing. (see `bin/create_py39_gpu_env.sh`)

```bash
git clone https://github.com/FarrOutLab/gwinferno.git
cd gwinferno
pip install .
```

## Citation

If GWInferno is used please cite this [paper](https://arxiv.org/abs/2210.12834)

We make use of and build upon other open source software projects, please cite them if using GWInferno.

- [jax](https://github.com/google/jax)
- [numpyro](https://github.com/pyro-ppl/numpyro)
- [astropy](https://github.com/astropy/astropy)
- [gwpopulation](https://github.com/ColmTalbot/gwpopulation)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [arviz](https://github.com/arviz-devs/arviz)

## Authors

- Bruce Edelman -- bedelman@uoregon.edu
- Jaxen Godfrey -- jaxeng@uoregon.edu
- Ben Farr -- bfarr@uoregon.edu
