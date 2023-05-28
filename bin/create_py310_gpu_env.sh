#!/bin/bash

conda create -n gwinferno_gpu --override-channels -c conda-forge -c nvidia python=3.10 jaxlib=*=*cuda* jax cuda-nvcc numpyro h5py
conda init bash
conda activate gwinferno_gpu
pip install --upgrade pip
pip install astropy tqdm xarray deepdish arviz funsor pre-commit
python -m pip install -e .
conda env export > gwinferno_py310_gpu_env.yml
conda deactivate