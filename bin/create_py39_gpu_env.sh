#!/bin/bash

conda create -n gwinferno python=3.9
conda activate gwinferno
pip install --upgrade pip
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
python -m pip install -e .
conda deactivate
