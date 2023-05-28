#!/bin/bash

conda create -n gwinferno python=3.9
conda activate gwinferno
pip install --upgrade pip
pip install numpyro[cpu]
pip install -r requirements.txt
python -m pip install -e .
conda deactivate