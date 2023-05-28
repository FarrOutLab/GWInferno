#!/bin/bash

conda create -n gwinferno_cpu python=3.10 --override-channels -c conda-forge numpyro 
conda init bash
conda activate gwinferno_cpu
pip install --upgrade pip
pip install -r pip_requirements.txt
python -m pip install -e .
conda env export > gwinferno_py310_cpu_env.yml
conda deactivate