=====================================================================
Installation
=====================================================================


Clone the repository

.. code-block:: bash

        git clone https://github.com/FarrOutLab/GWInferno.git
        cd gwinferno

Recommended to use conda to set up your environment with python versions newer than at least 3.9:

For CPU usage only create an environment and install requirements and GWInferno with:

.. code-block:: bash

        cd gwinferno
        conda create -n gwinferno python=3.10
        conda activate gwinferno
        conda install -c conda-forge numpyro h5py 
        pip install --upgrade pip
        pip install -r pip_requirements.txt
        python -m pip install .

To enable JAX access to CUDA enabled GPUs we need to specify specific versions to install (See `JAX <https://github.com/google/jax>`_ installation instructions for more details). For a GPU enabled environment use:

.. code-block:: bash

        cd gwinferno
        conda create -n gwinferno_gpu python=3.10
        conda activate gwinferno_gpu
        conda install -c nvidia -c conda-forge jaxlib=*=*cuda* jax cuda-nvcc numpyro h5py
        pip install --upgrade pip
        pip install -r pip_requirements.txt
        python -m pip install .
