=====================================================================
Installation
=====================================================================


Clone the repository

.. code-block:: bash

        git clone https://github.com/FarrOutLab/GWInferno.git
        cd gwinferno

It is recommended to use conda to set up your environment with python versions newer than at least 3.12:

---------------------------------------------------------------------
For CPU
---------------------------------------------------------------------

For CPU usage only, create an environment and install requirements and GWInferno with:

.. code-block:: bash

        cd gwinferno
        conda create -n gwinferno python=3.12
        conda activate gwinferno
        conda install -c conda-forge numpyro h5py 
        pip install --upgrade pip
        pip install -r pip_requirements.txt
        python -m pip install .

---------------------------------------------------------------------
For GPU
---------------------------------------------------------------------

To enable JAX access to CUDA enabled GPUs, we need to specify specific versions to install. The following procedure will only work for Linux x86_64 and Linux aarch64; for other platforms see `Jax documentation <https://jax.readthedocs.io/en/latest/installation.html>`_.

Jax recommends installing Nvidia CUDA and cuDNN with pip wheels. If you use local installations of CUDA and cuDNN, which could be the case for a remote cluster, then you'll need to install jax from the single CUDA wheel variant it offers. As of writing, this wheel is only compatible with CUDA >= 12.1 and cuDNN >= 9.1 < 10.0. See `JAX <https://github.com/google/jax>`_ installation instructions for more details.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installation process for CUDA installed via pip:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

        cd gwinferno
        conda create -n gwinferno_gpu python=3.12
        conda activate gwinferno_gpu
        pip install --upgrade pip

Next, install CUDA and cuDNN pip wheels. See `Nvidia's CUDA quickstart guide <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#pip-wheels-linux>`_ for the CUDA installation procedure and the `cuDNN documentation <https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#installing-cudnn-with-pip>`_ for the cuDNN installation procedure. Once that has finished, continue with these steps:

.. code-block:: bash

        pip install --upgrade "jax[cuda12]"
        pip install numpyro[cuda]
        pip install -r pip_requirements.txt
        python -m pip install .

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installation process for locally installed CUDA and cuDNN:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure `Nvidia CUDA <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux>`_ and `cuDNN <https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html>`_ are installed locally.

.. code-block:: bash

        cd gwinferno
        conda create -n gwinferno_gpu python=3.12
        conda activate gwinferno_gpu
        pip install --upgrade pip
        pip install --upgrade "jax[cuda12_local]"
        pip install numpyro[cuda]
        pip install -r pip_requirements.txt
        python -m pip install .
