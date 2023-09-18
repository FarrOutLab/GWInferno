=====================================================================
Welcome to GWInferno's documentation!
=====================================================================

GWInferno: Gravitational-Wave Hierarchical Inference with NumPyro
---------------------------------------------------------------------

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    installation
    example_usage

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:

    Using Basis Splines
    Saving Data with PopSummary

---------------------------------------------------------------------
API:
---------------------------------------------------------------------
.. automodule:: gwinferno
    :members:

.. autosummary::
    :toctree: api
    :template: custom-module-template.rst
    :caption: API:
    :recursive:

    cosmology
    distributions
    interpolation
    models
    numpyro_distributions
    parameter_estimation
    pipeline
    postprocess
    preprocess


Citation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If GWInferno is used please cite this `paper <https://arxiv.org/abs/2210.12834>`_.

We make use of and build upon other open source software projects, please cite them if using GWInferno.

* `jax <https://github.com/google/jax>`_
* `numpyro <https://github.com/pyro-ppl/numpyro>`_
* `gwpopulation <https://github.com/ColmTalbot/gwpopulation>`_

Authors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Bruce Edelman -- bedelman@uoregon.edu
* Jaxen Godfrey -- jaxeng@uoregon.edu
* Gino Carrillo -- gcarrilo3@uoregon.edu
* Ben Farr -- bfarr@uoregon.edu
