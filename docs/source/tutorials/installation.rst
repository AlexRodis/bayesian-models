Installation
*************


`bayesian_models` can be installed with `pip` via its private repo. After
you obtain access to `the repo <https://github.com/AlexRodis/bayesian_models>`_
and `set up ssh authentication <https://docs.github.com/en/
authentication/connecting-to-github-with-ssh>`_ (required by git itself) you
can simply install with:

.. code-block::

    pit install pip install git+ssh://git@github.com/AlexRodis/bayesian_models.git

`bayesian_models` offers support for faster inference with `numpyro`. To
install with GPU support use:

.. code-block::

    pip install 'bayesian_models[GPU]@ git+ssh://git@github.com/AlexRodis/bayesian_models.git'

.. DANGER::

    The `numpyro` dependency, especially if used with GPU support requires
    setting up `cudatoolkit` and `cudnn` among other external dependencies.
    At present `pymc.sampling.jax.sample_numpyro_nuts` is unstable. Expect
    breaking changes and random errors

To install the development version of the library use:

.. code-block::
    
    pip install 'bayesian_models[dev]@ git+ssh://git@github.com/AlexRodis/bayesian_models.git@dev-main'