# bayesian-models
`bayeian-models` is a small library build on top of  `pymc` that
implements common statistical models

`bayesian_models` aims to implement `sklearn` style classes,
representing general types of models a user may wish to specify. Since
there is a very large variety of statistical models available, only some
are included in this library in a somewhat ad-hoc  manner. The following
models are planned for implementation:

* BEST (Bayesian Estimation Superceeds the t Test) := Statistical
  comparisons' between groups, analoguous to hypothesis testing
  (COMPLETED)

## Installation

`bayesian-models` can be installed with pip

```
pip install bayesian-models
```

Newer releases are first published to TestPyPI. They are installable as
follows

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple bayesian-models
```

To install from git:

```
pip install git+ssh://git@github.com/AlexRodis/bayesian-models.git
```

To install the developement version run:
```
pip install 'bayesian_models[dev]@ git+ssh://git@github.com/AlexRodis/bayesian_models.git@dev-main'
```

It is often desirable to run models with a GPU if available. At present,
there are known issues with the `numpyro` dependency. Only these
versions are supported:

```
jax==0.4.1
jaxlib==0.4.1
```


To attempt to install with GPU support run:

```
pip install 'bayesian_models[GPU]@git+ssh://git@github.com/AlexRodis/bayesian-models.git'
```

Note: the GPU version is unstable

You must also set the following environment variable prior to all other
commands, including imports

```
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

These dependencies are only required with
`pymc.sampling.jax.sample_numpyro_nuts` and if using the default options
can be ignored

