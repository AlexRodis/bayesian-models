# bayesian_models
A small library build on top of  `pymc` that implements many common models

## Introduction

Over the course of many projects, I've often had to implement many small variations of the same general model, in `pymc`. The `pymc` Probabilistic Programming Language (PPM) is powerfull library for bayesian inference, allowing the user to specify and perform inference on near arbitrary models. It's extremelly frexible and provides the user with the basic tools for model construction. Over the course of various projects, I've often had to write many different version of the same model, for example estimating group differences with the Kruschke's BEST model. For large datasets and complex problems, this can easily result in messy code, especially with multiple versions of the same model need to be accessible. Out of this need for reusability this project was born.

`bayesian_models` aims to implement `sklearn` style classes, representing general types of models a user may wish to specify. Since there is a very large variety of statistical models available, only some are included in this library in a somewhat ad-hoc  manner. The following models are planned for implementation:

* BEST (Bayesian Estimation Superceeds the t Test) := Statistical comparisons' between groups, analoguous to hypothesis testing (COMPLETED)

* BayesianNeuralNetworks := While neural networks are a far to heterogenuous category of models to be considered in full, common neural networks will be included (WIP)

* Gaussian Processes := GPs are a powerfull modeling tool, exhibiting very high degree of flexibility, whose unique features make it very appealing for scientific applications (INITIALIZED)

* Bayesian Optimization := Bayesian Optimization is a very powerfull optimization algorithm that seems uniquely well suited for many scientific applications. While several software packages are available in Python, these implementations seem to be written with programmatic applications in mind. Hence they all tightly couple the evalutation of the loss function, which makes their use difficult in cases where the actuall evaluation process is a physical process rather that a piece of code. Python provides a unique, high-level idiom well suited for this task, the Generator. (WIP)

* SEM (Structural Equation Models) := SEM models are exceedingly popular in certain fields like sociology and psychology, where they've been used with great success. Despite their many promising application to other scientific fields, they remain largely unknown (CONSIDERED)

* Other models considered GAMs, Splines

## Installation

Until official publication the library can be installed directly via the repo. First contact me to obtain access to the repository and ensure you've set up ssh authentication (required by git).

You can then install the library with the command:

```
git install pip install git+ssh://git@github.com/AlexRodis/bayesian_models.git
```
It is often desirable to run models with a GPU if available. At present, there are known issues with the `numpyro` dependency. Only these versions are supported:

```
jax==0.4.1
jaxlib==0.4.1
```
To install the developement version run:
```
pip install 'bayesian_models[dev]@ git+ssh://git@github.com/AlexRodis/bayesian_models.git@dev-main'
```

To attempt to install with GPU support run:
```
pip install 'bayesian_models[GPU]@ git+ssh://git@github.com/AlexRodis/bayesian_models.git'
```
Note the GPU version is unstable

You must also set the following environment variable prior to all other commands, including imports

```
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

These dependencies are only required with `pymc.sampling.jax.sample_numpyro_nuts` and if using the default options can be ignored

WARNING! The library is still in pre-alpha stage. Expect breaking changes and bugs
