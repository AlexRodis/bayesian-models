How To Run BEST with Different Hyperparameters
***********************************************

In this tutorial we will see how the `BEST` model can be run with
non-default parameters. We begin with some setup:

.. code-block::

    from sklearn.datasets import load_iris
    from bayesian_models.models import BEST
    import pandas as pd
    X, y = load_iris(return_X_y=True, as_frame=True)
    names=load_iris().target_names
    y=y.replace({i:names[i] for i in range(len(names))})
    df = pd.concat([X,y], axis=1)

There are many parameters that could be changed, the most common ones would the the limits of the Uniform prior over the standard deviations and the derived quantities to be calculated. Most hyperparameters for the BEST model are stored as class level attributes.

.. code-block::

    # Select more diffuse priors
    BEST.std_upper = 1e3
    BEST.std_lower = 1e-3

See the documentation for more options. To calculate difference of standard deviations and the effect size, pass the flags to the object constructor:

.. code-block::

    obj = BEST(std_difference=True, effect_magnitude=True)(df, "target")
    # Run inference
    idata = obj.fit()

ROPE limits can be specified explicitly:

.. code-block::

    res_dict = obj.predict(var_names =['Δμ', 'Δσ'],  ropes=[(-.1,.1),
    (-.5,.5)], hdis=[.95, .90])