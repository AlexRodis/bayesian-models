Bayesian Estimation Superceeds the t Test (BEST)
*************************************************

The BEST model (*Kruschke, 2014*) is a Bayesian model used to estimate
the differences between two (or more) groups, with respect to one (or
more) variables. We'll apply this model to the well known iris classification
dataset.

.. code-block::
    
    from sklearn.datasets import load_iris
    import pandas as pd
    from bayesian_models import BEST

    # Collect the data
    X, y = load_iris(return_X_y=True, as_frame=True)
    names = load_iris().target_names
    y.replace({
    i:names[i] for i in range(len(names))
    }).to_frame().T
    df = pd.concat([X,y], axis=1)

    # Initialize, supply the data and variable to group by
    obj = BEST()(df, 'target')
    # Perform inference
    obj.fit()
    # Deduce the differences
    obj.predict()
