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
    y = y.replace({
    i:names[i] for i in range(len(names))
    }).to_frame()
    df = pd.concat([X,y], axis=1)
    df
                sepal length (cm)  sepal width (cm)  ...  petal width (cm)   target
        0                  5.1               3.5  ...               0.2     setosa
        1                  4.9               3.0  ...               0.2     setosa
        2                  4.7               3.2  ...               0.2     setosa
        3                  4.6               3.1  ...               0.2     setosa
        4                  5.0               3.6  ...               0.2     setosa
        ..                 ...               ...  ...               ...        ...
        145                6.7               3.0  ...               2.3  virginica
        146                6.3               2.5  ...               1.9  virginica
        147                6.5               3.0  ...               2.0  virginica
        148                6.2               3.4  ...               2.3  virginica
        149                5.9               3.0  ...               1.8  virginica

    # Initialize, supply the data and variable to group by
    obj = BEST()(df, 'target')
    # Perform inference
    obj.fit()
    # Deduce the differences
    obj.predict()['Δμ']
    Δμ(setosa, versicolor)  sepal length (cm) -0.930  0.196  ...    1.0 Indeterminate
                            sepal width (cm)   0.653  0.199  ...    1.0  Indeterminate
                            petal length (cm) -2.800  0.192  ...    1.0  Indeterminate
                            petal width (cm)  -1.076  0.182  ...    1.0  Indeterminate
    Δμ(setosa, virginica)   sepal length (cm) -1.584  0.196  ...    1.0  Indeterminate
                            sepal width (cm)   0.454  0.199  ...    1.0  Indeterminate
                            petal length (cm) -4.088  0.198  ...    1.0  Indeterminate
                            petal width (cm)  -1.778  0.182  ...    1.0  Indeterminate
    Δμ(versicolor, virginica) sepal length (cm) -0.654  0.202  ...    1.0  Indeterminate
                            sepal width (cm)  -0.199  0.198  ...    1.0  Indeterminate
                            petal length (cm) -1.288  0.203  ...    1.0  Indeterminate
                            petal width (cm)  -0.702  0.194  ...    1.0  Indeterminate