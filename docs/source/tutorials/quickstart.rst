Quickstart
***********

We'll be using the BEST model by Kruschke as an example. We start by
creating some data

.. code-block::
    
    import numpy as np
    import pandas as pd
    from bayesian_models import BEST

    drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101)
    placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99)

    y1 = np.array(drug)
    y2 = np.array(placebo)
    df = pd.DataFrame(
        dict(value=np.r_[y1, y2], group=np.r_[["drug"] * len(drug), ["placebo"] * len(placebo)])
    )

Set up the model:

.. code-block::

    model = BEST()(df, "group")

Train the model:

.. code-block::
    
    model.fit()

Recover and examine results:

.. code-block::

    model.predict()