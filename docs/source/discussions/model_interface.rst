Model Interfaces
******************


Introduction
--------------

`bayesian_models` aims to provide "prebuilt" models that are usable by non
specialists, while maintaining enough flexibility to allow users to alter 
deeper implementation details if they have special requirements. The API
seen most commonly is typically a two or three stage process:

.. code-block::
    :caption: "A three stage approach"

    # Initialize the model
    
    obj = Model(...)
    
    # Train the model
    
    obj.fit(Xtrain,Ytrain)

    # Predict on some new data

    obj.predict(Xnew)

.. code-block:: 
    :caption: "A two stage approach"

    obj = Model(options, Xtest, Ytest)
    obj.predict(Xnew)

We've found this type of API is too tightly coupled to allow for the degree
of flexibility needed here. In some cases it can result in overly verbose
API's especially in the `__init__` method. Consider that not all models
aim to predict some measured quantity strictly speaking (e.g. group comparisons).
Hence `bayesian_models` implements a four stage API which aims to decouple
various stages of a model objects' life-cycle. This interface is specified
by the `BayesianModel` abstract base class, which all models inherit from.
These stages are as follows:

.. code-block:: 
    :caption: "The key model API"

    class BayesianModel:

        def __init__(self, ...):
            ...

        def __call__(self, Xtrain [Ytrain], ...):
            pass

        def fit(self, *args, sampler=pyms.sample, **kwargs):
            ...
        
        def predict(self, Xnew, ...):
            pass

This approach enables advanced users to alter tailor a model to their specific
needs while enabling rapid default uses as well i.e.:

.. code-block:: 
    :caption: "Common use case"

    from bayesian_models import BEST

    obj = BEST()(X,"var_name").fit(2000, chains=2)
    obj.predict(...)


Phase 1: Model Hyperparameter Specification
--------------------------------------------

The first stage of the models' life-cycle is the initialization of the
model object itself. This is distinct from the initialization of the
actual model (in a model abstract sense). Most models, i.e. neural networks
are really large famillies of models, defined by their various hyperparameters.
In this stage the exact structure of the model is specified in terms of its
hyperparameters, a specification that's independant of the actuall dataset
itself. The various options are set either as object or as class method.
The latter case is better suited for models where a user is likely to
select a specific model and stick with for multiple calls. The former is
better suited for cases where a user is more likely to experiment with
different variations of the same general model. Object should transparently
declare all attributes during this time even if they remain unset for until
some later stage, since object mutation is very error prone. This method
only **begins** the actuall initialization process, no computation graphs
are generated at this stage.


Stage 2: Model Intialization
-----------------------------

The next stage involves implementing the model by specifying the full
probability model in `pymc`. In this stage input (and possibly) output
data are included as destinct nodes in the tree. Data are always included
in `pymc.Data` containers to enable them to potentially be swapped out. All
models accept some sort of input data for training, while many also accept
output data. These nodes are `theano.sharedTensor`s and may be mutable, or not.
Most models are "predictive", i.e. the attempt to predict some output quantity
`Y` from some input information `X`. In most of these cases predicting on new
data can be achieved by simply swapping out the training inputs with the new
ones. However not all models conform to this general pattern. Some models have
specialized API's such as the `GaussianProcess`, which receives its inputs
via the `.prior` method, and generates its outputs via the `conditional` method.
Another exception are models which aren't exactly predictive, such as group
comparisons or oversampling.


Stage 3: Model Training
-------------------------

Once the model is fully specified, what follows is training, or a bit more
accuratly **inference**. While `pymc` offers essentially three different ways
to do this, point estimates optimization via `pymc.find_MAP`, full MCMC inference
via `pymc.sample` and Variational Inference approximations via `pymc.ADVI` (and
others), at present only full HMC is supported. MCMC is technically the supperior
approach, as it comes with asymptotic guarantees while the others are strict
estimates (but faster). These return completely different structures:
a dictionary, a collection of high rank tensors (as an `xarray.DataSet`) and
a dictionary, respectivelly. While it would be desirable to offer (potentially)
faster training via `VI` algorithms, this would somewhat complecates things
is isn't an immediate plan. `HMC` is a computationally expensive algorith
and several third party packages offer implementations potentially much faster
that `pymc`s default. These are available in recent versions of the library
(in `pymc.sampling`) generally, but come with other dependencies. The training
process itself is the crux of `pymc` and hence, it is essentially identicall
between all models.


Stage 4: outputs
-------------------

After inference is performed, the user usually expects some sort of conclusive
output. This may be predictions on new data but for varies greatly between models.
Hence this is by far the most implementation-specific stage. For statistical
comparisons, the output is a dictionary of dataframes, for predictive-style models
it a tensor (as an `xarray.DataArray`) containing probabilistic predictions but
other, more exotic cases exist. For example, for `BayesianOptimization` the entire
model is iterative, and the output is a Generator.

