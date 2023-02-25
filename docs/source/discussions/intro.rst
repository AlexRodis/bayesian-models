Introduction
*************


`bayesian_models` is a library born out of need to balance model reusability
with model flexibility. Written from a 'wet lab' scientific perspective, it
aims to make bayesian inference accessible to a wider audience with possibly
limited or even no expertise in the field. A lab scientist will typically
devote most of their time and resources to process of **collecting** the
data, since unlike many 'programmatic' datasets (be it clicks, economic
data, online text etc) scientific data are very expensive to produce, both
literly and figuratively. 

While `pymc` is a very powerfull and flexibly tool for bayesian inference, it
maintains its flexibility by defering to the user for the process of specifying
the model itself. Yet, model building is fairly sophisticated and specialized
field, often outside the field of expertise of most scientists. Since most
scientists can devote relatively litle time to the process of modeling itself
they tend to defer to 'premade' estimators, that apply a specific model to
some dataset.

A good example of this the `scikit-learn` library, which offers a wide selection
of common models for various tasks, through a simple, common API. This package
aims to provide similar functionality but from a bayesian perspective. Key
objectives is to provide as wide of selection of models as possible, while
minimizing the amount of specialized technical knowledge needed to understand
implementation details. At the same time, real world scientific data are often
messy and scientists often need to extract diverse information from datasets,
considering how expensive they are to generate. On the other hand, different
research areas will have difference requirements, such that the models need
to allow for more complex specification and to be able to adapt to the needs
of the analysis.

Scientific also has many special requirements, foremost among them 
**robust uncertainty estimates** and **interpretability**. The machine learning
community often disregards these two requirements focusing more of actionable
results. To a scientist however, understanding how a physical system operates
is just as important as making correct predictions. In fact, predictions are
often an afterthought, only serving as evidence in favor of the hypothesis
represented by the model itself.

There is of course a vast array of statistical models available, and not 
all can be reasonably implemented and maintained. Which models get selected
for implementation is something of an ad-hoc process. However this library
is written explicitly from the perspective of bayesian inference and is really
just a convenient wrapper around `pymc` itself. Some models and statistical
technique are explicitly outside the scope of this library, generally because
they are either frequentist or are "algorithmic" (i.e.  Support Vector Machines,
Partial Least Squares, etc).

In summary the key design goals of this project are to provide, prebuilt model
implementations, that can operate "out-of-the-box" with as litle tinkering
as possible, while enabling more specialized users to dive deeper and alter
aspects of the general model itself, according to their specific needs. Data
are cannot be assumed to derive from software sources, which may require
decoupling the data-generating process itself from the software. For example
many bayesian optimization packages couple the cost function evalutation
process to the software itself, which hinders application of this "model" to 
scientific experiments, since there the evalutation itself is a physical process.


Errors
-------

Bearing in mind the fact that this library does not assume much prior knowledge,
especially technical knowledge on the part of the user, informative error messages
are important. Several libraries, including `pymc` itself are pretty (in)famous
for their sometimes cryptic error messages.  Where possible, error messsages
should list the received problematic input or value and the expected value
itself. Where possible, underlying errors should be wrapped rather than be
allowed to bubble up to the user, with custom error types, where feasible.



