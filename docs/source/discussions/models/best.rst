Bayesian Estimation Superceeds The t Test (BEST)
*************************************************

This short introduction is adapted and heavily based on the one from the
official `pymc` documentation `found here <https://docs.pymc.io/en/v3/
pymc-examples/examples/case_studies/BEST.html>`_.


Introduction
-------------

A common problem in statistical inference is deciding if two or more groups
are different with respect to some measured quantity of intrest or not. Making
this inferece is complicated somewhat, by the fact that real data are "contaminated"
with some amount of randomness that hinders attempts based on direct differences
in the data. The *de facto* method and standard for addressing these questions
is **ANOVA** (*One Way Analysis Of Variance*) and the **t Test** - more generally
**hypothesis testing**. These proccedures typically involve expressing a **null
hypothesis** (:math:`$H_0$`) which usually declares that there are no differences, and
an **alternative hypothesis** (:math:`$H_1$``) which typically states some differences
exist. A **test statistic** is then formulated, which is a quantity that determines
if the observed data (in terms of their distribution) are sufficiently plausible
under the given hypothesis.

Unfortunatelly, it can be quit difficult to use hypothesis tests correctly. There
are several subjective choices involved in the proccedure such as the test statistic
and the hypothesis. These are usually abstracted away form the user and a kind
of "default" is used, based on habit and tradition rather than justification
on the problem at hand. Moreover the justification for these proccedures are
are seen as questionable by some, as they are essentially based on maximization
of the repeatability of the experiment - but surely, we should care about the
outcomes of individual experiments as well. More over these proccedures frequently
fail to accuratly describe reality, for they do not generally contain any notion
of the extent or size of the difference. Two different datasets with major differences
in extent of they differences may for example give the same P-value. The interpretation
of their results is also rather difficult and error prone, especially for non
experts. For example if the P-value does not exceed some arbitrarily selected 
threshold (often 95% by tradition) this **does not** mean the null hypothesis
is upheld - indeed the null hypothesis can only be rejected, never upheld.


A fundamentally more informative approach is one based on **estimation** rather
than **testing**, and was proposed by *Kruschke, John. (2012) Bayesian estimation 
supersedes the t-test. Journal of Experimental Psychology: General.* According
to this approach complete distributions are fit to the data in each group, which
enables calculation of any Deterministic derived quantity of intrest, such as
difference of means, or difference of standard deviations.

**[DIAGRAMS  HERE]**

We'll use a famous example from Kruschke to illustrate this model. Suppose
we are testing a "smart drug" that aims to make people smarter. We choose to
use IQ as a measure of intelligence and test the candidate drug by splitting
drug trial participants into a treatment and a control group. We fit a distribution
to each group and compare the distributions.


Model Specification
---------------------

In Kruschke's original paper a model is described for the case of two groups
and one quantity of intrest. In real world data, we are dealing with multiple
variables of intrest and possibly, multiple groups. While Kruschke provides
some guidelines for these extentions, these are left somewhat nebulous and open
to interpretation. The following extentions are specific to this library.


Let :math:`$\overset{N \times M+1}{X}$` by a matrix representing our measurements. This
matrix is composed of $N$ observations of $M$ continuous variables, and we further
assume an additional, categorical variable whose values represents the groups
themselves. Suppose this group variables takes $K$ possible values. We split the
data according to the values of this variable :math:`$\{\mathbf{X_0, \dots, X_K}\}$`
and fit distributions to each group. The distribution choses here is the
**Student T** distribution. This distribution is chosen because, while similar
to the Normal, it has thicker tails and hence better at describing data with
"extreme" values, which are quite common in practice.

The Student T distribution (in one dimention) is given by:

.. math:: 

    f(x|\mu, \lambda, \nu) = \frac{\Gamma (\frac{\nu+1}2 )}{\Gamma (
        \frac \nu 2)} \big( \frac{\lambda}{\pi\nu} \big)^{\frac 12}
        \big[1+\frac{\lambda (x-\mu)^2}{\nu} \big]^{-\frac {\nu+1}2}

This distribution is defined by three shape parameters, two which identicall
to the Normal (mean and standard deviation) and a unique parameter :math:`\nu`.
This parameter is called the **degrees of freedom** and determines essentially
the "normality" of the distribution. For values of :math:`\nu` close to 1, the
distribution has thick tails, which shrink toward the equivalent normal as it
increases. Therefore, we can express this grouping as:

.. math:: 

    \begin{array}{c}
    \mathbf{X_0} \thicksim \mathcal{T}(\nu_0, \mu_0, \sigma_0)\\
    \mathbf{X_1} \thicksim \mathcal{T}(\nu_1, \mu_1, \sigma_1)\\
    \vdots\\
    \mathbf{X_K} \thicksim \mathcal{T}(\nu_K, \mu_K, \sigma_K)\\
    \end{array}

We must specify **priors** for these free parameters, which will be estimated
from the observed data itself. Kruschke proposes the following prior distributions:

.. math:: 

    \begin{array}{c}
    \nu_0=\nu_1=\dots =\nu_K=\nu\\
    \\
    \nu \thicksim 1 + \mathcal{Exp}(\lambda=\frac 1 {29.0})\\
    \\
    \mu_k \thicksim \mathcal{N}(\bar{\mu}_k,\ 2\bar{\sigma})\\
    \\
    \sigma_k \thicksim \mathcal{U}(10^{-3}, 10^3)
    \end{array}

We can make the simplifying assumption that the degrees of freedom are the same
across all groups, which is usually a good idea (but not necessary) as we've
found the alternative usually results in **overfitting**. For the means, we
declare one per group (as per Kruschke's paper) and set them to a Normal centered
around the empirical pooled mean and standard deviation of the group itself.
This quantities are derived from the observations themselves, and constitute
wide, uninformative priors. More importantly, this does not favor a group over
the other a priori. For standard deviations we select a uniform distribution
in the interval :math:`[0.0001, 1000]`. As Kruschke himself notes, these priors
are exceedingly wide, and described this way only to abstractly encompass all
possible data (astronomical, medical, etc). For most real-world applications,
they are excessively large and should be adjusted according the data and their
units.

From the above, we can calculate any derived, deterministic quantity, primarily
the **difference of means **(:math:`\Delta \mu`), the **difference of standard
deviations** (:math:`\Delta \sigma`) and an additional quantity which Kruschke
terms the **effect size** (:math:`E`). 

.. math::

    E_{ij} \triangleq \dfrac{\Delta \mu_{ij}}{
    \frac{\sqrt{
        \sigma_i^2+\sigma_j^2
        }} {2}
    }

This quantity is no longer expressed in the same units as the inputs and hence it is somewhat harder to interpret than
the others. :math:`\Delta \mu` expresses the expected difference between the
groups and hence is the quantity we are primarily intrested in. :math:`\Delta \sigma`
expresses the difference is variaces between groups and hence can be interpreted
as expressing whether specific instances of the observed, are affected differently
for certain values of the grouping variable. The effect size expresses a standardized
measure of the "magnitude" of the difference.


Making Decitions:
------------------
To move from the hazy space of propabilities to that of actionable decitions,
we need to propose, apply and justify a **decition rule**. Classical hypothesis
testing typically justifies its decition making process by appealing to reproducibility.
Applying the t-Test is valid, because it ensures Type I and Type II errors are
minimized in the long run, for infinite experiments and infinite researchers.
There are multiple rules by which one can arrive to binary (or really tertiary)
decitions - the one considered here is the earliest proposed for this model
and is called the **HDI+ROPE** rule. The true innovation of this approach is
the definition of the **ROPE** or the **Region Of Practical Significance**.
This is a region of values that practically equivalent to zero. Hence the ROPE
interval is defined on a problem specific basis - *how big enough of difference
is big enough?*. The **credible interval** (CI) is Bayesian equivalent of the
confidence interval and expresses a similar idea. It can be though of as the
interval of values considered plausible (under some probability threshold).
From these two we can arrive at our decitions according to the following rule:

If the CI and the ROPE have nothing in common, then none of the plausible values
are equivalent to zero. We therefore decide the groups are **different**. If the
CI is entirely inside the ROPE, then all plausible values are equivalent to zero.
Therefore we decide the groups are **the same**. If the two intervals partially
overlap, then *some* plausible values are equivalent to zero, while some are not.
We therefore **withold out decition** or simple declare it **Indeterminate**.

This decition rule can be justified via a branch of mathematics called 
**Bayesian Decition Theory** which studies decitions under a state of uncertainty.
A full description is outside the scope of this document (and this library), however
we can provide a rough outline.

We begin be observing the decitions are generally not made in a vacuum but are
attended by **actions**. We therefore denote :math:`\alpha\in \Alpha` the set
all possible actions, :math:`\Theta` the space of all possible values of model
parameters and :math:`\mathcal{X}` the space of all possible inputs. We define
a **loss function**:

.. math:: 

    L:\mathcal{\Theta}\times\mathcal{\Alpha}\rightarrow \mathbb{R}_0^+

This function maps values of the model and decitions taked to a "cost" of making
this decition. We then define a **decition function**:

.. math:: 

    \delta :\Theta \rightarrow \Alpha

From this, justification can be derived in a general way. In practice formulating
a problem specific loss function is usually difficult in practice




Extentions: Multiple Inputs
----------------------------

To extend the model for multiple inputs, there are several options. We can
replace the univariate with a **multivariate Student T**, or use multiple, univariate
distributions. The multivariate Student T has a pdf of:

.. math:: 

        f(\mathbf{x}| \nu,\mu,\Sigma) =
        \frac
            {\Gamma\left[(\nu+p)/2\right]}
            {\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}
             \left|{\Sigma}\right|^{1/2}
             \left[
               1+\frac{1}{\nu}
               ({\mathbf x}-{\mu})^T
               {\Sigma}^{-1}({\mathbf x}-{\mu})
             \right]^{-(\nu+p)/2}}

Note the Multivariate distribution, replaces the parameter :math:`\sigma` with
a **covariance matrix**. The element of the diagonal of his matrix are variances
accross each input variable, and the off-diagonal elements are **covariances**
between different variables. We assume this matrix is diagonal i.e. that there
are no covariances between the input variables (since we are generally not 
intrested in describing the covariances themselves). Alternatively we can
fit independant univariate distributions for each input variable. We can assume
the same degrees of freedom for each of allow them independant. The former case
is theoretically identical to the multivariate with a diagonal covariance matrix,
but more computationally expensive. The latter case is possible but prone to
overfitting.

Extentions: Multiple Groups:
------------------------------

Kruschke provides some guidelines on how to extend this model for multiple groups
but provides no tangible examples. We intrepret his recommendation to place
independant :math:`\mu_k` and :math:`\nu_k` as pair-wise comparisons between
the groups. Hence we implement comparisons for all combinations of values of
the grouping variable. Therefore, for :math:`i,j\in\{0, \dots K\}` we generate
all combinations (unique pairs withough repetition or order) and the derived
metrics would be:

.. math:: 

    \begin{array}{c}
    \Delta\mu_{00}\\
    \Delta\sigma_{00}\\
    E_{00}\\
    \Delta\mu_{01}\\
    \Delta\sigma_{01}\\
    E_{01}\\
    \vdots\\
    \Delta\mu_{ij}\\
    \Delta \sigma_{ij}\\
    E_{ij}\\
    \vdots\\
    \Delta\mu_{KK}\\
    \Delta \sigma_{KK}\\
    E_{KK}\\
    \end{array}


Options and Alternative Formulations
----------------------------------------

There are multiple ways to extend the base model, that Kruchke mentions, not
all are accounted for in this implementation. The distribution could be replaced
with a LogNormal for special cases, for example. There are alternative ways to
generalize the model as well. One could for example relax the diagonality assumption
and model the resulting matrices with the specialized LKJCholesky distribution
(included in `pymc`). Another possibility is including a hierarchical distribution
for the means, however this formulation is rather nebuluous and would likely
overcomplicate things. For this reason we chose not to include it. It could potentially
be considered in the future however. In terms of the options that are present in
this implementation, the default option is univariate distribution with common
degrees of freedom. Allowing independant degrees of freedom for input variables,
we've found (via loo) usually results in overfitting. This case is identical to
the multivariate, but more efficient.


Implementation:
-----------------

The terms of the actual software implementation, the above model is represented
by the `bayesian_models.BEST` class. Like all models it implements the general
four methods `__init__`, `__call__`, `fit` and `predict`. The `__init__` method
is responsible of setting all parameters and hyperparameters of the model itself.
Unusually, these are class methods rather than object methods. This choice was
made to enable rapid prototyping. The expected use-case is that a single 'version'
of the model can be applied to multiple data, in the sense of multiple grouping
variables. We often deal with dataset with multiple categorical variables and
we are intrested in knowing whether these variables impact other continuous variables.
This model accepts no "testing" data, and hence its data containers are declared
immutable. The `predict` method itself renders the decisions and returns findings
as a dictionary, mapping each quantity :math:`\Delta \mu_{ij}`, :math:`\Delta \sigma_{ij}`, etc
to a `pandas.DataFrame` containing the results. These are typically of the same
shape as those of `arviz.summary` with an additional column representing the
decision, according to the **HDI+ROPE** rule. Other rules are possible, but they
are not implemented at present, as we think this rule is very intuitive, easy to
use and works well for many common cases. The decision making process itself
is defered to the `predict` method, since it is meaningless prior to training
completion. This way, a user could also rapidly test many options of ROPEs and
HDI thresholds on the same model and aggragate the results, without retraining
each time. After all the decition making process itself is independant of the
inference process.