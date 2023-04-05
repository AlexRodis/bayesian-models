Documentation
***************


General
---------


`bayesian_models` has as one of its key goals *accessibility*, to provide
access to bayesian inference to non specialists. Hence documentation is critical
and should not assume any but the most basic knowledge. For the same reasons,
highly specialized jargon is discouraged and should be avoided. The documentation
should be kept up to data at all times, and constitutes a critical priority of
this project. It's general structure is inspired by that of Django and 
`this talk by daniele procida <https://www.youtube.com/watch?v=t4vKPhjcMZg>`_.
It is split into four general categories.

.. important:: 

    To ensure a high quality documentation is maintained, the docs should
    be updated with every major and minor release. Any major changes, especially
    new features and API changes should not be merged to main until documentation
    is reviewed, and updated where needed

Tutorials
-----------

Perhaps one of the most important types of documentation this is likely the
first thing a prospective user will see. Its key purpose is to build confidence
in the capabilities of the project itself. It should have little to no explanations
and should by reproducible, always. It takes the form of a (short) recipe, a list
of instructions. It should not contain any options or altenatives. There should
be a tutorial for the library itself, as well as one for each model


How-To Guides
---------------

How-To guides are another major category of documentation, similar but distinct
from tutorials. These guides are problem specific. They set a specific goal and
provide a list of instructions to solve the problem. They should contain a short 
description of the problem but should not explore alternatives and should
avoid complex, multi faceted tasks. Their titles should be short and descriptive.
They are addressed to users that typically already have some familiarity with
the library itself, and may try to answer questions that a true beginer may
not even think to ask.


Reference
-----------

The most common form of documentation. Its a technical guide explaining the
components of the project from a technical, programmatic perspective. It includes
function and class definitions and the like. It is information oriented, and should
list and catalogue things.


Discussions
------------

You are reading this kind of documentation right now. The purpose of discussion
documents is to explain the philosophy behind various library topics, i.e. explain
**why** things are the way they are. They are more descriptive, rather than
practical. `bayesian_models` is a rather specialized library, namelly it involves
around statistical models. Pages explaining should be place in this section.
The How-To guides themselves should only contain only a basic introduction, whatever
is absolutely necessery for a new user to know.