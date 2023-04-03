
#   Copyright 2023 Alexander Rodis
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# This module contains basic functionality for model building and basic
# machinery the library requires

from dataclasses import dataclass, field
from typing import Any, Type, Callable, Optional, Union, Iterable
from abc import ABC, abstractmethod
import pymc
from collections import defaultdict, namedtuple
from bayesian_models.data import Data
from bayesian_models.utilities import extract_dist_shape, invert_dict


Response = namedtuple("Response", ["name", "func", "target", "record"])
Response.__doc__ = r"""
Container object for response functions

Response functions are model components that transform model variables
into other model variables. This is a :code:`collections.namedtuple`
container and has no methods

Object Attributes:
------------------

    - | name:str := An internal name for the transformation and its
        result
    
    - | func:Callable := The :code:`Callable` object that handles the
        actual transformation
    
    - | target:str := Internal name for the variable that is to be
        transformed. The result of a lookup for this variable becomes
        the input to the :code:`Callable` in the :code:`func` field.
        This should be the only argument to the Callable
    
    - | record:bool := If :code:`True` the result of the transformation
        will be wrapped in a deterministic variable and preserved in the
        posterior trace. Else they will be accessible only internally,
        via the :code:`variables` property

"""
ModelVars = defaultdict(default_factory = lambda : None)
Real = Union[float, int]

@dataclass(slots=True, kw_only = True)
class Distribution:
    r'''
        Data container class for user-specified distributions. 
        
        Supply a distribution :code:`dist` along with any desired
        arguments :code:`dist_args` and :code:`dist_kwargs`. If the
        distribution needs to be modified, for example by shifting it by
        some amount, supply a :code:`dist_transform:Callable`.
        Optionally  a :code:`name` can be provided. As a data container
        class, it has no methods.
        
        NOTE: For data nodes, the :code:`dist` argument should by the
        function :code:`pymc.Data`
        
            Example usage:

        .. code-block:: python

            dist = Distribution(
                dist=pymc.Normal, 
                name='some_name', 
                dist_args=(0,), 
                dist_kwargs=dict(sigma=1) 
            )

            dist = Distribution(
                dist=pymc.Exponential, 
                name='exp', 
                dist_args=(1/29.0,), 
                dist_transform=lambda d:d+1 
            )

            dist = Distribution(
                name='inputs', 
                dist=pymc.Data,
                dist_args=(np.random.rand(50,5), ),
                dist_kwargs=dict(mutable=True)
            )
            
        Object Attributes:
        ------------------
        
            - | name:str='' := A name for the distribution / random
                variable. Technically optional but should be provided
            
            - | dist := For random variables the distribution to be used
                as a prior. Initially  a reference to the class, will be
                latter swapped to an instance of the class. For data
                nodes, the function :code:`pymc.Data` can be passed
                instead
            
            - | dist_args:tuple := Positional arguments to the
                distribution
        
            - | dist_kwargs:dict := Keyword arguments to the
                distribution

            - | dist_transform:Optional[Callable]=None := A Callable
                that is used to perform a numerical transform to the
                variable. Will not be explicitly tracked by the model
                (its result will). Used to offset, and manipulate core distributions. Optional. Defaults to :code:`None`
    
    '''
    
    
    name:str = ''
    dist:Union[Type[
        pymc.Distribution], Type[Callable]
               ] = pymc.Distribution
    dist_args:tuple = tuple()
    dist_kwargs:dict = field(default_factory=dict)
    dist_transform:Optional[Callable] = None

    def __post_init__(self)->None:
        r'''
            Check for invalid inputs
         '''
        if any([
            any([
                not isinstance(self.name,str),
                self.name == None,
                self.name == '',
                ]
            ),
            not isinstance(self.dist_args, tuple),
            not isinstance(self.dist_kwargs, dict),
        ]):
            raise ValueError(("Illegal values received"))
        
def distribution(dist:pymc.Distribution,name:str,
                 *args, transform:Optional[Callable]=None, 
                 **kwargs)->Distribution:
    '''
        Convenience method for fresh :code:`Distribution` instance
        creation.
        
        
        Accepts a a distribution and a name, along with optional args
        and kwargs and returns a :code:`Distribution` object matching
        those parameters.
        
        Example usage:
        
        .. code-block:: python
        
            distribution(pymc.Normal, 'W', 0,1)
            # Equivalent to Distribution(dist = pymc.Normal, 
            # dist_name = 'W', dist_args = (0,1), dist_kwargs=dict()
            # )
            distribution(pymc.Beta, 'b', alpha=1,beta=1)
            # Equivalent to Distribution(dist=pymc.Beta, dist_name='b',
            # dist_args=tuple(), dist_kwargs=dict(alpha=1, beta=1))
            distribution(pymc.StudentT, 'T', 0, sigma=1, nu=2)
            # Equivalent to Distribution(dist=pymc.StudentT, 
            # dist_name='b', dist_args=(0,), dist_kwargs=dict(sigma=1,
            # nu=2))
    '''
    return Distribution(dist = dist, name = name,
                        dist_args = args, dist_kwargs = kwargs,
                        dist_transform = transform)

@dataclass(slots=True)
class FreeVariablesComponent:
    r'''
        Component representing additional variables to be inserted to
        the model. 
        
        Used for variables not explicitly involved in the core model
        itself, i.e. the equations describing the model. For example
        the noise parameter in linear regression:
        
        .. math::

            \begin{array}{c}
                w0 \thicksim \mathcal{N}(0,1)\\
                w1 \thicksim \mathcal{N}(0,1)\\
                μ = X*w0+w1\\
                σ \thicksim \mathcal{N}(0,1)\\
                y \thicksim \mathcal{N}(μ, σ)\\
            \end{array}
        
        The :code:`dists` argument supplied is a :code:`dict` of names
        to :code:`Distribution` instances, representing the
        distributions to be inserted to the model.
        
        Example usage:
        
            .. code-block:: python
            
                fvars = FreeVariablesComponent(
                        dict( sigma = Distribution(name='sigma',
                            dist=pymc.Normal, dist_args=(0,1)
                            )
                        ) # Insert random variable 'sigma' to the model
                    
        Object Attributes:
        ------------------
        
            - | variables:dict := A catalogue of variables inserted into
                the model. Is a dictionary mapping variable names to
                references to the appropriate object
              
            - | dists:dict[str, Distribution] := Distributions to be
                added to the model
                
        Object Methods:
        ---------------
        
            - | __call__()->None := Expects to be executed with a
                :code:`pymc.Model` context stack open. Adds all
                distributions to the model and updates the
                :code:`variables` attribute
    '''
    
    variables:Any = field(init=False, default_factory=dict)
    dists:dict[str, Distribution] = field(default_factory=dict)
    
    def __post_init__(self):
        '''
            Verify inputs
        '''
        if self.dists  == dict():
            raise ValueError((
                "Attempting to create an free variables but no "
                "distributions have been passed"
            ))
        if not isinstance(self.dists, dict):
            raise TypeError((
                "Expected a dict of free variables names to Distribution "
                f"object. Saw {self.dists} instead"
            ))
        if not all([
            isinstance(k,str) for k in self.dists.keys()
        ]):
            illegals = {
                k:type(k) for k in self.dists.keys()
                }
            raise TypeError((
                "Variable names must be strings. Received "
                f"{illegals} instead"
            ))
        if not all([
            isinstance(v, Distribution) for _,v in self.dists.items()
        ]):
            illegals = {
                v:type(v) for _, v in self.dists.items()
            }
            raise TypeError((
                "Items of of input dicts must be instances of :code:`Distribution`. "
                f"Received {illegals} instead"
            ))
    
    def __call__(self)->None:
        for name, dist in self.dists.items():
            d = dist.dist(dist.name, *dist.dist_args,
                          **dist.dist_kwargs
                          )
            self.variables[name] = d 
    
@dataclass(slots=True)
class ResponseFunctions:
    '''
        Data container for Response functions. 
        
        Accepts three mappings as dicts. All three map strings
        representing variable names for the result of the response to a
        parameter. The :code:`functions` argument maps names to the
        actual functions themselves. The :code:`application_targets`
        parameter maps transformed variable names to the variables that
        are inputs to the transformation. The :code:`records` parameter
        maps variable names to a boolean representing whether they
        should be recorded or not. :code:`True` will wrap the result of
        the transform into a deterministic node, False will not. The
        :code:`application_targets` and the :code:`records` parameters
        can be partially or completely omitted. In this case, record
        defaults to True and the application_target default to 'f' a
        general name for the raw model output. If any variable is found
        in either :code:`application_targets` or :code:`records` but not
        in :code:`functions` an exception is raised, since not
        reasonable inference can be made for the transform function
        instead. 
        
        Example usage:
        
        .. code-block:: python
        
            # Pass all parameters explicitly (recommended)
            
            ResponseFunctions(
                functions = dict(exp = pymc.math.exp, tanh =
                pymc.math.tanh), records = dict(exp=True, tanh=False),
                application_targets = dict(exp="f", tanh="exp")
            ) # Partially omit application_targets using defaults 'f'
            
            ResponseFunctions(
                functions = dict(exp = pymc.math.exp, tanh =
                pymc.math.tanh), records = dict(exp=True, tanh=False),
                application_targets = dict(tanh="exp") 
                ) # Pass the
                # desired Callables leaving everything to their
                # defaults. In this case two different response functions
                # are applied to the same input 'f'. Both are recorded
                # with :code:`pymc.Deterministic` the name is the key 
                # provided
            
            ResponseFunctions( functions = dict(exp = pymc.math.exp,
            tanh = pymc.math.tanh) )
            
        This object is an Iterable over response functions
            
        Object Attributes:
        ------------------

            - | functions:dict[str, Callable] := A dictionary mapping
                internal variable names to Callable objects defining the
                transformation
                
            - | records:dict[str, bool] := A dictionary mapping internal
                variable names to booleans. If the boolean is
                :code:`True` the variable is wrapped in a deterministic
                node and preserved in the posterior trance. Else it
                accessible only internally. Optional. Anything not
                explicitly defined here is inferred to be true
            
            - | application_targets:dict[str, str] := A dictionary
                mapping response names to the internal name of the
                variable to be transformed. Optional. Any keys missing
                are automatically mapped to 'f', the default name for
                core model output tensor
                
        Private Attributes:
        ===================
        
            - | _sf:set[str] := Set of function names specified
            
            - | _st:set[str] := Set of function targets specified
                (via application_targets keys)
            
            - | _missing_records:set[str] := Set of function records
                (keys) not explicitly specified by the user
            
            - | _missing_targets:set[str] := Set of function targets
                (keys of :code:`application_targets`) not explicitly set
                by the user
            
            - | _iter:Optional[Iterable] := Iterator object over the
                specified :code:`functions` used internally to make the
                object an iterable
    
    '''
    
    _sf:set[str] = field(init=False, default_factory=set)
    _st:set[str] = field(init=False, default_factory=set)
    _sr:set[str] = field(init=False, default_factory=set)
    _missing_records:set[str] = field(init=False, default_factory=set)
    _missing_targets:set[str] = field(init=False, default_factory=set)
    functions:dict[str, Callable] = field(default_factory=dict)
    application_targets:dict[str, str] = field(default_factory=dict)
    records:dict[str, bool] = field(default_factory=dict)
    _iter:Optional[Iterable[str]] = field(init=False)
    
    
    def _validate_inputs(self)->None:
        '''
            Validate inputs by raising on incompatible specs
        '''
        if self.functions == dict():
            raise ValueError((
                "Attempting to add response functions but no functional "
                "mapping was provided. :code:`functions` must be a dict mapping "
                f"new variable names to Callables. Received {self.functions} "
                "instead"
                ))
        if self._sr-self._sf!=set():
            raise ValueError(
                ('New response variable specified in records not '
                 f'specified in function. Variables {self._sr-self._sf} not found'
                 'in provided functions')
            )
        elif self._st-self._sf:
            raise ValueError(
                ('New response variable specified in application_targets'
                 'not '
                 f'specified in function. Variables {self._st-self._sf} not found'
                 'in provided functions')
            )

    def __post_init__(self)->None:
        '''
            Collect specifications, validate inputs and fill missing
            values with defaults. Also initialized a self iterator for 
            the object
        '''
        self._sf = set(self.functions.keys())
        self._st = set(self.application_targets.keys())
        self._sr = set(self.records.keys())
        self._validate_inputs()
        self._missing_records:set[str] = self._sf-self._sr
        self._missing_targets:set[str] = self._sf-self._st
        self.records = self.records|{
            k:True for k in self._missing_records}
        self.application_targets = self.application_targets|{
            k:"f" for k in self._missing_targets
        }
        self._iter = iter(self.functions.keys())
    
    def get_function(self, func:str)->Response:
        '''
            Returns all data kept on a single response function as
            a namedtuple for ease of access. Looks up all specs for
            a response and returns a namedtuple with the results 
            packaged. Fields provided are :code:`name`, :code:`func`, :code:`target` and
            :code:`record`
        '''
        try:
            fetched = Response(
                name = func,
                func = self.functions[func],
                target = self.application_targets[func],
                record = self.records[func]
            )
            return fetched
        except KeyError:
            raise RuntimeError((
                f"Requested response function {func} not found"
                ))
            
    def __next__(self):
        '''
            Return a namedtuple for the next response function
        '''
        name = next(self._iter) #type: ignore
        return self.get_function(name)

    def __iter__(self):
        return self

@dataclass(slots=True)
class ResponseFunctionComponent:
    '''
        Model component representing Response functions. 
        
        Accepts response functions specified via the
        :code:`ResponseFunctions` class. Adds them to the model and
        maintains an internal catalogue for variables added as
        :code:`variables`. 
        
        Example usage:
        
        .. code-block:: python
        
            res_comp = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = dict(
                        exp = pymc.math.exp, 
                        tanh = pymc.math.tanh
                    ),
                    records = dict(exp=True, tanh=False),
                    application_targets = dict(exp="f", 
                    tanh="exp")
                )
            )
        
        Object Attributes:
        ------------------
        
            - | responses:ResponseFunctions := A
                :code:`ResponseFunctions` object encapsulating all the
                response functions to be added to the model
            
            - | variables:dict[str, Any] := A catalogue of variables
                added to the model, stored as mappings of strings
                (names) to references to the underlying object
                
        Object Methods:
        ---------------
        
            - | __call__(modelvars:dict[str, Any])->None := Add
                specified response functions to the model. Assumes a
                :code:`pymc.Model` context is open. Updates the :code:`variables`
                attribute. :code:`modelvars` is the general catalogue of
                variables present in the model as name:str to object
                mappings (supplied by the builder)
    '''
    responses:ResponseFunctions
    variables:dict=field(init=False, default_factory=dict)
    
    def __call__(self, modelvars:dict[str,Any]):
        '''
            Add specified response functions to the model stack. 
            
            :code:`modelvars` is a general index of all the variables
            present in the model as a 'var_name':var_ref mapping.
            Supplied by the builder
            
            Args:
            -----
            
                - | modelvars:dict[str, Any] := The catalogue of all
                    variables present in the model. Contained as
                    var_name:str = var_obj pairs. Should be supplied by
                    the builder (who holds the master variable
                    catalogue)
                    
            Returns:
            --------
            
                - None
                
            Raises:
            -------
            
                - | ValueError := If :code:`modelvars` is not a valid
                    dictionary
                    
                - | RuntimeError := If a specified target variable is
                    not found in the model
        '''
        if modelvars is None or not isinstance( modelvars, dict):
            raise ValueError((
                f"Illegal model variables received {modelvars}"
                ))
        for response in self.responses:
            try:
                expr = response.func(modelvars[response.target])
                f_trans = expr
                if response.record:
                    f_trans = pymc.Deterministic(
                        response.name, f_trans
                    )
                self.variables[response.name] = f_trans
            except KeyError:
                raise RuntimeError((
                    f"Response function {response.name} attempted to "
                    f"operate on non-existant model variable "
                    f"{response.target}. Variable {response.target} not "
                    "found on the model"
                ))
            
    
@dataclass(kw_only=True, slots=True)
class ModelAdaptorComponent:
    '''
        Adds an 'output adaptor' component to the model, which 'splits'
        the models output tensor into other variables. 
        
        The splitting logic is specified by the :code:`var_mapping`
        argument, which is a dict whose keys are strings representing
        variable names, and whose items are Callables that accept and
        return :code:`theano` tensors. The :code:`record` boolean
        argument decides if the result of all "splits" will be wrapped
        in a deterministic node.
        
        Example usage:
        
        .. code-block:: python
        
            adaptor = ModelAdaptorComponent(
                record = True, # Keep the new variables as 
                # Deterministic nodes for later access
                var_mapping = dict(
                    mu = lambda tensor: tensor.T[0,...],
                    sigma = lambda tensor: tensor.T[1,...],
                ) # Split model output 'f', a tensor, into two 
                # tensors named 'mu' and 'sigma', wrapped as 
                # deterministics in the model.
        
        Object Attributes:
        ------------------
        
            - | var_mapping:dict[str, Callable] := Mapping of variable
                names to Callable objects which return the variable. All
                Callables here receive exactly one argument, the tensor
                that is the output of the model and the Callable should
                execute the splitting, returning exactly one subtensor
        
            - | record:bool=True := If :code:`True` (default) records
                the subtensors as deterministic variables, adding them
                to the models posterior trace. Otherwise, they are
                treated as nuissance intermediates, accessible via the
                models' :code:`var_names` catalogue
        
            - | variables:dict := Catalogue of new objects added to the
                model by this component. Keys are names, and items are
                references to the subtensor objects themselves. Allows
                'anonymous' non-deterministic subtensors to be accessed
                by other components without being added to the models'
                trace
                
        Object Methods:
        ---------------
        
            - | __call__(output)->None := Creates and adds specified
                subtensors to the model. Assumes a :code:`pymc.Model`
                context stack is open. :code:`output` is a reference to
                the models output. Also updates the :code:`variables`
                attributed with all the subtensors created.
                :code:`output` will be passed to all the Callables
                specified in :code:`var_mapping`
    
    '''
    
    variables:dict = field(init=False, default_factory=dict)
    record:bool = True
    var_mapping:dict[str, Callable] = field(default_factory=dict)
    
    def __call__(self, output):
        '''
            Called by the builder to add the specified variables to the
            model. :code:`output` is a ref to the tensor object to be
            split and should be what the :code:`var_mapping` items
            accept
        '''
        for new_var, func in self.var_mapping.items():
            if self.record:
                v = pymc.Deterministic(new_var, func(output))
            else:
                v = func(output)
            self.variables[new_var] = v



@dataclass(kw_only=True, slots=True)
class LikelihoodComponent:
    '''
        Adds a likelihood component to the model. 
        
        :code:`var_mapping` must be supplied, defining which model
        variables map to which likelihood shape parameters. For example
        :code:`distribution=pymc.Normal` and `var_mapping=dict(mu = 'f',
        sigma='sigma')`. The keys of the :code:`var_mapping` dict are
        strings which must exactly match the shape parameters of the
        likelihood. Values should be the internal names of model
        variables, specified during their creation. :code:`name` and
        :code:`observed` correspond to the name of the likelihood object
        itself and the name of the observed data node in the model.
        Should generally be left to their defaults, except when the
        model has multiple likelihood/observed nodes.
        
        Example usage:
        
        .. code-block:: python
        
            like = LikelihoodComponent(
                name = 'y_obs', # Default likelihood name observed =
                'observed', # Default name for # the name of the input
                data node distribution = pymc.Dirichlet, # Defaults # to
                pymc.Normal var_mapping = dict(
                    a = 'f', ) # Mapping of distribution shape
                    parameters # to model parameter names. 'f' is
                    usually # the default name for the core model #
                    output
                )
        
        Object Attributes:
        ------------------
        
            - name:str='y_obs' := The name for the likelihood variable
            
            - | observed:str='outputs' := The internal name of the data
                node containing the observations. Should match the
                :code:`name` attribute of the :code:`Distribution`
                object that defined the data node
                
            - | distribution:Type[pymc.Distribution]=pymc.Normal := The
                distribution for the likelihood/observations
                
            - var_mapping:dict[str, str] := Mapping of the likelihoods'
              shape parameters to internal names of model variables to
              be supplied to the respective parameter. The keys are
              shape arguments to the supplied distribution. The items
              are strings representing variable names that will be
              looked up in the variable catalogue
              
        Object Methods:
        ----------------
        
            - | __call__(observed, var_mappingLdict)->None := Expects to
                be called with :code:`pymc.Model` context open and wil
                add the likelihood to the model. :code:`observed` is the
                result of a lookup to the models' :code:`variables`
                catalogue for the :code:`observed` attribute and is a
                reference the observations. :code:`var_mapping` has the
                same keys as the original and its items are the result
                of a lookup on the :code:`var_mapping` item lookup into
                the models :code:`variables` catalogue
    '''
    name:str = 'y_obs'
    observed:str = 'outputs'
    distribution:Type[pymc.Distribution] = pymc.Normal
    var_mapping:dict[str, str] = field(default_factory = dict)
    
    def __post_init__(self)->None:
        if self.var_mapping == dict():
            self.var_mapping = dict(
                mu = 'f', sigma = 'ε'
            )
    
    def __call__(self, observed,
                 **var_mapping:dict[str, pymc.Distribution])->None:
        if observed is None:
            raise RuntimeError((
                "Attempting to specify a likelihood without "
                f"observations. Observed variable {self.observed} "
                "not found in the model. Did you forget to specify "
                "inputs or update their name?"
                ))
        if any([
            var_mapping == dict(), 
            not isinstance(var_mapping, dict),
            ]):
            raise ValueError(
                ("Invalid variable mapping. Expected "
                 "a dict of strings to Distribution, received "
                 f"{var_mapping} instead"))
        y_obs = self.distribution(
            self.name, observed = observed,
            **var_mapping
        )#type: ignore

class ModelBuilder(ABC):
    '''
        Abstract base class for model builders 
    '''
    
    @property
    @abstractmethod
    def model_variables(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()

@dataclass(kw_only=True, slots=True)
class CoreModelComponent:
    '''
        Core model component. 
        
        Inserts basic random variables to the context stack, maintaining
        an internal catalogue of variables inserted for latter access.
        Concrete model objects should subclass this component and extend
        its call method by defining the models' equations in
        deterministic nodes. If a concrete model produces some explicit
        quantity as an output, this should generally  be named 'f'.
        
        Example usage:
        
        .. code-block:: python
        
            core = CoreModelComponent(
                        distributions = dict(
                            beta0 = distribution(pymc.Normal,
                            'beta0', 0,1),
                            beta1 = distribution(pymc.Normal,
                            'beta1', mu = 0, sigma=1),
                        ) # OLS-like example, using convenience
                        # method :code:`distribution` to access 
                        # the :code:`Distribution` class
                )
                
        Object Attributes:
        ------------------
        
            - | distributions:dict[str, Distribution] := A dictionary
                mapping random variable names to their priors/
                distributions. The :code:`Distribution` class
                encapsulates all the information for a prior
                distribution. Can be accessed via the convenience method
                :code:`distribution`
            
            - | variables[str, Any] := A catalogue of basic random
                variables added to the model. Child classes extend this
                by adding their deterministics
              
            - | model:Optional[pymc.Model] := A reference to
                :code:`pymc.Model` object itself. Is created or defined
                by the :code:`ModelBuildDirector` and handed to the
                builder to construct the model
                
        Object Methods:
        ----------------
        
            - __call__()->None := Assumes a :code:`pymc.Model` context
              stack is open. Adds all specified random variables to the
              model by calling them from the :code:`distributions` dict
              and adds them to the :code:`variables` catalogue
    '''
    
    distributions:dict[str, Distribution]= field(default_factory = dict)
    variables:dict = field(init=False, default_factory=dict)
    model:Optional[pymc.Model] = None
    
    def __post_init__(self)->None:
        '''
            Check for illegal inputs
        '''
        
        if self.distributions == dict():
            raise ValueError(("Attempting to initialize core component "
                              "with no variables"))
        if not isinstance(self.distributions, dict):
            raise ValueError(
                ("distributions argument must be a dictionary whose "
                 "keys are variable names and values are "
                 ":code:`bayesian_models.Distribution` instances. Received "
                 f"{self.distributions} of type "
                 f"{type(self.distributions)} instead")
                )
        vs = self.distributions.items()
        try:
            assert all(isinstance(e, Distribution) for _,e in vs)
        except AssertionError:
            illegal:dict[str, Any] = {
                k:type(v) for k,v in vs if not isinstance(v, Distribution)
                }
            raise ValueError((
                "Core component distributions must be supplied as "
                ":code:`bayesian_model.Distribution` instances. Received "
                f"illegal values {illegal}"))

    def __call__(self)->None:
        for key_name, dist in self.distributions.items():
            d = dist.dist(dist.name, *dist.dist_args,
                **dist.dist_kwargs
                )
            self.variables[key_name] = d
       
            
class LinearRegressionCoreComponent(CoreModelComponent):
    '''
        Core model component for linear regression
        
        Specifies the model as
        
        .. math::
            
            f = XW+b
        
        Inserts the deterministic variable 'f' to the model. Delegates
        definition of the random variables to the
        :code:`CoreModelComponent` parent.
        
        NOTE: When all the priors and the likelihood are Normal this is
        analytically tractable and MCMC is not needed. Included here as
        a reference example mostly, though one could run a linear
        regression with an unusual likelihood / response functions etc
        
        Class Attributes:
        -----------------
        
            - | var_names:dict[str, str] := Alternate names for the
                various variables of the model. Allows users to refer to
                the variables by a different name e.g. "b" instead of
                'W'
                
            - model_vars:set := The models' variable names
        
        Object Attributes:
        ------------------
        
            - | variables:dict[str, str] : Catalogue mapping variables
                names to references to the variables. Created by the
                parent and extended to include the deterministic 'f'
                
            - | var_names:dict[str, str] := Mapping of internal variable
                names to user-defined names. Optional. Defaults to an
                empty dict. Any user supplied variables not
                corresponding to known model variables are ignored. Any
                variable aliases not supplied by the user are replaced
                with their defaults. Warns on any mismatch
                
        Private Attributes:
        ====================
        
            Variables used internally, mainly for input validation
            
            - | s1:set[str] := User-defined/supplied variable names
            
            - | s2:set[str] := Model variables (class level attribute)
            
        Object Methods:
        ----------------
        
            - | __call__()->None := Add the models' equation to the
                model as a deterministic node. Assumes a
                :code:`pymc.Model` context stack is open. Model output
                is named :code:`var_names['equation']` 'f' by default
    '''
    
    var_names = dict(
        slope = "W", intercept = "b", 
        data = "inputs", equation = "f"
    )
    model_vars = {"slope", "intercept", "data", "equation"}
    
    
    def __init__(self, distributions:dict[str, Distribution]=dict(),
                 var_names:dict[str, str] = {},
                 model:Optional[pymc.Model] = None)->None:
        from warnings import warn
        super().__init__(distributions = distributions, 
                         model = model)
        s1:set[str] = set(var_names.keys())
        s2:set[str] = set(
            LinearRegressionCoreComponent.var_names.keys()
            )
        cls = LinearRegressionCoreComponent
        if var_names == dict():
            self.var_names = LinearRegressionCoreComponent.var_names
        else:
            if s1^s2==set():
                self.var_names = var_names
            elif s1-s2 != set():
                warn((
                    f"Unknown model variable found {s1-s2} and will be "
                    f"ignored. Valid variables are {cls.model_vars}"
                    ))
                self.var_names ={
                    k:v for k,v in var_names.items() if k not in s1-s2
                    }
                
            else:
                warn((
                    f"Missing model variables found {s2-s1} and will be "
                    "set to their defaults. Valid variables are "
                    f"{cls.model_vars}"
                    ))
                self.var_names = {
                    k:v for k,v in var_names.items() if k in s1^s2
                    } | {
                        k:v for k,v in cls.var_names if k in s2-s1
                    }
        
    def __call__(self)->None:
        r'''
            Add the variables to the model.
            
            Basic random variables are handled by the parent. Assumes a
            :code:`pymc.Model` context stack is open
        '''
        super().__call__()
        W = self.variables[self.var_names['slope']]
        X = self.variables[self.var_names['data']]
        b = self.variables[self.var_names['intercept']]
        f_name = self.var_names['equation']
        expr = W*X + b
        f = pymc.Deterministic(f_name, expr)
        self.variables[f_name] = f
        

class BESTCoreComponent(CoreModelComponent):
    r'''
        Core Model Component for the BEST group comparison model
        
        Injecting basic random variables into the model (the
        :code:`distributions` argument) is delegated to the parent
        :code:`CoreModelComponent`. This component inserts the following
        deterministics instead:
        
        .. math::
        
            \begin{array}{c}
                \Delta\mu_{i,j} =\ \mu_i-\mu_j \\
                \Delta\sigma_{i, j} = \sigma_i-\sigma_j\\
                E = \dfrac{\hat{x_i}-\hat{x_j}}{
                    \sqrt{
                        \frac{\sigma_i^2+\sigma_j^2}{2}
                        }
                }
            \end{array}
            
        By default only the :math::code:`\Delta\mu` quantity is added.
        Pass :code:`std_difference=True` to include
        :math::code:`\Delta\sigma` and :code:`effect_size=False` to add
        the effect size to the trace (will also automatically add
        :math::code:`\Delta\mu`). Used internally only. The :code:`BEST`
        class handles the external API.
        
        Example Usage:
        
        .. code-block:: python
        
            core_dists = dict(
                            ν = distribution(
                            pm.Exponential, "ν_minus_one", 1/29.0, 
                            transform = lambda e: e+1.0
                            ),
                            "obs_0" = distribution(
                                pm.Data, "y0", X0, mutable=False
                                ),
                            "obs_1" = distribution(
                                pm.Data, "y1", X1, mutable=False
                                ),
                            μ_0 = distribution(
                                    pm.Normal, "μ_0", 
                                        mu = X0.mean(),   
                                        sigma = 2*X0.std(),
                                        shape = X0.shape[-1]
                                    ),
                            σ_0 = distribution(
                                pm.Uniform, 'σ_0', 
                                lower = 1e-1,
                                upper = 10, 
                                shape=X0.shape[-1]),
                            μ_1 = distribution(
                                    pm.Normal, "μ_1", 
                                        mu = X1.mean(),   
                                        sigma = 2*X1.std(),
                                        shape = X1.shape[-1]
                                    ),
                            σ_1 = distribution(
                                pm.Uniform, 'σ_1', 
                                lower = 1e-1,
                                upper = 10, 
                                shape=X1.shape[-1])
            }
            likelihoods = [
                    LikelihoodComponent(
                        name = "y_obs_0",
                        observed = "obs_0",
                        distribution = pm.StudentT,
                        var_mapping = dict(
                            mu = 'μ_,
                            sigma = 'σ_1',
                            nu = 'ν',
                        )
                    ),
                    LikelihoodComponent(
                        name = "y_obs_0",
                        observed = "obs_0",
                        distribution = pm.StudentT,
                        var_mapping = dict(
                            mu = 'μ_,
                            sigma = 'σ_1',
                            nu = 'ν',
                        )
                    ),
                ]
            # Likelihoods passed to the builder instead
            BESTCoreComponent(
                            distributions = core_dists,
                            group_distributions = ...,
                            permutations = ..., 
                            std_difference = False, 
                            effect_magnitude = False,
            )

        
        Object Attributes:
        ------------------
        
            - | _group_distributions:tuple := Aggregated groupwise
                distributions
            
            - | _permutations:tuple := (Technically combinations) all
                possible unique pairs of levels for the categorical
                variable defining the groups. Tracks all possible
                pair-wise comparisons
            
            - | _derived_quantities:dict[str, str]=dict(means=[], stds =
                [], effect_magnitude = []) := Aggregated deterministics
                
            - | _std_difference:bool=False := If :code:`True` computes
                the :math::code:`\Delta\sigma` deterministic and adds it
                to the trace. Optional. Defaults to :code:`False`
            
            - | _effect_magnitude:bool=False := If :code:`True` compute
                the 'effect_size' deterministic quantity. Optional and
                defaults to :code:`False`
              
            - | variables:dict[str, Any] := A dictionary mapping
                internal variable names to a variable references. Added
                by the :code:`CoreModelObject` parent class. Enables
                different components to access variables other than
                those in their context (i.e. :code:`LikelihoodComponent`
                can access the variables to be linked to its shape
                parameters)
    '''
    
    __slots__ = ('_group_distributions', 'variables', '_permutations',
                 '_derived_quantities', '_std_difference', 
                 '_effect_magnitude', )
    
    def __init__(self,
                    distributions:dict[str, Distribution]=dict(),
                    model:Optional[pymc.Model] = None,
                    group_distributions = tuple(),
                    permutations = tuple(),
                    std_difference:bool = False,  
                    effect_magnitude:bool = False,
                  )->None:
        
        super().__init__(distributions = distributions, 
                         model = model)
        self._group_distributions = group_distributions
        self._permutations:tuple = permutations
        self._derived_quantities:dict[str, list] = dict(
            means = [], stds = [], effect_magnitude = [],
        )
        self._std_difference:bool = std_difference
        self._effect_magnitude:bool = effect_magnitude
        
    def __call__(self)->None:
        super().__call__()
        for permutation in self._permutations:
            pair_id = "({one_level}, {other_level})".format(
                one_level = permutation[0],
                other_level = permutation[1],
            )
            ν_name_mu = "{mean_symbol}{pair}".format(
                mean_symbol = 'Δμ',
                pair = pair_id,
            )
            diff = pymc.Deterministic(
                ν_name_mu, 
                self.variables[
                    f'μ_{permutation[0]}'
                    ] - self.variables[f'μ_{permutation[1]}'
                                                    ],
                    dims = 'dimensions'
            )
            self.variables[ν_name_mu] = diff
            self._derived_quantities['means'].append(ν_name_mu)
            
            if self._std_difference:
                v_name_std = "{std_symbol}{pair}".format(
                std_symbol = 'Δσ', pair = pair_id)
                std1=self.variables[f'σ_{permutation[0]}']
                std2 = self.variables[f'σ_{permutation[1]}']
                std_diff = pymc.Deterministic(v_name_std,
                                    std1-std2,
                                    dims='dimensions')
                self._derived_quantities['stds'].append(v_name_std)
                self.variables[v_name_std] = std_diff
            
            if self._effect_magnitude:
                v_name_magnitude = "{ef_size_sym}{pair}".format(
                    ef_size_sym='Effect_Size', pair=pair_id)
                effect_magnitude = pymc.Deterministic(
                    v_name_magnitude, diff/pymc.math.sqrt(

                        (std1**2+std2**2)/2), dims='dimensions')
                self.variables[v_name_magnitude] = effect_magnitude
                self._derived_quantities['effect_magnitude'].append(
                    v_name_magnitude
                )

@dataclass(kw_only=True, slots=True)
class CoreModelBuilder(ModelBuilder):
    
    r'''
        Core model builder object. 
        
        Sequentially composes the model object by inserting it's various
        components. Every model should supply at least a subclass of
        :code:`CoreModelComponent` and a list of :code:`Likelihood` components. All
        available components, in order are:


            - | core_model:CoreModelComponents := The basic structure of
                the model, including its equations and basic random
                variables (REQUIRED)

            - | link:LinkFunctionComponent := A link function to be
                applied to the model. (OPTIONAL)
            
            - | adaptors:ModeAdaptor := An 'output adaptor' component
                that splits the model output tensor into multiple
                subtensor and inserts them to the computation graph.
                (OPTIONAL)
            
            - | response:ResponseFunctionComponent := A response
                function whose model outputs will be passed through.
                (OPTIONAL)
            
            - | free_vars:FreeVariablesComponent := A additional
                variables component that inserts variables not
                explicitly involved in the models' core equations.
                (OPTIONAL)
                
            - | likelihoods:Sequence[LikelihoodComponent] := A
                collection of likelihood components, to be added to the
                model. (REQUIRED)
            
        Example usage:
        
        .. code-block:: python
        
            from bayesian_models.core import CoreModelBuilder
            from bayesian_models.core import LikelihoodComponent
            from bayesian_models.core import ResponseFunctionComponent
            from bayesian_models.core import FreeVariablesComponent
            from bayesian_models.core import ModelAdaptorComponent
            
            # Should be called indirectly via the director
            d = CoreModelBuilder(
                core_model = CoreModelComponent(),
                likelihoods=[LikelihoodComponent(), ...], 
                response  = ResponseFunctionComponent(), 
                free_vars = FreeVariablesComponent(),
                adaptor = ModelAdaptorComponent(), 
                )
                
        Object Attributes:
        ------------------

                - | core_model:Optional[CoreModelComponent] = None := 
                    The core model component. Should subclass
                    :code:`CoreModelComponent`
    
                - | likelihoods:Optional[list[LikelihoodComponent]] =
                    None := The likelihood component object, defining
                    the likelihood for the model

                - | model:Optional[pymc.Model] := The :code:`pymc.Model`
                    object. Should be created elsewhere and handed to
                    the builder
                    
                - | model_variables:dict[str, Any] := A general
                    catalogue of all the models' variables. Maps
                    variable names to references to the variables,
                    allowing for easy lookup
                
                - | free_vars:Optional[FreeVariablesComponent]=None :=
                    The component defining 'extra' random variables,
                    i.e. those not directly participating in the models'
                    equations
                
                - | adaptor:Optional[ModelAdaptorComponent]=None := A
                    model adaptor component which splits the models' raw
                    output into subtensors
                    
                - | response:Optional[ResponseFunctionComponent]=None :=
                    The ResponseFunction component, defining functions
                    that transform variables in the model to other
                    variables
                
                - | coords:dict[str,Any] = field(default_factory=dict)
                    := Coordinates for the model object itself. Passed
                    as a dict to the :code:`pymc` allowing for label
                    coordinates after inference. Collected from the data
                    object itself
                

                  
        Object Methods:
        ---------------
        
            - __post_init__()->None := Validate that minimal components
              are present in the model
              
            - _validate_likelihoods()->None := Validate that all shape
              parameters of the specified likelihoods are defined in the
              model stack
              
              .. danger::
                
                At present, little validation is being done, to assume that all of the distributions' parameters are correctly defined and the domains match. Responsibility for these validations in on the user
              
              
            - build()->None := Build the model by sequentially  calling
              separate components to add their variables to the context
              stack. Will also update the builders' internal catalogue
              of all variables present in the model
              
            - __call__()->None := Build the model updating or create the
              underlying :code:`pymc.Model` object
        

    '''
    
    core_model:Optional[CoreModelComponent] = None
    likelihoods:Optional[list[LikelihoodComponent]] = None
    model:Optional[pymc.Model] = field(init=False)
    model_variables:dict = field(default_factory = dict, init=False)
    free_vars:Optional[FreeVariablesComponent] = None
    adaptor:Optional[ModelAdaptorComponent] = None
    coords:dict[str,Any] = field(default_factory=dict)
    response:Optional[ResponseFunctionComponent] = None
    
    
    def __post_init__(self)->None:
        r'''
            Validate that minimal model components are present
        '''
        if any([
            self.core_model is None,
            self.likelihoods is None,
            self.likelihoods == list(),
        ]):
            raise ValueError(("Attempting model construcion without"
                              " minimal components. Core component and"
                              " likelihood component expected but "
                              f"received {self.core_model} and "
                              f"{self.likelihoods} instead"))
        self.model = None
            
    def _validate_likelihoods(self, user_spec:dict[str,str])->None:
        r'''
            Perform self validation, ensuring all likelihoods have all
            of their shape parameters linked to a model variable
        '''
        for i, likelihood in enumerate(self.likelihoods):
            spec = sorted(list(likelihood.var_mapping.keys()))
            required = sorted(
                extract_dist_shape(likelihood.distribution)
                )
            if spec != required:
                name = likelihood.name
                unspecified = set(required)-set(spec)
                raise ValueError(
                    ("Likelihood with unspecified shape found. All "
                     "supplied mappings should have exactly one key "
                     f"for each shape parameter. For the {i}th "
                     "likelihood parameter specified, with name "
                     f"{name}, mappings were given for shape "
                     f"parameters {spec} but the distribution has "
                     f"parameters {required}. {unspecified} shape"
                     "parameters have no mappings"))
            
    
    def build(self)->None:
        r'''
            Construct the model according to the specified components.
            
            Call specified components, in order, to add the variables of
            each to the model. Maintains an internal catalogue of
            variable names mapped to refs to the objects
        '''
        self.core_model()
        self.model_variables = self.model_variables|self.core_model.variables
        if self.free_vars is not None:
            self.free_vars()
            self.model_variables = \
                self.model_variables|self.free_vars.variables
        if self.adaptor is not None:
            self.adaptor(self.model_variables.get('f'))
            self.model_variables = self.model_variables| self.adaptor.variables
        if self.response is not None:
            self.response(self.model_variables)
            self.model_variables=self.model_variables|self.response.variables
        
        # Apply users specified mapping from likelihood shape params
        # to internal model objects. A lookup is performed, to translate
        # internal variable name i.e. 'f' to a ref to the object itself
        for likelihood in self.likelihoods:
            user_var_spec:dict[str, str] = likelihood.var_mapping
            user_vars:set[str] = set(v for _,v in user_var_spec.items())
            model_vars:set[str] = set(
                k for k in self.model_variables.keys())
            if not user_vars.issubset(model_vars):
                unbound:set[str] = user_vars-model_vars
                raise RuntimeError((
                    f"Variables {unbound}, specified in the likelihoods "
                    f"var_mapping are unbound. Variables {unbound} not "
                    "found in the model"
                ))
            likelihood_kwargs = {
                shape:self.model_variables[
                    modelvar 
                    ]  for shape, modelvar in user_var_spec.items()
                }
            # Lots of back-and-forth between builder and LikelihoodComponent object
            # See if it can be refactored out. Redux has a similar patter
            observed = self.model_variables.get(likelihood.observed)
            likelihood(observed,
                            **likelihood_kwargs)
    
    def __call__(self):
        if self.model is None:
            with pymc.Model(coords = self.coords) as model:
                self.build()
            self.model = model
        else:
            with self.model:
                self.build()
        return self.model
    

class ModelDirector:
    r'''
        Model construction object.
        
        Delegates model construction to a specified model builder.
        
        Example usage:
        
        .. code-block:: python
        
            # Will not work with empty objects. All except the first
            # two arguments are optional and can be ignored
            d = ModelDirector(
                CoreModelComponent(),
                LikelihoodComponent(),
                response_component  = ResponseFunctionComponent(),
                free_vars_component = FreeVariablesComponent(),
                adaptor_component = ModelAdaptorComponent(),
            )
        Object Attributes:
        -------------------
        
            - builder:Type[ModelBuilder] := The model builder object.
              Should be left to the default as only a single Builder is
              implemented
              
        Object Properties:
        ------------------

            - model:pymc.Model := The underlying :code:`pymc.Model`
              object, exposed to the user. Is not an actual object
              property, but exposes the one of the underlying builder
              object
              
        Object Methods:
        ---------------
        
            - __call__()->None := Calls the underlying model builder
              object to construct the actual model object
    '''
    builder:Type[ModelBuilder] = CoreModelBuilder
    
    def __init__(self,
                 core_component:CoreModelComponent,
                 likelihood_component:list[LikelihoodComponent],
                 response_component:Optional[ResponseFunctionComponent] = None,
                 free_vars_component:Optional[FreeVariablesComponent] = None,
                 adaptor_component:Optional[ModelAdaptorComponent] = None,
                 coords:Optional[dict] = dict(),
                 )->None:
    
        self.builder:ModelBuilder = self.builder(
            free_vars = free_vars_component,
            adaptor = adaptor_component,
            response = response_component,
            core_model = core_component,
            likelihoods = likelihood_component,
            coords = coords,
        )
        
    
    def __call__(self)->pymc.Model:
        r'''
            Build the model according the specified components
        '''
        return self.builder()
    
    @property
    def model(self)->Optional[pymc.Model]:
        return self.builder.model