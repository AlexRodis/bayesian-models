# Model builder module, containing tools to construct arbitrary
# models with a common interface
from dataclasses import dataclass, field, InitVar
from typing import Any, Type, Callable, Optional, Union
from abc import ABC, abstractmethod
import pymc
from collections import defaultdict, namedtuple
from functools import partial
from bayesian_models.data import Data
from bayesian_models.utilities import extract_dist_shape, invert_dict


Response = namedtuple("Response", ["name", "func", "target", "record"])
ModelVars = defaultdict(default_factory = lambda : None)
Real = Union[float, int]



@dataclass(slots=True, kw_only = True)
class Distribution:
    '''
        Data container class for user-specified distributions. Supply a
        distribution `dist` along with any desired arguments `dist_args` and
        `dist_kwargs`. If the distribution needs to be modified, for example
        by shifting it by some amount, supply a `dist_transform:Callable`
        Optionally  a `name` can be provided.
        NOTE: For data nodes, the `dist` argument should by the function 
        `pymc.Data`
    '''
    name:str = ''
    dist:Union[Type[
        pymc.Distribution], Type[Callable]
               ] = pymc.Distribution
    dist_args:tuple = tuple()
    dist_kwargs:dict = field(default_factory=dict)
    dist_transform:Optional[Callable] = None

    def __post_init__(self)->None:
        if any([
            not isinstance(self.name,str) or self.name == 'None',
            not isinstance(self.dist_args, tuple),
            not isinstance(self.dist_kwargs, dict),
        ]):
            raise ValueError(("Illegal values received"))
        

@dataclass(slots=True)
class FreeVariablesComponent:
    '''
        Component representing additional variables to be inserted to the
        model not explicitly involved in the core model itself, i.e. the
        equations describing the model. For example the noise parameter in
        linear regression:
        .. math::
            w0 \thicksim \mathcal{N}(0,1)
            w1 \thicksim \mathcal{N}(0,1)
            μ = X*w0+w1
            σ \thicksim \mathcal{N}(0,1)
            y \thicksim \mathcal{N}(μ, σ)
        
        The `dists` argument supplied is a `dict` or name to `Distribution`
        instances, representing the distributions to be inserted to the model
    '''
    
    variables:Any = field(init=False, default_factory=dict)
    dists:dict[str, Distribution] = field(default_factory=dict)
    
    def __call__(self)->None:
        for name, dist in self.dists.items():
            d = dist.dist(dist.name, *dist.dist_args,
                          **dist.dist_kwargs
                          )
            self.variables[name] = d 
            
            

@dataclass(kw_only=True, slots=True)
class LinkFunctionComponent:
    '''
        Add a link function to the model. The link function will be applied
        to the models' inputs. The link function is provided a dict whose
        key(s) are the desired internal names for the result of applying the
        function to the inputs. The `record` variable determines if the 
        result will be wrapped in a deterministic node. If True the result
        will be recorded in the trace but will consume additional memory
    '''
    link_function:dict[str, Callable] = field(default_factory=dict)
    record:bool = False
    variables:dict = field(init=False, default_factory=dict)
    
    def __call__(self):
        raise NotImplementedError()
    
@dataclass(slots=True)
class ResponseFunctions:
    '''
        Data container for Response functions. Accepts three mappings
        as dicts. All three map strings representing variable names for
        the result of the response to a parameter. The `functions` argument
        maps names to the actual functions themselves. The 
        `application_targets` parameter maps transformed variable names
        to the variables that are inputs to the transformation. The
        `records` parameter maps variable names to a boolean representing
        whether they should be recorded or not. `True` will wrap the
        result of the transform into a deterministic node, False will not
        The `application_targets` and `records` parameters can be partially
        or completely omitted. In this case, record defaults to True
        and the application_target default to 'f' a general name for the
        raw model output. If any variable is found in either 
        `application_targets` or `records` but not in `functions` an
        exception is raised, since not reasobly inference can be made for
        the tranform function instead. Example usage:
        
        .. code-block::
                    # Pass all parameters explicitly (recommended)
            In [4]: r = ResponseFunctions(
            ...:     functions = dict(exp = pc.math.exp, tanh = pc.math.tanh),
            ...:     records = dict(exp=True, tanh=False),
            ...:     application_targets = dict(exp="f", tanh="exp")
            ...:     )
            # Partially ommit application_targets using defaults 'f'
            In [5]: r = ResponseFunctions(
            ...:     functions = dict(exp = pc.math.exp, tanh = pc.math.tanh),
            ...:     records = dict(exp=True, tanh=False),
            ...:     application_targets = dict(tanh="exp")
            ...:     )
            # Pass the desired Callables leaving everything to their defaults
            # In this case two different response functions are applied
            # to the same input 'f'. Both are recorded with `pymc.Deterministic`
            # the name is the key provided
            In [6]: r = ResponseFunctions(
            ...:     functions = dict(exp = pc.math.exp, tanh = pc.math.tanh)
            ...:     )
    
    '''
    
    _sf:set[str] = field(init=False, default_factory=set)
    _st:set[str] = field(init=False, default_factory=set)
    _sr:set[str] = field(init=False, default_factory=set)
    _missing_records:set[str] = field(init=False, default_factory=set)
    _missing_targets:set[str] = field(init=False, default_factory=set)
    functions:dict[str, Callable] = field(default_factory=dict)
    application_targets:dict[str, str] = field(default_factory=dict)
    records:dict[str, bool] = field(default_factory=dict)
    _iter:Optional[Any] = field(init=False)
    
    
    def _validate_inputs(self)->None:
        '''
            Validate inputs by raises on incompatible specs
        '''
        if self.functions == dict():
            raise ValueError((
                "Attempting to add response functions but no functional "
                "mapping was provided. `functions` must be a dict mapping "
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
            packaged. Fields provided are `name`, `func`, `target` and
            `record`
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
        name = next(self._iter)
        return self.get_function(name)

    def __iter__(self):
        return self

@dataclass(slots=True)
class ResponseFunctionComponent:
    '''
        Model component representing Response functions. Accepts response
        functions specified via the `ResponseFunctions` class. Adds them
        to the model and maintains an internal catalogue for variables as
        `variables`
    '''
    responses:ResponseFunctions
    variables:dict=field(init=False, default_factory=dict)
    
    def __call__(self):
        pass
            
    
@dataclass(kw_only=True, slots=True)
class ModelAdaptorComponent:
    '''
        Adds an 'output adaptor' to the model, which 'splits' the models
        output tensor into other variables. The splitting logic is specified
        by the `var_mapping` argument, which is a dict whose keys are strings
        representing variable names, and whose items are Callables that
        accept and return `theano` tensors. The `record` boolean argument
        decides if the result of all "splits" will be wrapped in a
        deterministic node
    
    '''
    
    variables:dict = field(init=False, default_factory=dict)
    record:bool = True
    var_mapping:dict[str, Callable] = field(default_factory=dict)
    
    def __call__(self, output):
        '''
            Called by the builder to add the specified variables to the
            model. `output` is a ref to the tensor object to be split
            and should be what the `var_mapping` items accept
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
        Adds a likelihood to the model. `var_mapping` must be supplied
        defining which model variables map to which likelihood shape
        parameters. For example `distribution=pymc.Normal` and 
        `var_mapping=dict(mu = 'f', sigma='sigma')`. The keys of the
        `var_mapping` dict are strings which must exactly match the shape
        parameters of the likelihood. Values should be the internal names
        of model variables, specified during their creation. `name` and
        `observed` correspond to the name of the likelihood object itself
        and the name of the observed data node in the model. Should generaly
        be left to their defaults, except when the model has multiple
        likelihood/observed nodes
    '''
    name:str = 'y_obs'
    observed:str = 'outputs'
    distribution:Type[pymc.Distribution] = pymc.Normal
    var_mapping:dict = field(default_factory = dict)
    
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
        Core model object. Inserts basic random variables to the context
        stack, maintaining an internal catalogue of variables inserted
        for latter access. Concrete model objects should subclass this
        component and extend its call method by defining the models' 
        equations in deterministic nodes. If a concrete model produces some
        explicit quantity as an output, this should generally  be named
        'f'. 
    '''
    # Make this consistant with FreeVars by adding Distribution nametuples
    
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
                 "`bayesian_models.Distribution` instances. Received "
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
                "`bayesian_model.Distribution` instances. Received "
                f"illegal values {illegal}"))

    def __call__(self)->None:
        for key_name, dist in self.distributions.items():
            d = dist.dist(dist.name, *dist.dist_args,
                **dist.dist_kwargs
                )
            self.variables[key_name] = d
       
            
class LinearRegressionCoreComponent(CoreModelComponent):
    '''
        Core model component for linear regression, specified as:
        .. math::
            f = XW+b
        Inserts the deterministic variable 'f' to the model
    '''
    
    var_names = dict(
        slope = "W", intercept = "b", 
        data = "inputs", equation = "f"
    )
    model_vars = {"slope", "intercept", "data", "equation"}
    
    def __init__(self, distributions:dict[str, Distribution]=dict(),
                 variables:dict=dict(),
                 var_names:dict[str, str] = {},
                 model:Optional[pymc.Model] = None)->None:
        from warnings import warn
        super().__init__(distributions = distributions, 
                         variables = variables, model = model)
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
        super().__call__()
        W = self.variables[self.var_names['slope']]
        X = self.variables[self.var_names['data']]
        b = self.variables[self.var_names['intercept']]
        f_name = self.var_names['equation']
        expr = W*X + b
        f = pymc.Deterministic(f_name, expr)
        self.variables[f_name] = f


class NeuralNetCoreComponent(CoreModelComponent):
    '''
        WIP Raw testing only
    '''
    def __init__(self, n_layers:int = 3,
                 distributions:dict[str, Distribution] = dict(),
                 record:bool = False)->None:
        super().__init__(distributions = distributions)
        self.record = record
        self.n_layers = n_layers
    
    def __call__(self)->None:
        super().__call__()
        f = self.variables['inputs']
        for i in range(self.n_layers):
            f = pymc.math.dot(
                f,self.variables.get(f'W{i}')) + self.variables.get(
                    f'b{i}')
        f = pymc.Deterministic('f', f)
        self.variables['f'] = f

@dataclass(kw_only=True, slots=True)
class CoreModelBuilder(ModelBuilder):
    
    '''
        Core model builder object. Sequentialy composes the model object by
        inserting it's various components. Every model should supply at
        leat a subclass of `CoreModelComponent` and a list of `Likelihood`
        components. All available components, in order are:


            - core_model:CoreModelComponents := The basic structure of
            the model, including its equations and basic random variables
            (REQUIRED)

            - link:LinkFunctionComponent := A link function to be applied
            to the model. (OPTIONAL)
            
            - adaptors:ModeAdaptor := An 'output adaptor' component that
            splits the model output tensor into multiple subtensor and 
            inserts them to the computation graph. (OPTIONAL)
            
            - response:ResponseFunctionComponent := A response function
            whose model outputs will be passed through. (OPTIONAL)
            
            - free_vars:FreeVariablesComponent := A additional variables
            component that inserts variables not explicitly involved in
            the models' core equations. (OPTIONAL)
            
            - likelihoods:Sequence[LikelihoodComponent] := A collection
            of likelihood components, to be added to the model. (REQUIRED)
    '''
    
    model:Optional[pymc.Model] = None
    core_model:Optional[CoreModelComponent] = None
    model_variables:dict = field(default_factory = dict)
    free_vars:Optional[FreeVariablesComponent] = None
    adaptor:Optional[ModelAdaptorComponent] = None
    response:Optional[ResponseFunctionComponent] = None
    link:Optional[LinkFunctionComponent] = None
    likelihoods:Optional[list[LikelihoodComponent]] = None
    
    def __post_init__(self)->None:
        if any([
            self.core_model is None,
            self.likelihoods is None
        ]):
            raise ValueError(("Attempting model construcion without"
                              " minimal components. Core component and"
                              " likelihood component expect but "
                              f"received {self.core_model} and "
                              f"{self.likelihoods} instead"))
            
    def _validate_likelihoods(self, user_spec:dict[str,str])->None:
        '''
            Perform self validation, ensuring all likelihoods have all of
            their shape parameters linked to a model variable
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
        if self.link is not None:
            self.link()
            self.model_variables =  self.model_variables|self.link.variables
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
            self.response()
            self.model_variables=self.model_variables|self.response.variables
        
        # Apply users specified mapping from likelihood shape params
        # to internal model objects. A lookup is performed, to translate
        # internal variable name i.e. 'f' to a ref to the object itself
        for likelihood in self.likelihoods:
            user_var_spec:dict[str, str] = likelihood.var_mapping
            likelihood_kwargs = {
                shape:self.model_variables[modelvar]  for shape, modelvar in user_var_spec.items()
                }
            # Lots of back-and-forth between builder and LikelihoodComponent object
            # See if it can be refactored out
            outputs = self.model_variables.get('outputs')
            observed = outputs if outputs is not None else self.model_variables.get(
                                                                                    likelihood.observed)
            likelihood(observed,
                            **likelihood_kwargs)
    
    def __call__(self):
        if self.model is None:
            with pymc.Model() as model:
                self.build()
            self.model = model
        else:
            with self.model:
                self.build()
        return self.model
    

@dataclass(kw_only=True, slots=True)
class ModelDirector:
    '''
        Model construction object. Delegates model construction to a
        specified model builder
    '''
    # Maybe all keyword args is not a good idea
    builder_type:Type[ModelBuilder] = CoreModelBuilder
    builder:Optional[ModelBuilder] = None
    free_vars_component:Optional[FreeVariablesComponent] = None
    adaptor_component:Optional[ModelAdaptorComponent] = None
    response_component:Optional[ResponseFunctionComponent] = None
    link_component:Optional[LinkFunctionComponent] = None
    core_component:Optional[CoreModelComponent] = None
    likelihood_components:Optional[list[LikelihoodComponent]] = None
    
    def __post_init__(self)->None:
        
        self.builder = self.builder_type(
            free_vars = self.free_vars_component,
            adaptor = self.adaptor_component,
            response = self.response_component,
            link = self.link_component,
            core_model = self.core_component,
            likelihoods = self.likelihood_components
        ) # type:ignore
        
    
    def __call__(self)->pymc.Model:
        return self.builder() 