from dataclasses import dataclass, field, InitVar
from typing import Any, Type, Callable, Optional
from abc import ABC, abstractmethod
import pymc
import pytensor
from collections import namedtuple, defaultdict
from functools import partial
from bayesian_models.data import Data

Distribution = namedtuple('distribution', ['name', 'dist', 'dist_args',
                                           'dist_kwargs'])
ModelVars = defaultdict(default_factory = lambda : None)


@dataclass(slots=True)
class FreeVariables:
    '''
        Add additional variables to the model, that arent explicitly
        included in the models' equations 
    '''
    
    variables:Any = field(init=False, default_factory=dict)
    dists:list[Distribution] = field(default_factory=list)
    
    def __call__(self)->None:
        for dist in self.dists:
            d = dist.dist(dist.name, *dist.dist_args,
                          **dist.dist_kwargs
                          )
            self.variables[dist.name] = d 
            
            

@dataclass(kw_only=True, slots=True)
class LinkFunction:
    '''
        Add a link function to the model. The link function will be applied
        to the models' inputs
    '''
    variables:dict = field(default_factory=dict)
    link_function:Callable = lambda e: e
    
    def __call__(self):
        pass
    
    
@dataclass(kw_only=True, slots=True)
class ResponseFunction:
    '''
        Adds a response function to the model. The models' outputs will be
        passed through this function, prior to their inclusion to the
        likelihood
    '''
    variables:dict=field(default_factory=dict)
    response_function:Callable = lambda e: e

    def __call__(self):
        pass
    
@dataclass(kw_only=True, slots=True)
class ModelAdaptor:
    '''
        Adds an 'output adaptor' to the model, which maps the models' inputs
        to multiple new variables
    
    '''
    
    variables:dict = field(default_factory=dict)
    var_mapping:dict[str, Callable] = field(default_factory=dict)
    
    def __call__(self):
        pass



@dataclass(kw_only=True, slots=True)
class Likelihood:
    '''
        Adds the likelihood to the model. `var_mapping` must be supplied
        defining which model variables map to which likelihood shape
        parameters
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
class CoreModel:
    '''
        Core model object. Inserts basic random variables to the context
        stack, maintaining an internal catalogue of variables inserted
        for latter access. Concrete model objects should subclass this
        component and extend its call method by defining the models' 
        equations in deterministic nodes. Concrete models with 'predictive'
        style outputs should define the model output variable as 'f'
    '''
    
    variables:dict = field(default_factory=dict)
    model:Optional[pymc.Model] = None

    def __call__(self)->None:
        for k, v in self.variables.items():
            self.variables[k] = v()
        
class LinearRegressionCoreComponent(CoreModel):
    
    def __call__(self)->None:
        super().__call__()
        expr = self.variables['W']*self.variables['inputs'] + self.variables['b']
        f = pymc.Deterministic('f', expr)
        self.variables['f'] = f


@dataclass(kw_only=True, slots=True)
class CoreModelBuilder(ModelBuilder):
    
    '''
        Core model builder object. Sequentialy composes the model object by
        inserting sepperate components
    '''
    
    model:Optional[pymc.Model] = None
    core_model:Optional[CoreModel] = None
    model_variables:dict = field(default_factory = dict)
    free_vars:Optional[FreeVariables] = None
    adaptor:Optional[ModelAdaptor] = None
    response:Optional[ResponseFunction] = None
    link:Optional[LinkFunction] = None
    likelihoods:Optional[list[Likelihood]] = None
    
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
            self.adaptor()
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
            # Lots of back-and-forth between builder and Likelihood object
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
        Model building director
    '''
    # Maybe all keyword args is not a good idea
    builder_type:Type[ModelBuilder] = CoreModelBuilder
    builder:Optional[ModelBuilder] = None
    free_vars_component:Optional[FreeVariables] = None
    adaptor_component:Optional[ModelAdaptor] = None
    response_component:Optional[ResponseFunction] = None
    link_component:Optional[LinkFunction] = None
    core_component:Optional[CoreModel] = None
    likelihood_components:Optional[list[Likelihood]] = None
    
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