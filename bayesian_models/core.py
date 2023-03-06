from dataclasses import dataclass, field
from typing import Any, Type, Callable, Optional
from abc import ABC, abstractmethod
import pymc
import pytensor
from collections import namedtuple
from functools import partial

Distribution = namedtuple('distribution', ['name', 'dist', 'dist_args',
                                           'dist_kwargs'])



@dataclass(kw_only=True, slots=True)
class FreeVariables:
    '''
        Add additional variables to the model, that arent explicitly
        included in the models' equations themselves
    '''
    
    variables:Any = field(default_factory=dict)
    args:list[Distribution] = field(default_factory=list)
    
    def __post_init__(self)->None:
        self.variables = {arg.name: partial(arg.dist(
            arg.dist(arg.name, *arg.dist_args, **arg.dist_kwargs))
            ) for arg in self.args}
    
    
    def __call__(self):
        for var in self.variables:
            var()
            
            

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
    
    distribution:Type[pymc.Distribution] = pymc.Normal
    observed: Optional[Callable] = None
    var_mapping:dict = field(default_factory = dict)
    
    def __post_init__(self)->None:
        if self.var_mapping == dict():
            self.var_mapping = dict(
                mu = 'f', sigma = 'ε'
            )
    
    def __call__(self):
        y_obs = self.distribution(
            'y_obs', observed = self.observed,
            **self.var_mapping
        )#type: ignore

class ModelBuilder(ABC):
    '''
        Abstract base class for model builders 
    '''
    
    @property
    @abstractmethod
    def model_vars(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()

@dataclass(kw_only=True, slots=True)
class CoreModel:
    '''
        Core model object, which inserts basic variables to the model
    '''
    
    data_nodes:dict = field(default_factory=dict)
    variables:dict = field(default_factory=dict)
    model:Optional[pymc.Model] = None

    def __call__(self):
        for k, v in self.variables.items():
            v()
        


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
    likelihood:Optional[Likelihood] = None
    model_vars:dict = field(default_factory=dict)
    
    def __call__(self):
        with pymc.Model() as model:
            if self.link is not None:
                self.link()
                
            self.core_model()
            self.model_variables = \
                self.model_variables|self.core_model.variables
            
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
                
            self.likelihood.distribution(
                'y_obs', observed = None
            )
        
        self.model = model
        
        return self.model
    

@dataclass(kw_only=True, slots=True)
class ModelDirector:
    '''
        Model building director
    '''
    
    builder_type:Type[ModelBuilder] = CoreModelBuilder
    builder:Optional[ModelBuilder] = None
    free_vars_component:Optional[FreeVariables] = None
    adaptor_component:Optional[ModelAdaptor] = None
    response_component:Optional[ResponseFunction] = None
    link_component:Optional[LinkFunction] = None
    core_component:Optional[CoreModel] = None
    likelihood_component:Likelihood = Likelihood()
    
    def __post_init__(self)->None:
        
        self.builder = self.builder_type(
            free_vars = self.free_vars_component,
            adaptor = self.adaptor_component,
            response = self.response_component,
            link = self.link_component,
            core_model = self.core_component,
            likelihood = self.likelihood_component
        ) # type:ignore
        
    
    def __call__(self):
        self.builder() 