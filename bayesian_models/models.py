import pymc
import pymc as pm
import typing
from typing import Any, Union, Callable, Optional, Iterable
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .utilities import SklearnDataFrameScaler, tidy_multiindex
import xarray as xr
import arviz as az
import pytensor
from interval import Interval
import functools
from bayesian_models.math import ELU, GELU, SiLU, SWISS, ReLU
from bayesian_models.core import ModelDirector
from bayesian_models.data import Data
from dataclasses import dataclass, field

__all__ = (
        'BayesianModel',
        'BayesianEstimator',
        'BayesianNeuralNetwork',
        'BEST',
        'Layer',
        'MapLayer',
        'ReLU',
        'GELU',
        'ELU',
        'SWISS',
        'SiLU',
        'FreeAdditionLayer',
        )


class BayesianModel(ABC):
    '''
        Abstract base class for all Bayesian Models in pymc. All models
        should conform to the following general API:

        Methods:
        --------

            - __init__ := Begin initializing the model by setting all of the
            models parameters and hyperparameters. If a model has multiple
            variations, these are set here. The subclass should define valid
            hyperparameters as class attributes, along with their setter
            methods

            - __call__ := Initialized the object by specifying the full
            probability model for inference. This method should also receive
            the training data. These should be set using `pymc.Data`
            containers. For most simple, and predictive models, that accept
            some input information and attempt to predict some output
            quantity, the inputs should be declared as mutable shared tensors
            `pymc.Data(name, X, mutable=True)` and the `predict` method should
            invoke `pymc.set_data(inputs)` to replace the input tensor with a
            new one. For other special cases see the `predict` method. Should
            return the object itself for easy method chaining. Should set the
            objects' `initialized` property to signal the fit method can
            be called.

            - fit := Sample from the posterior according to the `sampler`
            argument. Forwards all other arguments to the sampler, sets the
            models' `idata` and `trained` flag, signaling the `predict` and
            other posterior methods can be called. `infer` is an alias for
            fit

            - predict := Produce output from the model. The implementation
            details of this method, vary depending on the model.

                - For predictive type models that attempt to predict some Y
                based on some X, this method should sample from the posterior
                predictive and return an appropriate Y.

                    - For most simple models should call `pymc.set_data(dict(
                    inputs=X_new))` and sample `y_obs` from the posterior
                    predictive

                    - For models with special predictive APIs like Gaussian
                    Process models, should declare a new variable as a shared
                    tensor `pymc.Data(X_new, mutable=True)` the first time
                    predictions are made, additional nodes corresponding to
                    the special API (i.e. `gp.condintional`) and any further
                    postprocessing nodes corresponding to data transforms and
                    response functions. Further calls to this method should
                    invoke `pymc.set_data` to replace the previous inputs with
                    the new ones followed by `pymc.sample_posterior_predictive`

                    - For non-predictive models (i.e. oversampling models,
                    statistical comparison models etc), data node(s) are
                    immutable and `predict` should perform the equivalent
                    operation for these models (i.e. render importance
                    decisions), yield augmented or rebalanced datasets etc.
            
            - plot_trace := Wrapper for `arviz.plot_posterior`

            - plot_posterior := Wrapper for `arviz.plot_posterior`

            - summary := Wrapper for `arviz.summary`
    '''


    @property
    @abstractmethod
    def save_path(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def initialized(self):
        pass

    @property
    @abstractmethod
    def trained(self):
        pass

    @property
    @abstractmethod
    def idata(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class BayesianEstimator(BayesianModel):
    '''
        Abstract base class for "predictive" style models. These models
        take some input information X and return some output predicted
        quantity Y.
    '''
    @property
    @abstractmethod
    def posterior_trace(self):
        pass

class IOMixin:
    '''
        Cooperative inheritance class that handles model saving and
        loading. Injects the `save` and `load` methods to subclasses
        and adds the `save_path` property

        Object Methods:
        ----------------

            save(self, save_path:Optional[str], method:str = ['pickle',
            'netcdf']) -> None := Save the model using one of two protocols.
            For `method='netcdf'` (default) saves the models' posterior
            as netcdf file. For `method='pickle'` (experimental) attempts
            so serialize the entire object using `pickle`.

            load(self, save_path:Optional[str]) -> None := Load a pretrained 
            model from memory, using the selected protocol. Only meaningful
            for models saved with `method='netcdf'`. For `method='pickle'`
            unpickle the object directly.

        Object Properties:
        -------------------

            - save_path:Optional[str] = None := Adds the `save_path`
            attribute, allowing users to specify a save path during object
            initialization
    '''


    def __init__(self, save_path: Optional[str] = None, *args,
        **kwargs)->None:
        self._save_path = save_path
        super().__init__(*args, **kwargs)
    
    @property
    def save_path(self)->Optional[str]:
        return self._save_path
    @save_path.setter
    def save_path(self, val:str)->None:
        self._save_path = val

    def save(self, save_path:Optional[str] = None,
            method:str = 'netcdf')->None:
        '''
            Save a trained model, allowing later reuse.

            Args:
            -----

                - save_path:Optional[str] = None := A string specifing
                the save location. Must provided either here, or during
                the objects' initialization phase. If a save_path is not
                provided either here or during object initialization, will
                raise an error.

                method:str='netcdf' := The saving method. Valid options are
                `netcdf` and `pickle`. For `method='netcdf'` (default) saves
                only the models' posterior trace and requires the
                initialization and call steps to be repeated after priors to
                loading. For `method='pickle'` (experimental) attempts to save
                the entire object be serializing.

                NOTE: No checks are made to verify that the loaded models'
                structure and the one infered from trace are compatible. Will
                likely result in unpredictable errors.


            Returns:
            --------
                - None

            Raises:
            -------

                - RuntimeError := If a `save_path` was not provided during
                initialization and `save` or `load` are called without a
                `save_path` argument
        '''
        spath = save_path if save_path is not None else \
            self.save_path
        if spath is None:
            raise RuntimeError(("`save_path` not specified"))
        if method=="pickle":
            import pickle
            with open(spath, 'wb') as file:
                pickle.dump(self, file)
        elif method=="netcdf":
            self.idata.to_netcdf(spath)
        else:
            raise RuntimeError((f"method={method} is an invalid option."
                    " Specify either 'pickle' to pickle the entire class"
                    ", or 'netcdf', to save only the posterior "
                    "(recommended)"
                    ))

    def load(self, load_path:str):
        '''
            Load a pretrained model.

            Args:
            -----

                - load_path:str := Path to the saved object

            Returns:
            ---------
                
                - BEST := The loaded model

            NOTE: Updates the models' `idata` and `trained` attributes
        '''
        # Add consistency checks to ensure the posterior
        # trace loaded is compatible with the initialized
        # model
        self.idata = az.from_netcdf(load_path)
        self.trained = True
        return self

class ConvergenceChecksMixin:
    '''
        WIP: Mixin for implementing posterior convergence checks. Currently
        only alerts if divergences are detected and reports their number

        Object Methods:
        ----------------
            Adds the following methods to the inheriting object:


            - _check_divergences_ := If there are non-zero divergences
            raises a warning and reports the divergences

        Object Properties:
        --------------------
            Adds the following  properties to the inheriting object:

            - divergences:Optional[xarray.DataArray] := A `xarray.DataArray`
            containing diverging samples
    '''


    def __init__(self, *args, **kwargs):

        self._divergences:Optional[xr.DataArray] = None
        super().__init__(*args, **kwargs)

    @property
    def divergences(self)->Optional[xr.DataArray]:
        return self._divergences
    @divergences.setter
    def divergences(self, val:xr.DataArray)->None:
        self._divergences = val

    def _check_divergences_(self):
        from warnings import warn
        assert self.idata is not None
        # Make this HTML
        self.divergences=self.idata.sample_stats['diverging']
        if self.divergences.sum() > 0:
            warn((f'There were {self.divergences.sum().values} '
                'divergences after tuning'))

class DataValidationMixin:
    '''
        Cooperative inheritance mixin adding data validation capapilities
        to inheriting models.

        Class Attributes:
        ------------------
            Adds the following class attributes to inheriting class:

            - nan_handling_values:tuple[str] := Defines valid values for the
            `nan_handling` argument
            
            
            - valid_deterministic_nodes:set := Valid deterministic nodes
            that could constitude elements of the `predict(var_names)`
            argument. 

        Static Methods:
        ----------------

            Adds the following static methods to inheriting classes:

            - exclude_missing_nan(df:pandas.DataFrame)->pandas.DataFrame:
            := Drops all rows with missing values.

            - impute_missing_nan(df:pandas.DataFrame)->None: := Imputes
            missings data values. NotImplemented and will raise an error

            - check_missing_nan(df:pandas.DataFrame, nan_handling:str
                )->bool := Returns a flag signaling missing values are
                present in the input dataset. Warns if True.

        Object Properties:
        ----------------

            Adds the following properties to inheriting objects:

            - nan_handling:str='exclude' := Selects the strategy in dealing
            with missing values. Valid options are 'exclude' and 'impute'. The
            latter is not implemented and will raise an error.

            - tidify_data:Optional[Callable[pandas.DataFrame,
                pandas.DataFrame]]=tidify_multindex := Optional Callable that
                takes as input DataFrame with multilevel indices and squashes
                them to a single level. By default will append all labels
                joined by dots i.e. `level_0_name.level_1_name.level_2_name`.
                WARNING! At present does not check that dataframe actually has
                a multilevel index and join all letters in single level inputs
                will sepperated by dots, i.e. 'ABCD' becomes 'A.B.C.D'.

            - scaler:Optional[SklearnDataFrameScaler]=None := Scaler object 
            handling possible rescaling of input data. Initialize the wrapper
            class `SklearnDataFrameScaler` with the scaler argument as one of
            `sklearn.preprocessing` scaler classes.
    '''


    nan_handling_values = set(("exclude", "impute"))
    valid_deterministic_nodes = set(('Δμ',"Δσ", "Effect_Size"))  

    def __init__(self,tidify_data:typing.Callable[...,Any]=tidy_multiindex,
                scaler:Optional[SklearnDataFrameScaler] = None,
                nan_handling:str='exclude',*args, **kwargs)->None:
            self._tidify_data = tidify_data
            self._scaler = scaler
            self._nan_present_flag:Optional[bool]=None
            if nan_handling in DataValidationMixin.nan_handling_values:
                self._nan_handling=nan_handling
            else:
                raise ValueError((f"{nan_handling} is not valid option for "
                    "`nan_handling`. Valid options are `exclude` and "
                    "`impute` "))
            super().__init__(*args, **kwargs)
        

    @property
    def nan_handling(self)->str:
        return self._nan_handling
    @nan_handling.setter
    def nan_handling(self, val:str)->None:
        self._nan_handling=val
    
    @property
    def tidify_data(self)->Callable[..., Any]:
        return self._tidify_data
    @tidify_data.setter
    def tidify_data(self, val:Callable[..., Any])->None:
        self._tidify_data = val
    @property
    def nan_present_flag(self)->Optional[bool]:
        return self._nan_present_flag
    @nan_present_flag.setter
    def nan_present_flag(self, val:bool)->None:
        self._nan_present_flag = val
    @property
    def scaler(self)->Optional[SklearnDataFrameScaler]:
        return self._scaler
    @scaler.setter
    def scaler(self, val:SklearnDataFrameScaler)->None:
        self._scaler = val


class BESTBase:
    '''
        Base class for class level variable injection to the BEST
        model
    '''
    WarperFunction = Callable[[pd.DataFrame], pd.Series]
    std_upper:float = 1e1
    std_lower:float = 1e0
    ν_λ:float = 1/29.0
    ν_offset:float = 1
    ddof:int = 1
    std_diffusion_factor:int = 2
    μ_mean:WarperFunction = lambda df, axis=0: df.mean(axis=axis)
    μ_std:WarperFunction = lambda df,ddof=ddof,\
        η=std_diffusion_factor,ϵ=1e-4, axis=0:df.std(
            ddof=ddof, axis=axis).replace({0.0:ϵ})*η
    jax_device:str = 'gpu'
    jax_device_count: int =1

@dataclass(slots=True)
class BEST(BESTBase):
    '''
        Bayesian Group difference estimation with pymc. The implementation
        is based on the official pymc documentation. The model assumes
        StudentT likelihood over observations for added robustness.
        
        Class Attrs:
        ------------
        
            - std_upper=1e1 := Upper bound for the uniform prior over
            group-wise standard deviations
            
            - std_lower=1e0 := Lower bound for the uniform prior over
            group-wise standard deviations
            
            - ν_λ=1/29.0 := Exponential shape factor for shape parameter
            ν's prior distribution
            
            - ν_offset=1 := Offset factor for shape parameter
            ν's prior distribution
            
            - μ_mean:Callable[pandas.DataFrame,pandas.Series]=
            lambda df, axis=0: df.mean(axis=axis) := Callable that
            returns a series of means for prior setting. Must take
            a `pandas.DataFrame` as input and return a series or
            numpy vector
            
            - μ_std:Callable[pandas.DataFrame,pandas.Series]=
            lambda df,ddof=1, axis=0, η=2, ϵ=1e-4:df.std(
            ddof=ddof, axis=axis).replace({0.0:ϵ})*η:=
            Callable that returns a series of means for prior
            setting. Must take a `pandas.DataFrame` as input and return
            a series or numpy vector or appropriate size
            
            - ddof:int=1 := Degrees of freedom parameter for empirical
            pooled standard_deviation estimation. Must be non negative.
            Optional Defaults to 1.
            
            - std_diffusion_factor:float=2 := A prior diffusion parameter
            applied to priors over standard deviations. Means are set to
            this parameter times the pooled empirical standard deviations.
            Optional. Defaults to 2.
            
            - jax_device:str='gpu' := Specify which device `numpyro`
            uses for MCMC sampling. Set to either 'gpu' or 'cpu'. Currently
            unused.
            
            - jax_device_count:int=1 := The number of devices to be used
            by jax during inference. Optional. Defaults to False. Is
            ignored unless the `pymc.sampling.jax.sample_numpyro_nuts`
            is passed during object call. Currently ignored
            
            
            
        
        Object Attrs:
        -------------

            - group_var:`pandas.Index` := An index specifying the factor
            column in the provided DataFrame. Must be a valid column. Note,
            if `tidify_data` is not None, it will be invoked prior to all
            calls, hence `group_var` should reflect the variables' name after
            tidification (i.e. ('chemical', 'antioxidants', 'squalene')->
            'chemical.antioxidants.squalene')
            
            - effect_magnitude:bool=False := Whether to compute an 
            'effect size' during inference. This metric is somewhat more
            abstract than direct differences and is defined as
            .. math::
                ES=\dfrac{\Delta\mu_{1,2}}{\sqrt{\dfrac{
                    \sigma_{1}^{2}\sigma_{2}^{2}}{2}}}
            
            - std_differences:bool=False := Selects whether or not to estimate
            standard deviation differenes between groups. Optional. Defaults
            to False. If `effect_magnitude=True` this value is ignored and the
            difference is computed automatically.
            
            - common_shape:bool=True := If True make the simplfying assumption
            that all input dimentions have the same shape parameter `ν`.
            Else, assign distinct shape parameters to all dimentions of the
            input array. Optional. Defaults to True. If switched off, warns of
            likely unidentifiable model.

            - multivariate_likelihood:bool=True := Flag signaling whether a
            multivariate likelihood or a univariate one. Optional.Defaults to
            False.
            
            - save_name:Optional[str]=None := A string specifying location and
            filename to save the model's inference results. Optional. If set
            during objects' construction, the `save` method may be called
            without a save path.
            
            - nan_handling:str='exclude' := Specify how missing values are
            handled. Either `'exclude'` to remove all rows with missing values
            or `'impute'` to attempt to impute them. Optional. Defaults to
            `'exclude'`. `'impute'` not implemented and raises an error if
            specified. Ignored if no missing values are present.
            
            - tidify_data:Optional[Callable[pandas.DataFrame,pandas.DataFrame
            ]]=tidy_multiindex := Callable that takes the input DataFrame and
            squashes MultiLevel indices for ease of display. Defaults to
            custom `tidy_multiindex` which squashes all levels of the input
            index, sepperated by '.'. Optional. Defaults to None. Note
            `tidy_multindex` does not check for the present of multilevel
            indices and if called with non-multilevel index dataframes, will
            result in erratic behavior
            
            - scaler:Optional[Callable[pandas.DataFrame, pandas.DataFrame]]=
            std_scale := A Callable that scales the input DataFrame. Use the
            special wrapper class `SklearnDataFrameScaler` with the `scale`
            argument being the `sklearn.preprocessing` scaler class, to return
            `pandas.DataFrames` instead of `numpy` arrays.
            
            - idata:Optional[arviz.InferenceData]=None := 
            `arviz.InferenceData` object containing the results of model 
            inference
            
            - trained:bool=False := Sentinel signaling whether the model has
            been trained or not. Defaults to False. Should be switched on
            after calling `fit`.

            - initialized:bool=False := Sentinel signaling whether the model
            has been full initialized of not. Optional. Defaults to False.
            Should be switched on after calling the object.

            - var_names:dict[str:list[str]] := A dictionary with the 'means',
            'stds' and 'effect_magnitude' keys, mapped to lists of the
            variables

            - _permutations:Optional[Iterable[tuple[str,str]]]=None := An
            iterable of groups per `group_var`. Contains all unique pairs
            of unique values of `group_var`.

            - _n_perms:Optional[int]=None := The number of groups.

            _ levels:Optional[Iterable[str]]=None := Group levels, 
            corresponding to all unique values of `group_var`

            - _ndims:Optional[int]=None := Number of input features. Only
            meaningfull if `common_shape` is `False`.

            - num_levels:Optional[int]=None := The number of unique groups/
            values of `group_var`.

            - coords := dict-like of dimention labels `xarray` coords. Will be
            infered from inputs and used to label the posterior

            - _group_distributions:Optional[Dict[str, pymc.Distribution]]=None
            := A dict used internally to map infered distributions. Defaults
            to None.

            - _model:Optional[pymc.Model] := The model

        Object Methods:
        ---------------

            - __init__:= Begin initializing the object by setting all
            options, parameters and hyperparameters
            
            - __call__ := Initialize the model by specifying the full
            probability model accords to options passed to `__init__`

            - fit := Perform inferece on the model

            - predict := Compute results of group comparisons and return
            a dictionary mapping derived metrics to `pandas.DataFrame`s
            containing inference summary

            - summary := Wrapper for `arviz.summary`. Returns summary results
            of model inferece

            - plot_posterior := Wrapper for `arviz.plot_posterior`. Plot
            the infered posterior

            - plot_trace := Wrapper for `arviz.plot_trace`. Plot inferece
            results

            - _consistency_checks_ := Check model parameters and 
            hyperparameters are consistent and compatible

            - _preprocessing_ := Preprocess data. Rescale and tidify data,
            detect and handle missing values, infer parameters coords,
            permutations etc

            - _fetch_differential_permutations_ := Compute all unique pairs
            of groups. Sets objects

        
        Class Methods:
        --------------
            Setters for all class attributes. Named set_attribute
        
    '''
    
    group_var:Optional[Union[str, int]] = field(
        init=True, default_factory=lambda : None)
    _permutations:Any = field(init=False, default_factory=lambda : None)
    _n_perms:Optional[int] = field(init=False, 
                                   default_factory=lambda : None)
    _data_dimentions:Any = field(init=False, default_factory=lambda : None)
    effect_magnitude:bool = False
    std_difference:bool = False
    common_shape:bool = True
    multivariate_likelihood:bool = False
    _levels:Any = field(init=False, default_factory=lambda : None)
    _ndims:Any = field(init=False, default_factory=lambda : None)
    num_levels:Optional[int] = field(init=False, 
                                     default_factory=lambda : None)
    _coords:Any = field(init=False, default_factory=lambda : None)
    _group_distributions:Any = field(init=False, default_factory=dict)
    _idata:Optional[az.InferenceData] = field(
        init=False, default_factory=lambda : None)
    _data_processor:Data = field(
        init=False, default_factory = lambda : Data(cast=None)
        )
    _model:Optional[pymc.Model] = field(init=False, 
                                        default_factory=lambda : None)
    var_names:dict[str, list] = field(init=False, 
                      default_factory = lambda : dict(
                          means=[], stds=[], effect_magnitude=[])
                      )
    _initialized:Optional[bool] = field(default_factory=lambda : False)
    _trained:Optional[bool] = field(default_factory=lambda : False)
    nan_present_flag:Optional[bool] = field(
        init=False, default_factory=lambda :None)
    
    
    # def _consistency_checks_(self):
    #     '''
    #         Ensures all options specified are mutualy compatible
    #     '''
    #     from warnings import warn
    #     if not self.common_shape:
    #         warn(("Allowing independant degrees of freedom for all"
    #         " features may result in unidentifiable models, overfitting" 
    #         " and multimodal posteriors"))
    #     if self.multivariate_likelihood and not self.common_shape:
    #         warn(("Degrees of freedom parameter for a multivariate"
    #         " StudentT must be a scalar. `common_shape` will be "
    #         "ignored"))
    #         self.common_shape=not self.common_shape
    
    
    # def __init__(self, effect_magnitude:bool=False,
    #             std_difference:bool=False,
    #             tidify_data:typing.Callable[...,Any]=tidy_multiindex,
    #             scaler:Optional[SklearnDataFrameScaler] = None,
    #             common_shape:bool=True, nan_handling:str='exclude',
    #             save_path:typing.Optional[str]=None,
    #             multivariate_likelihood:bool = False,
    #             ):
    #     super().__init__(tidify_data= tidify_data, scaler = scaler, 
    #     nan_handling=nan_handling, save_path = save_path)
    #     self.group_var = None
    #     self._permutations = None
    #     self._n_perms = None
    #     self._data_dimentions = None
    #     self.effect_magnitude = effect_magnitude
    #     self.std_difference = std_difference or effect_magnitude
    #     self.common_shape = common_shape
    #     self.multivariate_likelihood = multivariate_likelihood
    #     self._consistency_checks_()
    #     self._levels=None
    #     self._ndims=None
    #     self.num_levels=None
    #     self._coords=None
    #     self._group_distributions=dict()
    #     self._idata=None
    #     self._model=None
    #     self.var_names=dict(means=[], stds=[], effect_magnitude=[])
    #     self._initialized = False
    #     self._trained = False
    
    @property
    def coords(self)->Optional[dict[str,Any]]:
        return self._coords
    @coords.setter
    def coords(self, val:dict[str, Any])->None:
        self._coords = val

    @property
    def idata(self)->Optional[az.InferenceData]:
        return self._idata

    @idata.setter
    def idata(self, val:az.InferenceData)->None:
        self._idata = val

    @property
    def model(self)->Optional[pymc.Model]:
        return self._model

    @model.setter
    def model(self, val:pymc.Model)->None:
        self._model = val
    
    @property
    def save_path(self)->Optional[str]:
        return self._save_path

    @save_path.setter
    def save_path(self, val:str)->None:
        self._save_path = val 


    @property
    def trained(self)->bool:
        return self._trained

    @trained.setter
    def trained(self, val:bool)->None:
        self._trained = val

    @property
    def initialized(self)->bool:
        return self._initialized

    @initialized.setter
    def initialized(self, val:bool)->None:
        self._initialized = val
            
    
    def _preprocessing_(self, data):
        '''
            Handled data preprocessing steps by 1. checking and 
            handling missing values, 2. collapsing multiindices
            3. extracting groups, 4. extracting feature labels
            (coordinates)
            
            Args:
            -----
            
                - data:pandas.DataFrame := The input data
                
                - group_var:str := Label for the categorical
                variable that determines the groups
                
            Returns:
            -------
            
                - None
        '''
        
        self.levels = [
            e for e in next(data[:,self.group_var].unique())[1]
            ]
        self.num_levels=len(self.levels)
        self.features = [
            k for k in data.coords()[list(data.coords().keys())[1]] if k != self.group_var
            ]
        self._ndims=len(self.features)
        
        groups = {
            level: data[:, self.group_var]==level for level in self.levels
        }

        self._groups = groups
        self._coords = dict(
            dimentions = data.coords()[data.coords().keys()[0]] 
        )
            
        self._fetch_differential_permutations_()
    
    def _fetch_differential_permutations_(self):
        '''
            Generate all possible unique pairs of levels
            of the target factor.
        '''
        from itertools import combinations
        self._permutations=list(combinations(self.levels ,2))
        self._n_perms=len(self._permutations)
    
    @classmethod
    def set_std_upper(cls,val:float)->None:
        cls.std_upper=val
    
    @classmethod
    def set_std_lower(cls,val:float)->None:
        cls.std_lower=val
    
    @classmethod
    def set_shape_factor(cls, val:float)->None:
        cls.ν_λ = val
    
    @classmethod
    def set_shape_offset(cls, val:float)->None:
        cls.ν_offset=val
        
    @classmethod
    def set_mean_of_means(cls,
                          func:typing.Callable[...,np.typing.NDArray]
                          )->None:
        cls.μ_mean=func
    
    @classmethod
    def set_std_of_means(cls, 
                         func:typing.Callable[...,np.typing.NDArray]
                         )->None:
        cls.μ_std=func
    
    @classmethod
    def set_jax_device(cls, device:str)->None:
        if not device in ('gpu','cpu'):
            raise ValueError(('Valid values for `jax.device` are'
                             ' either "gpu" or "cpu". Received '
                             f' {device} instead'))
        else:
            cls.jax_device=device
    
    @classmethod
    def set_jax_device_count(cls, val:int)->None:
        cls.jax_device_count=val
    
    @classmethod
    def set_diffusion_factor(cls, val:float)->None:
        cls.diffusion_factor=val
        
    @classmethod
    def set_degrees_of_freedom(cls,val:int)->None:
        cls.ddof=val
    
    @staticmethod
    def warp_input(data, row_indexer, column_indexer, transform,
                  unwrap:bool=True):
        '''
            Utility method that selects a subset of the data and applies
            `transform` one it.
            
            Args:
            -----
                
                - data:pandas.DataFrame:= The dataframe to process
                
                - row_indexer:pandas.Index:= Indexer for row selection
                
                - column_indexer:pandas.Index := Indexer for column
                selection
                
                - transform:Callable[pandas.DataFrame, Union[pandas.DataFrame,
                pandas.Series]]:= A callable that takes a `pandas.DataFrame`
                and returns a data structure
                
                - unwrap:bool=True := When True unwraps the resulting
                `pandas.DataFrame` to the underlying `numpy.NDArray`
                object. Optional. Defaults to True and returns a numpy
                array object
                
            Returns:
            --------
            
                - ndf:pandas.DataFrame := A subset of the original
                DataFrame
                
                - warped_input:pandas.Series:=
                The output of `transform`. Generally a `pandas.Series`
                of empirical means, or standard deviations
        '''
        
        selected = data.loc[row_indexer, column_indexer]
        transformed = transform(selected)
        if unwrap:
            warped_input=transformed.values
        else:
            warped_input=transformed
        return warped_input
        
    
    def __call__(self, data, group_var:Union[str, tuple[str]]):
        '''
            Initialized the full probability model

            Args:
            -----

                - data:pandas.DataFrame := Input information. The data is 
                assumed to have a single categorical column, defining the 
                variable to group by

                - group_var:Union[str,tuple[str]] := A valid pandas indexer
                defining the column specifying the variable to group by.
                Note if `tidify_data` is set, it will be used before accessing
                the variable. Thus the indexer here should be the squashed
                version

            Returns:
            --------
                
                - obj:BEST := The object
        '''
        self.group_var = group_var
        data = self._data_processor(data)
        # data =self.tidify_data(data) if self.tidify_data is not None \
        #     else data
        self._preprocessing_(data)
        # if self.scaler is not None:
        #     data = self.scaler(data.loc[:,self.features])
        with pymc.Model(coords=self._coords) as BEST_model:
            
            σ_lower=BEST.std_lower
            σ_upper=BEST.std_upper
            
            if self.common_shape:
                ν  = pm.Exponential("ν_minus_one", 
                                    BEST.ν_λ) + BEST.ν_offset
            for level in self.levels:
                obs = pymc.Data(f"y_{level}",
                    BEST.warp_input(data, self._groups[level],
                        self.features, lambda df:df), mutable=False)
                μ = pymc.Normal(f'μ_{level}',
                              mu = BEST.warp_input(
                                  data, self._groups[level], 
                                  self.features,
                                  BEST.μ_mean
                                  ),
                                
                              sigma = BEST.warp_input(
                                  data, self._groups[level], 
                                  self.features,
                                  BEST.μ_std
                                  ),
                              shape=self._ndims
                             )
                σ = pymc.Uniform(f'{level}_std', lower=σ_lower,
                upper=σ_upper, shape=self._ndims)
                
                if not self.common_shape:
                    ν  = pm.Exponential(f"ν_minus_one_{level}",
                    BEST.ν_λ,
                                        shape=self._ndims) + \
                                            BEST.ν_offset
                if not self.multivariate_likelihood:
                    y_obs = pymc.StudentT(f'y_obs_{level}', nu=ν,
                    mu =μ, lam=σ**-2, observed= obs)
                else:
                    cov = pytensor.tensor.diag(σ**-2)
                    y_obs = pymc.MvStudentT(f'y_obs_{level}', nu=ν,
                    mu =μ, scale=cov, observed= obs)
                
                self._group_distributions[level]=dict( mean=μ, std=σ,
                shape=ν)
        
        with BEST_model:
            
            for permutation in self._permutations:
                pair_id = "({one_level}, {other_level})".format(
                one_level = permutation[0],
                other_level = permutation[1]
                )
                v_name_mu = "{mean_symbol}{pair}".format(mean_symbol='Δμ', 
                                                  pair = pair_id)
                # We may not need self._group_distributions for anything
                diff = pymc.Deterministic(v_name_mu,
                                          self._group_distributions[
                                            permutation[0]]['mean']-\
                                            self._group_distributions[
                                                permutation[1]]['mean'],
                                          dims='dimentions')
                
                self.var_names['means'].append(v_name_mu)
                # Possible feature enhancement: Add custom Deterministic
                # nodes for used-defined derived quantities
                
                if self.std_difference:
                    v_name_std="{std_symbol}{pair}".format(
                    std_symbol = 'Δσ', pair = pair_id)
                    std1=self._group_distributions[permutation[0]][
                        'std']
                    std2=self._group_distributions[permutation[1]][
                        'std']
                    std_diff = pymc.Deterministic(v_name_std,
                                          std1-std2,
                                          dims='dimentions')
                    self.var_names['stds'].append(v_name_std)
                    
                if self.effect_magnitude:
                    v_name_magnitude = 'Effect_Size('+permutation[0]+\
                        ','+permutation[1]+')'
                    v_name_magnitude = "{ef_size_sym}{pair}".format(
                        ef_size_sym='Effect_Size', pair=pair_id)
                    
                    effect_magnitude = pymc.Deterministic(
                        v_name_magnitude, diff/np.sqrt(
                            (std1**2+std2**2)/2), dims='dimentions')
                    self.var_names['effect_magnitude'].append(
                        v_name_magnitude)
        self._model=BEST_model
        self.initialized = True
        return self
    

    def fit(self, *args, sampler=pymc.sample , **kwargs)->az.InferenceData:
        '''
            Perform inference by sampling from the posterior. `infer` is
            an alias for `fit`

            Args:
            -----

                - sampler=pymc.sample := The `pymc` sampler to run MCMC
                with. Optional. Defaults to `pymc.sample`

                - *args:tuple[Any] := Arguements to be forwarded to the 
                sampler

                - **kwargs:dict[str, Any] := Optional keyword arguments to
                be forwarded to the sampler

            Returns:
            --------

                - idata:arviz.InferenceData := The results of inference
        

            Raises:
            -------

                - RuntimeError := If called before `fit` has been called

            Warns:
            ------

                - If any divergences are detected
        '''
        if not self.initialized:
            raise RuntimeError(("Cannot run inference. Model is not "
                "initialized"))
        with self._model:
            self.idata=sampler(*args, **kwargs)
        self._check_divergences_()
        self.trained = True
        return self.idata
    
    infer = fit

    def predict(self, var_names:typing.Sequence[str]=['Δμ'],
            ropes:typing.Sequence[tuple[float, float]]=[(-.1,.1)],
            hdis:typing.Sequence[float]=[.95],
            multilevel_on:str='[',  extend_summary:bool=True):
        '''
            Calculate inter-group differences according to the ROPE+HDI
            criterion. Results a DataFrame with a new column labeled
            'Significance' containing the decision for the variable indicated
            by the row. Decisions according to the ROPE_HDI criterior are
            rendered as follows:
                
                - HDI in ROPE := Not Significant

                - HDI & ROPE == () := Significant

                - HDI & ROPE != HDI ^ () := Withold decision ('Indeterminate')

            Args:
            -----

                - var_names:Iterable[str]=['Δμ'] := An iterable of derived
                metrics to return. Optional. Defaults to returning expected
                difference. Default names are 'Δμ' for the difference of
                means, 'Δσ' for the difference of standard deviations and
                'Effect_Size' for Kruschkes effect size.

                - ropes:Iterable[tuple[float, float]] := An Iterable of
                length-2 tuples of floats, defining the Region Of Practical
                Equivalence. Each rope is applied to all features for every
                variable in `var_names`.

                - hdis:Iterable[float] := An iterable of non-negative floats
                defining the probability threshold for the credible interval.
                Is applied to all features for each variable in `var_names`

                - multilevel_on:str='[' := A separator defining the multilevel
                index. `pymc` by default concatinates the label according to
                the form: {var_name}[{feature_label}]. The argument will
                reindex them in a multilevel fashion of the form (var_name,
                feature_label) in the resulting dataframe. Set to None to
                disable this behavior.

                extend_summary:bool=True := If True the new Significance
                column extends the summary dataframe. Else return a new 
                dataframe containing only the Significance results. Optional.
                Defaults to True and returns an extended version of the
                summary.
        '''
        
        if not self.trained:
            raise RuntimeError(("Cannot make predictions. Model has "
                "not been trained. Call the objects `fit` method first")
                )
        if 'Δσ' in var_names and not self.std_difference:
            raise RuntimeError(("var_names contains the variable 'Δσ'"
                                " but `std_difference` was not set to"
                                " True. Ensure you specify "
                                "`std_differnce=True` when initializing"
                                " if you need to return standard "
                                "deviations" ))
        if 'Effect_Size' in var_names and not self.effect_magnitude:
            raise RuntimeError(("var_names contains the variable 'Δσ'"
                                " but `effect_magnitude` was not set to"
                                " True. Ensure you specify "
                                "`effect_magnitude=True` when "
                                "initializing"
                                " if you need to return standard "
                                "deviations" ))
        if not all(
            (len(var_names) != len(ropes), len(ropes)!=len(hdis))
            ):
            from warnings import warn
            warn(("Length of variables, ropes and hdis not equal. The"
                  " shortest value will be considered"))
        results=dict()
        null_interval = Interval(0,0, lower_closed=False, 
                                 upper_closed=False)
        for var_name, rope,hdi in zip(var_names,ropes, hdis):
            raw_summary = az.summary(self.idata, var_names=[var_name],
            filter_vars='like', hdi_prob=hdi)
            rope=Interval(*rope)
            out=[]
            for idx,row in raw_summary.iterrows():
                ci=Interval(row[2],row[3])
                if ci in rope:
                    out.append("Not Significant")
                elif ci & rope != null_interval:
                    out.append("Indeterminate")
                else:
                    out.append("Significant")
            significance = pd.DataFrame(data=np.asarray(out),
                    index=raw_summary.index, columns=["Significance"])
            if extend_summary:
                result=pd.concat([
                    raw_summary, significance], axis=1)
                    
            else:
                result = significance
            if multilevel_on:
                nidx=np.stack(result.index.str.split("[").map(
                    np.asarray))
                mlvlidx=[nidx[:,0],np.asarray(
                    [e[:-1] for e in nidx[:,1]])]
                result.index=mlvlidx
            results[var_name]=result
        return results


    def summary(self, *args, **kwargs)->Union[
            xr.DataArray, pd.DataFrame]:
        '''
            Wrapper for `arviz.summary(idata)`

            Args:
            -----

                - *args:tuple[Any] := Arguments to be forwarded to 
                `arviz.summary`

                - **kwargs:dict[str, Any] := Keyword arguments to be
                forwarded to `arviz.summary`
            
            Returns:
            ---------

                - summary_results:xarray.DataArray := Summary results
        '''
        if not self.trained:
            raise RuntimeError(("Model has not been trained. Ensure "
                "you've called the objects `fit` method first"))
        return az.summary(self.idata, *args, **kwargs)


    def plot_posterior(self, *args, **kwargs):
        '''
            Wrapper for `arviz.plot_posterior`
        Args:
        -----

            - *args:tuple[Any] := Arguments to be forwarded to 
            `arviz.plot_posterior`

            - **kwargs:dict[str, Any] := Keyword arguments to be
            forwarded to`arviz.plot_posterior`
        '''
        if not self.trained:
            raise RuntimeError(
                "Cannot plot posterior. Model is untrained")
        return az.plot_posterior(self.idata, *args, **kwargs)


    def plot_trace(self, *args, **kwargs):
        '''
            Wrapper for `arviz.plot_trace`
        Args:
        -----

            - *args:tuple[Any] := Arguments to be forwarded to 
            `arviz.plot_trace`

            - **kwargs:dict[str, Any] := Keyword arguments to be
            forwarded to `arviz.plot_trace`
        '''
        if not self.trained:
            raise RuntimeError("Cannot plot trace. Model is untrained")
        
        return az.plot_trace(self.idata, *args, **kwargs)

class Layer:

    transfer_functions:dict[str, Optional[Callable]] = dict(
        exp = pymc.math.exp,
        softmax = pymc.math.softmax,
        sigmoid = pymc.math.invlogit,
        linear = lambda e:e,
        tanh = pymc.math.tanh,
        relu = ReLU,
        leaky_relu = functools.partial(ReLU, leak=1e-2),
        parameteric_relu = functools.partial(ReLU, leak=1),
        elu = ELU,
        swiss = SWISS,
        gelu = GELU,
        selu = SiLU,
    )
    transfer_function_names:set = set(transfer_functions.keys())
    transfer_function_callables = set([v for k,v in \
                                       transfer_functions.items()])

    '''
        Object representing a Neural Network layer

        Object Attributes:
        -------------------

            - n_neurons:int := The number of neurons in the layer

            - name:Optional[str] := An identifier for this layers

            - weight_priors:pymc.Continuous=pymc.Normal := Prior distribution
            for the weight in the layer

            - bias_priors:pymc.Continuous=pymc.Normal := Prior distribution
            for the layer biases

            - weight_prior_args:Optional[tuple[Any]] := Arguments to be be
            forwarded to the weights prior distribution

            - weight_prior_kwargs:Optional[tuple[aNY]] := Keyword arguments
            to be forwarded to the weights prior distribution

            - bias_prior_args:Optional[tuple[Any]] := Arguments to be be
            forwarded to the biases prior distribution

            - bias_prior_kwargs:Optional[tuple[aNY]] := Keyword arguments
            to be forwarded to the biases prior distribution


            - activation_function:Callable[..., Any] := The networks'
            activation function

        Object Methods:
        -----------------

            - __call__(self,X) := Adds the layer to the models' graph.
            Bugged and currently not working

            - __repr__(self) := Return a string representation of the
            object
        
        '''

    def _validate_inputs_(self, *args:tuple[Any],
                          **kwargs:dict[str,Any])->None:
        
        '''
            Validate object inputs and options
        '''
        if not isinstance(args[0], int):
            raise TypeError("n_neurons argument must be a positive "
                            f"integer. Received {type(args[0])} "
                            "instead")
        elif args[0]<=0:
            raise ValueError(("n_neurons must be a postive integer."
                              f"Received {args[0]} instead"))



    def __init__(self,n_neurons:int,
                weight_priors:pymc.Continuous = pymc.Normal,
                bias_priors:pymc.Continuous = pymc.Normal,
                weight_prior_args:tuple[Any]=(0,),
                weight_prior_kwargs:dict[str,Any] = dict(sigma=1),
                bias_prior_args:tuple[Any] = (0,),
                bias_prior_kwargs:dict[str, Any] = dict(sigma = 1),
                activation_function:Callable[...,Any] = pymc.math.tanh,
                name:Optional[str] = None)->None:
        self.n_neurons = n_neurons
        self.name = name
        self.weight_priors = weight_priors
        self.bias_priors = bias_priors
        self.weight_prior_args = weight_prior_args
        self.weight_prior_kwargs = weight_prior_kwargs
        self.bias_prior_args = bias_prior_args
        self.bias_prior_kwargs = bias_prior_kwargs
        self.activation_function = activation_function
    
    def __call__(self,X):
        W = self.weight_priors(f"W_{self.name}", *self.weight_prior_args,
                               **self.weight_prior_kwargs,
                               shape = (X.shape[1], self.n_neurons))
        b = self.bias_priors(f"b_{self.name}", *self.bias_prior_args,
                             **self.bias_prior_kwargs,
                             shape = self.n_neurons)
        node = self.activation_function(pytensor.tensor.dot(X, W)+b)
        return node
        
    def __repr__(self)->str:
        return ((f"Deep Layer <name = {self.name}, "
        f"n_neurons = {self.n_neurons}>, weights = {self.weight_priors}"
        f"({self.weight_prior_args}, {self.weight_prior_kwargs}), "
        f"biases = {self.bias_priors}({self.bias_prior_args}, "
        f"{self.bias_prior_kwargs}), "
        f"transfer function = {self.activation_functions}"))


class MapLayer:
    '''
        Special pseudo-layer that splits network outputs into several
        named variables. Useful for allowing the outputs of a network
        to be shape variables of a distribution

        WIP
    '''

    def __init__(self, splitter:dict[str, Callable],
                 name:str="output_map")->None:

        self.splitter = splitter
        self._name = name
    
    @property
    def name(self)->str:
        return self._name
    @name.setter
    def name(self, val:str)->None:
        self._name = val

    def __call__(self, tensor):
        for var, splitter in self.splitter.items():
            v = pymc.Deterministic(var, splitter(tensor))

class FreeAdditionLayer:
    '''
        Specify additional variables for a network model, i.e. noise
        parameters for the likelihood

        WIP
    '''


    def __init__(self, vars:Iterable[str], name:str):
        self._name = name
        self.vars = vars
    
    def __call__(self):
        pass




class BayesianNeuralNetwork:

    '''

        Class representing a dense Bayesian Neural Network (BNN)

        Object Attributes:
        --------------------

        - layers:Iterable[Layer] := A Sequence of Layer objects, representing
        the layers of the network. The input layer is addded automatically
        and should not be included here. The output layer should be included
        as the last layer object

        Object Properties:
        -------------------

        - trained:bool=False := Sentinel value indicating if the models'
        `fit` method has been called

        - initialized:bool=False := Sentinel value indicating if the models'
        `__init__` and `__call__` methods have been called

        - idata:Optional[arviz.InferenceData]=None := The models' posterior
        samplers, returned by the `fit` method

        - model:Optional[pymc.Model] := The `pymc.Model` object representing
        the underlying model

        - posterior_predictive:Optional[xarray.Dataset] := Samples from the
        posterior predictive. The result of calling the models' `predict`
        method

        Object Methods:
        ----------------

        - __init__(self, layers:Iterable[Layer]) := Begin object
        initialization by defining the networks' architecture

        - __call__(self, X:xarray.DataArray, Y:xarray.DataArray) := Complete
        object initialization by specifying the full probability model

        - fit(self, sampler=pymc.sample, *args:[tuple[Any]],
            **kwargs:dict[str,Any])->arviz.InferenceData := Perform inference
            on the model, by using `sampler` to sample from the posterior.
            `infer` is an alias for this method

        - predict(self, Xnew:xarray.DataArray)->xarray.Dataset := Predict
        using the models' posterior predictive distribution.



    '''

    def __init__(self, layers:Iterable[Layer]):
        self.layers = layers
        # self.layers:dict[str,Iterable[Layer]] 
        # i.e. dict(main = [Layer(), Layer()])
        # self.likelihood_map = dict(model_output = mu, s = sigma)
        self._trained:bool = False
        self._initialized:bool = False
        self._model:Optional[pymc.Model] = None
        self._idata:Optional[az.InferenceData] = None
        self._posterior_predictive:Optional[xr.Dataset] = None
    
    @property
    def trained(self)->bool:
        return self._trained
    @trained.setter
    def trained(self, val:bool)->None:
        self._trained = val
    @property
    def initialized(self)->bool:
        return self._initialized
    @initialized.setter
    def initialized(self, val:bool)->None:
        self._initialized = val
    @property
    def model(self)->Optional[pymc.Model]:
        return self._model
    @model.setter
    def model(self, val:pymc.Model)->None:
        self._model = val
    @property
    def idata(self)->Optional[az.InferenceData]:
        return self._idata
    @idata.setter
    def idata(self, val:az.InferenceData)->None:
        self._idata = val
    @property
    def posterior_predictive(self)->Optional[xr.Dataset]:
        return self._posterior_predictive
    @posterior_predictive.setter
    def posterior_predictive(self, val:xr.Dataset)->None:
        self._posterior_predictive = val

    def __call__(self, X_train:xr.DataArray, Y_train:xr.DataArray):
        '''
            Fully initialize the object by specifying the full probability
            model for inference

            Args:
            -----

                - X_train:xarray.DataArray := Model inputs

                - Y_test:xarray.DataArray := Model outputs (rank-2 tensor
                only)

            Returns:
            ---------

                - self := Return the object itself
        '''
        self._x_coords:dict = force_extract_coords(X_train)
        self._y_coords:dict = force_extract_coords(Y_train)
        self._coords:dict = self._x_coords|self._y_coords
        with pymc.Model(coords=self._coords) as bnn_model:
            inputs = pymc.Data("inputs", X_train, mutable = True)
            outputs = pymc.Data("outputs", Y_train, mutable = False)
            L = inputs
            # for region_id, layers in self.layers.items():
            #   for layer in layers:
            #       L = layer(L)
            #   L = pymc.Deterministic('region_id', L)
            for layer in self.layers:
                L = layer(L)
            y = pymc.Deterministic("y", L)
            y_obs = pymc.Dirichlet('y_obs', y, observed = outputs)
        self.initialized = True
        self.model = bnn_model
        return self
    

    def fit(self, *args, sampler=pymc.sample,
            **kwargs)->az.InferenceData:
        '''
            Perform inference on the model

            Args:
            ------

                - sampler:Callable=pymc.sample := The sampler to use for
                inference. Optional. Defaults to the default `pymc.sample`

                - *args:tuple[Any] := Positional arguements to be forwarded
                for the sampler

                - **kwargs:dict[str, Any] := Keyword arguments to be forwarded
                to the sampler

            Returns:
            ---------

                - self.idata:arviz.InferenceData := Samples from the
                posterior. The output of calling `sampler`
        '''
        with self.model:
            self.idata = sampler(*args, **kwargs)
        self.trained = True
        return self.idata
    
    def predict(self, X_new, *args, **kwargs)->xr.DataArray:
        '''
            Predict on new inputs, using the models' posterior predictive

            Args:
            ------

                - X_new:xarray.DataArray := The new points to predict

            Returns:
            --------

                - trace:xarray.Dataset := The `xarray.Dataset` containing
                the models' prediction. Only samples the output layer
                by default
        '''
        with self.model:
            pymc.set_data(dict(
                inputs = X_new,
            ))
            self.trace = pymc.sample_posterior_predictive(
                self.idata, var_names=["y"],*args,**kwargs)
        return self.trace


    def __repr__(self)->str:
        lstring:str=""
        for layer in self.layers:
            lstring += str(layer)
        return ((f"BayesianNeuralNetwork <{lstring}>"))



if __name__ == '__main__':
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
    obj=BEST()(df, "group")
    obj.fit(chains=2, draws=1000, tune=1000)
    obj.predict(
        var_names=['Δμ', 'Δσ'],
        ropes = [(0,1),(0,1)],
        hdis=[.95,.95]
    )