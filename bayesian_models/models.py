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
#   This module contains model definitions

import pymc as pm, pymc
import typing
from typing import Any, Union, Callable, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xarray as xr
import arviz as az
import pytensor
from interval import interval
from bayesian_models.math import ELU, GELU, SiLU, SWISS, ReLU
from bayesian_models.core import ModelDirector
from bayesian_models.core import LikelihoodComponent, distribution
from bayesian_models.core import Distribution, ResponseFunctions
from bayesian_models.core import BESTCoreComponent
from bayesian_models.core import ResponseFunctionComponent
from bayesian_models.data import Data
from dataclasses import dataclass, field
from warnings import warn

__all__ = (
        'BayesianModel',
        'BayesianEstimator',
        'BEST',
        'ReLU',
        'GELU',
        'ELU',
        'SWISS',
        'SiLU',
        )


class BayesianModel(ABC):
    r'''
        Abstract base class for all Bayesian Models in pymc.
        
        Defines a common API for all model objects

        Object Methods:
        ----------------

            - | __init__ := Begin initializing the model by setting all
                of the models parameters and hyperparameters. If a model
                has multiple variations, these are set here. The
                subclass should define valid hyperparameters as class or
                object attributes. Object level properties should be
                preferred where a user is expected to need to change
                these parameters for a single dataset. Class level
                parameters are preferred where a user is likely to want
                to apply the same model to multiple data (with the same
                parameters)

            - | __call__ := Initialize the object by specifying the full
                probability model for inference. This method should also
                receive the training data. These should be set using
                :code:`pymc.Data` containers. The data received should
                be placed in a :code:`Data` container, supplied by the
                :code:`data` submodule for preprocessing/ For most
                simple, and predictive models, that accept some input
                information and attempt to predict some output quantity,
                the inputs should be declared as mutable shared tensors
                `pymc.Data(name, X, mutable=True)` and the
                :code:`predict` method should invoke
                :code:`pymc.set_data(inputs)` to replace the input
                tensor with a new one. For other special cases see the
                :code:`predict` method. Should return the object itself
                for easy method chaining. Should set the objects'
                :code:`initialized` property to signal that the fit
                method can be called.

            - | fit := Sample from the posterior according to the
                :code:`sampler` argument. Forwards all other arguments
                to the sampler, sets the models' :code:`idata` and
                :code:`trained` flag, signaling the :code:`predict` and
                other posterior methods can be called. :code:`infer` is
                an alias for fit

            - | predict := Produce output from the model. The
                implementation details of this method, vary depending on
                the model.

                - | For predictive type models that attempt to predict
                    some Y based on some X, this method should sample
                    from the posterior predictive and return an
                    appropriate Y.

                    - | For most simple models should call
                        :code:`pymc.set_data(dict(inputs=X_new))` and
                        sample :code:`y_obs` from the posterior
                        predictive.
                        
                        Example:
                        
                        .. code-block:: python

                            def predict(self, Xnew, ...):
                                with self.model:
                                    pm.set_data(dict(inputs=Xnew))
                                    y = pm.sample_posterior()
                                return y

                    - | For models with special predictive APIs like
                        Gaussian Process models, should declare a new
                        variable as a shared tensor `pymc.Data(X_new,
                        mutable=True)` the first time predictions are
                        made, additional nodes corresponding to the
                        special API (i.e. :code:`gp.condintional`) and
                        any further postprocessing nodes corresponding
                        to data transforms and response functions.
                        Further calls to this method should invoke
                        :code:`pymc.set_data` to replace the previous
                        inputs with the new ones followed by
                        :code:`pymc.sample_posterior_predictive`.
                        
                        Example:
                        
                        .. code-block:: python
                        
                            def predict(self, Xnew):
                                with self.model:
                                    if not self.predictive_initialized:
                                        predicted = pm.Data(
                                            'predicted',
                                            Xnew, 
                                            mutable=True
                                            )
                                    ...
                                    ynew = pm.sample_posterior_predictive()
                                    return ynew
                                    


                    - | For non-predictive models (i.e. oversampling
                        models, statistical comparison models etc), data
                        node(s) are immutable and :code:`predict` should
                        perform the equivalent operation for these
                        models (i.e. render importance decisions), yield
                        augmented or rebalanced datasets etc.
            
            - plot_trace := Wrapper for :code:`arviz.plot_posterior`

            - plot_posterior := Wrapper for :code:`arviz.plot_posterior`

            - summary := Wrapper for :code:`arviz.summary`
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
    r'''
        Abstract base class for "predictive" style models.
        
        These models take some input information X and return some
        output predicted quantity Y.
    '''
    @property
    @abstractmethod
    def posterior_trace(self):
        pass

@dataclass(slots=True)
class ModelIOHandler:
    r'''
        Model save/load component
        
        Handles saving and loading pre-trained models
        
        Example usage:
        
            .. code-block:: python
            
                # Create and train a model
                
                obj = BEST(save_path=...)(df, "some_var")
                obj.fit()
                obj.save()
                # Or supply the save path directly
                # obj.save(save_path=...)
                nobj = BEST()(df, "some_var")
                nobj.load(save_path=...)
                

        Object Methods:
        ----------------

            - | save(save_path:Optional[str], method:str =
                ['pickle','netcdf']) -> None := Save the model using one
                of two protocols. For :code:`method='netcdf'` (default)
                saves the models' posterior as netcdf file. For
                :code:`method='pickle'` (experimental) attempts to
                serialize the entire object using :code:`pickle`.

            - | load(save_path:Optional[str]) -> None := Load a
                pretrained model from memory, using the selected
                protocol. Only meaningful for models saved with
                :code:`method='netcdf'`. For :code:`method='pickle'`
                unpickle the object directly.
                
            .. danger::
                
                Currently these is no validation being done to ensure the loaded model matches the initialized one. It is the responsibility of the user to ensure the model is being specified and loaded correctly

        Object Properties:
        -------------------

            - | accepted_methods:set[str] := Defines acceptable
                methods/protocols to save with. Possible save methods
                are 'pickle' and 'netcdf'. 
                
                .. caution::
                
                    Due to known issues (see #5) the 'pickle' method is
                    not recommended

            Private Object Properties:
            ==========================

                - | _save_path:Optional[str]=None := The path to save
                    the model to. Either accessed from the model
                    attribute or supplied during method call
                    
                - | _class:Optional[BayesianModel]=None := A reference
                    to the object being saved
    '''
    _model:Optional[pymc.Model] = field(init=False, default=None)
    _class:Optional[BayesianModel] = field(init=False,  default=None)
    _save_path:Optional[str] = field(
        init=True, default = None)
    accepted_methods:set[str] = field(
        init=True, default_factory = lambda : {"netcdf", "pickle"}
    )
    
    @property
    def save_path(self)->Optional[str]:
        return self._save_path
    @save_path.setter
    def save_path(self, val:str)->None:
        self._save_path = val

    def save(self, save_path:Optional[str] = None,
            method:str = 'netcdf')->None:
        r'''
            Save a trained model, allowing later reuse.

            Args:
            -----

                - save_path:Optional[str] = None := A string specifing
                the save location. Must provided either here, or during
                the objects' initialization phase. If a save_path is not
                provided either here or during object initialization,
                will raise an error.

                - | method:str='netcdf' := The saving method. Valid
                    options are :code:`netcdf` and :code:`pickle`. For
                    :code:`method='netcdf'` (default) saves only the
                    models' posterior trace and requires the
                    initialization and call steps to be repeated after
                    priors to loading. For :code:`method='pickle'`
                    (experimental) attempts to save the entire object be
                    serializing.

                    NOTE: No checks are made to verify that the loaded
                    models' structure and the one inferred from trace 
                    are compatible. Will likely result in unpredictable 
                    errors.


            Returns:
            --------
                - None

            Raises:
            -------

                - | RuntimeError := If a :code:`save_path` was not
                    provided during initialization and :code:`save` or
                    :code:`load` are called without a :code:`save_path`
                    argument
        '''
        spath = save_path if save_path is not None else \
            self.save_path
        if spath is None:
            raise RuntimeError((":code:`save_path` not specified"))
        if not self._class.trained:
            raise RuntimeError((
                    "Cannot save model. Model has not been trained"
                ))
        if method not in self.accepted_methods:
            raise RuntimeError((f"method={method} is an invalid option."
                    " Specify either 'pickle' to pickle the entire class"
                    ", or 'netcdf', to save only the posterior "
                    "(recommended)"
                    ))
        if method=="pickle":
            import pickle
            with open(spath, 'wb') as file:
                pickle.dump(self._class, file)
        elif method=="netcdf":
            self._class.idata.to_netcdf(spath)
        else:
            raise RuntimeError((
                "Unable to save model. Unspecified runtime error"
            ))
            

    def load(self, load_path:str,
             method:Optional[str]='netcdf'):
        r'''
            Load a pretrained model. This feature is experimental and 
            will likely fail

            Args:
            -----

                - load_path:str := Path to the saved object

            Returns:
            ---------
                
                - BEST := The loaded model

            NOTE: Updates the models' :code:`idata` and :code:`trained` attributes
        '''
        warn((
            "Warning! This feature is experimental and poorly tested. "
            "No validations are done for consistency between the loaded "
            "inference data the model."
        ))
        # Add consistency checks to ensure the posterior
        # trace loaded is compatible with the initialized
        # model
        if method == 'netcdf':
            idata = az.from_netcdf(load_path)
        
        self._class.idata=idata
        self._class._initialized = True
        self._class._trained = True
        return self._class

@dataclass(slots=True)
class ConvergencesHandler:
    r'''
        Convergence checking object. Called with the idata as the result
        of MCMC, detects divergences and warnings. 
        
        WIP: Should detect the divergences more extensively in the
        future
        
        Object Properties:
        -------------------
        
            - | divergences:xarray.DataArray := Diverging posterior
                samples
        
        Object Methods:
        ---------------
        
            - | __call__(idata) := Investigate the posterior trace for
                divergences. Warns if any diverging samples are found
    '''

    _divergences:Optional[xr.DataArray] = field(
        init = False, default=None)


    @property
    def divergences(self)->Optional[xr.DataArray]:
        return self._divergences
    @divergences.setter
    def divergences(self, val:xr.DataArray)->None:
        self._divergences = val

    def __call__(self, idata:xr.DataArray):
        '''
            Execute divergences checks
        '''
        from warnings import warn
        self.divergences = idata.sample_stats['diverging']
        if self.divergences.sum() > 0:
            warn((f'There were {self.divergences.sum().values} '
                'divergences after tuning'))
        return self.divergences

class BESTBase:
    '''
        Base class for class level variable injection to the BEST
        model
        
        Class Attributes:
        -----------------
        
            - WarperFunction := Type definition
            
            - | std_upper:float = 1e1 := Upper boundary for the standard
                deviations prior
            
            - | std_lower:float = 1e0 := Lower boundary for standard
                deviations prior
                
            .. note::
            
                The original model has exceedingly wide priors:
                
                .. math::
                
                    \sigma_k \thicksim \mathcal{U}(10^{-3}, 10^{3})

                These defaults are not sensible for most applications.
                [0.1-10] boundaries are implemented as defaults instead
                
            - | ν_λ:float = 1/29.0 := Exponential decay parameter for
                the :code:`ν` prior
            
            - | ν_offset:float = 1 := Offset parameter for the :code:`ν`
                prior. Since :math:`\nu \in [1,+ \infty])` this should
                be left unchanged
                
            - | ddof:int = 1 := Degrees of freedom parameter for
                empirical pooled standard deviations
            
            - | std_diffusion_factor:int = 2 := Scalar multiplier for
                empirical pooled standard deviations on the :code:`μ`
                prior. Controls how diffuse the prior is
            
            - | zero_offset:float = 1e-4 := Small offset parameter to
                avoid numerical errors with pooled standard deviations
            
            - | jax_device:str = 'gpu' := :code:`numpyro` parameter for
                alternate sampling. Controls which device
                :code:`numpyro` will use
            
            - | jax_device_count: int =1 := :code:`numpyro` setting.
                Number of devices to be used for HMC sampling (parallel)
                
            .. attention::
                Due to persistent problems with the :code:`numpyro`
                dependency these parameters are ignored
    '''
    WarperFunction = Callable[[pd.DataFrame], pd.Series]
    std_upper:float = 1e1
    std_lower:float = 1e0
    ν_λ:float = 1/29.0
    ν_offset:float = 1
    ddof:int = 1
    std_diffusion_factor:int = 2
    zero_offset:float = 1e-4
    jax_device:str = 'gpu'
    jax_device_count: int =1

@dataclass(slots=True)
class BEST(BESTBase):
    r'''
        Bayesian Group difference estimation with pymc.
        
        Kruschke's `Bayesian Estimation Superceeds the t-Test (BEST)        <https://pubmed.ncbi.nlm.nih.gov/22774788/>`_ model
        implementation for estimating differences between groups. The
        implementation is based on the `official pymc documentation
        <https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html>`_.
        
        The model assumes StudentT likelihood over observations for
        added robustness. See the discussions section :ref:`BEST Theory`
        for a more in depth explanation and motivation for this model
        
        Class Attributes:
        -----------------
        
            - WarperFunction := Type definition
            
            - | std_upper:float = 1e1 := Upper boundary for the standard
                deviations prior
            
            - | std_lower:float = 1e0 := Lower boundary for standard
                deviations prior
                
            .. note::
            
                The original model has exceedingly wide priors:
                
                .. math::
                
                    \sigma_k \thicksim \mathcal{U}(10^{-3}, 10^{3})

                These defaults are not sensible for most applications.
                [0.1-10] boundaries are implemented as defaults instead
                
            - | ν_λ:float = 1/29.0 := Exponential decay parameter for
                the :code:`ν` prior
            
            - | ν_offset:float = 1 := Offset parameter for the :code:`ν`
                prior. Since :math:`\nu \in [1,+ \infty])` this should
                be left unchanged
                
            - | ddof:int = 1 := Degrees of freedom parameter for
                empirical pooled standard deviations
            
            - | std_diffusion_factor:int = 2 := Scalar multiplier for
                empirical pooled standard deviations on the :code:`μ`
                prior. Controls how diffuse the prior is
            
            - | zero_offset:float = 1e-4 := Small offset parameter to
                avoid numerical errors with pooled standard deviations
            
            - | jax_device:str = 'gpu' := :code:`numpyro` parameter for
                alternate sampling. Controls which device
                :code:`numpyro` will use
            
            - | jax_device_count: int =1 := :code:`numpyro` setting.
                Number of devices to be used for HMC sampling (parallel)
                
            .. attention::
                Due to persistent problems with the :code:`numpyro`
                dependency these parameters are ignored
            
        Object Attributes:
        ------------------
        
            - | group_var:str := Coordinate label for the
                categorical variable to group by
                
                .. danger::
                
                    Due to issues with the underlying :code:`pymc`
                    implementation and limitations of
                    :code:`numpy.isnan` the categorical variables'
                    levels should be recorded as something castable to a
                    float, if numpy array is used to store the data. It
                    is recommended that a :code:`pandas.DataFrame` be
                    used instead
            
            - | effect_magnitude:bool=False := Whether to compute an
                'effect size' during inference. This metric is somewhat
                more more difficult to interpret, since it is no longer
                in the original units and is defined as:
            
                .. math::
                
                    E=\dfrac{\Delta\mu_{1,2}}{\sqrt{\dfrac{
                        \sigma_{1}^{2}\sigma_{2}^{2}}{2}}}
            
            - | std_differences:bool=False := Selects whether or not to
                estimate standard deviation differences between groups.
                Optional. Defaults to False. If
                :code:`effect_magnitude=True` this value is ignored and
                the difference is computed automatically.
            
            - | common_shape:bool=True := If True make the simplifying
                assumption that all input dimensions have the same shape
                parameter :code:`ν`. Else, assign distinct shape
                parameters to all dimensions of the input array.
                Optional. Defaults to True. If switched off, warns of
                likely unidentified model.

            - | multivariate_likelihood:bool=True := Flag signaling
                whether to use a multivariate likelihood or a univariate
                one. Optional.Defaults to False (use univariate
                likelihoods).
                
                .. note::
                    The multivariate likelihood is always assumed to have a diagonal scale matrix. Hence this option is equivalent to independent univariates with the common degrees of freedom assumption, but is more computationally expensive and should be avoided
            
            - | save_name:Optional[str]=None := A string specifying
                location and filename to save the model's inference
                results. Optional. If set during objects' construction,
                the :code:`save` method may be called without an
                explicit :code:`save_path` argument.
            
            
            - | idata:Optional[arviz.InferenceData]=None :=
                :code:`arviz.InferenceData` object containing the
                results of model inference. Becomes set after calling
                the :code:`fit` method
            
            - | trained:bool=False := Sentinel signaling whether the
                model has been trained or not. Defaults to False. Should
                be switched on after calling :code:`fit`. Prevents
                :code:`predict` from being called on an object that has
                not been trained.

            - | initialized:bool=False := Sentinel signaling whether the
                model has been full initialized. Defaults to False.
                Should be set after the object is called. Prevents
                :code:`fit` and :code:`predict` from being called prior
                to complete initialization.
                
        Private Attributes:
        ===================

            - | var_names:dict[str:list[str]] := A dictionary with
                the 'means', 'stds' and 'effect_magnitude' keys,
                mapped to lists of the variables

            - | _permutations:Optional[Iterable[tuple[str,str]]]=None
                := An iterable of groups per :code:`group_var`. Contains
                all unique pairs of unique values of :code:`group_var`.

            - | _n_perms:Optional[int]=None := The number of groups.

            - | levels:Optional[Iterable[str]] = None := Grouping
                variable levels. All unique values of :code:`group_var`.

            - | _ndims:Optional[int]=None := Number of input
                features. Only meaningful if :code:`common_shape` is
                :code:`False`.

            - | num_levels:Optional[int]=None := The number of unique
                groups/ values of :code:`group_var`.

            - | coords := dict-like of dimension labels :code:`xarray`
                coords. Will be inferred from inputs and used to
                label the posterior

            - | _group_distributions:Optional[Dict[str,
                pymc.Distribution]]=None := A dict used internally to
                map inferred distributions. Defaults to None.

            - | _model:Optional[pymc.Model] := The :code:`pymc.Model`
                object

        Object Methods:
        ---------------

            - | __init__:= Begin initializing the object by setting all
                options, parameters and hyperparameters
            
            - | __call__(data) := Initialize the model by specifying the
                full probability model according to options passed to
                :code:`__init__`. Accepts a data structure and a label
                indicating the variable that defines the groups

            - | fit(sampler, *args, **kwargs) := Perform inference on
                the model. :code:`sampler` is valid sampler, i.e.
                :code:`pymc.sample` or
                :code:`pymc.sampling.jax.sample_numpyro_nuts`. All other
                arguments are forwarded to the sampler. Returns a
                :code:`arviz.InferenceData` object containing the
                results of inference. Sets the :code:`trained` and
                :code:`idata` attributes

            - | predict(var_names:Iterable[str],
                ropes:Iterable[tuple[float, float]],
                hdis=Iterable[float])->dict[str, pandas.DataFrame] :=
                Compute results of group comparisons and return a
                dictionary mapping derived metrics to
                :code:`pandas.DataFrame` objects containing inference
                summary. Decisions are made using the :code:`ROPE+HDI`
                rule. Returns a dictionary mapping derived variable
                labels to pandas DataFrames containing the results.
                Accepts per variable rope limits and hdis. See the
                function for more details on these and other options

            - | summary := Wrapper for :code:`arviz.summary`. Returns
                summary results of model inference

            - | plot_posterior := Wrapper for
                :code:`arviz.plot_posterior`. Plot the inferred
                posterior

            - | plot_trace := Wrapper for :code:`arviz.plot_trace`. Plot
                inference results
                
            Private Methods:
            ================

            - | _consistency_checks_ := Check that model parameters and
                hyperparameters are consistent and compatible

            - | _preprocessing_ := Preprocess data. Delegates to the
                :code:`data` module and calls the specified processor to
                preprocess the data

            - | _fetch_differential_permutations_ := Compute all unique
                pairs of groups.

        
        Class Methods:
        --------------
        
            Setters for all class attributes. Named set_attribute

            - | set_std_upper(val:float)->None := Update the
                :code:`std_upper` class attribute
                
            - | set_std_lower(val:float)->None :=  Update the
                :code:`std_lower` class attribute
            
            - | set_shape_offset(val:float)->None := Update the
                :code:`ν_offset` class attribute
                
                .. caution::
                    
                    Should generally not be modified
            
            - | set_jax_device(device:str)->None := Update the
                :code:`jax_device` class attribute
            
            - | set_jax_device_count(val:int)->None := Update the
                :code:`jax_device_count` class attribute
            
            - | set_diffusion_factor(val:float)->None := Update the
                :code:`std_diffusion_factor` attribute
            
            - | set_degrees_of_freedom(val:int)->None := Update the
                :code:`ddof` class attribute
        
    '''
    
    nan_handling:str = field(
        init=True, default_factory = lambda : 'exclude')
    cast:Optional[np.dtype] = field(
        init=True, default_factory = lambda : None)
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
    _group_distributions:dict[str, Distribution] = field(init=False, default_factory=dict)
    _idata:Optional[az.InferenceData] = field(
        init=False, default_factory=lambda : None)
    _data_processor:Optional[Data] = field(
        default_factory = lambda : None
        )
    _model:Optional[pymc.Model] = field(init=False, 
                                        default_factory=lambda : None)
    var_names:dict[str, list] = field(init=False, 
                      default_factory = lambda : dict(
                          means=[], stds=[], effect_magnitude=[])
                      )
    _io_handler:Optional[ModelIOHandler] = field(
        init=False, default=None
    )
    _divergence_handler:Optional[ConvergencesHandler] = field(
        init=False, default=None
    )
    _initialized:Optional[bool] = field(default_factory=lambda : False)
    _trained:Optional[bool] = field(default_factory=lambda : False)
    save_path:Optional[str] = field(
        init=True, default = None)
    nan_present_flag:Optional[bool] = field(
        init=False, default_factory=lambda :None)
    
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
        
        
    def __post_init__(self)->None:
        r'''
            Post init actions
            
            Initialize handler objects and validate options
        '''
        self._data_processor = Data(
            nan_handling = self.nan_handling,
            cast = self.cast,
        )
        self._io_handler = ModelIOHandler(
            _save_path = self.save_path
        )
        self._divergence_handler = ConvergencesHandler()
        if not self.common_shape:
            warn((
                "Warning! Permitting independant degress of freedom for "
                "input features can result in overfit. It recommended that "
                "this be avoided, or else model comparison methods be used"
            ))
        if self.multivariate_likelihood:
            warn((
                "Warning! Deploying multivariate likelihood with a "
                "diagonal covariance matrix. This is equivalent to "
                "multiple independant univariate distributions, but more "
                "computationally expensive. It recommended "
                "that :code:`multivariate_likelihood` be set to :code:`False` instead"
            ))
        if self.multivariate_likelihood and not self.common_shape:
            warn((
                "Settings :code:`multivariate_likelihood=True` and "
                ":code:`common_shape=True` are incompatible. Degrees of freedom "
                "parameter for the Multivariate Student T must be a scalar. "
                "The :code:`common_shape` parameter will be ignored"
            ))
            self.common_shape = True
            
    
    
    def _preprocessing_(self, data):
        r'''
            Handled data preprocessing steps by 
            
            1. checking and handling missing values
            3. extracting groups, 
            4. extracting feature labels (coordinates)
            
            Args:
            -----
            
                - data:pandas.DataFrame := The input data
                
                - group_var:str := Label for the categorical
                variable that determines the groups
                
            Returns:
            -------
            
                - None
        '''
        
        def seek_group_indices(struct, lookup_val:str,
                               axis:int=0):
            '''
                Return the indices of cells where elem == lookup_var
                as a numpy vector
            '''
            # This type of lookup is weird. Need to consider a better
            # API to perform value lookups in structures
            this = np.where(
                (struct[:, self.group_var] == lookup_val).values()
                )[1]
            return this

        self.levels = [
            e for e in next(data[:,self.group_var].unique())[1]
            ]
        self.num_levels=len(self.levels)
        self.features = np.asarray([
            k for k in data.coords()[
                list(data.coords().keys())[1]
                                     ] if k != self.group_var
            ])
        self._ndims=len(self.features)
        
        self._groups = {
            level :  seek_group_indices(data, level).tolist() for level in self.levels
        }
        crds = [
            e for e in data.coords()[
                list(data.coords().keys())[-1]
                ] if e != self.group_var
            ]
        self._coords = dict(
            dimensions =  crds
        )
            
        self._fetch_differential_permutations_()
    
    def _fetch_differential_permutations_(self):
        r'''
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
            raise ValueError(('Valid values for :code:`jax.device` are'
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
    
    @classmethod
    def mean(cls, data, axis:int=0):
        return np.mean(data.values(), axis=axis)
    
    @classmethod
    def std(cls, data, ddof:Optional[int] = None, 
            scale:Optional[int] = None, 
            zero_offset:Optional[float]=None,
            axis:int=0,
            ):
        _ddof = ddof if ddof is not None else cls.ddof
        _scale = scale if scale is not None else cls.std_diffusion_factor
        _zero_offset = zero_offset if zero_offset is not None else cls.zero_offset
        # x1000 difference with ints. Explicitly typecast to floats
        stds = np.std(data.values().astype(float) ,ddof = _ddof, axis=axis)
        return _scale*stds
    
    @staticmethod
    def warp_input(data, row_indexer, column_indexer, transform,
                  unwrap:bool=True):
        r'''
            Utility method that selects a subset of the data and applies
            :code:`transform` one it.
            
            Used to supply the data or values extracted form the
            data(mean, standard deviation) to the computation graph
            
            Args:
            -----
                
                - | data:Any := The data to process
                
                - | row_indexer:Any := Indexer for row selection
                
                - | column_indexer:Any := Indexer for column
                    selection
                
                - | transform:Callable:= A callable that takes a data
                    structure and returns a data structure
                
                - | unwrap:bool=True := When True unwraps the resulting
                    :code:`pandas.DataFrame` to the underlying
                    :code:`numpy.NDArray` object. Optional. Defaults to
                    True and returns a numpy array object
                
            Returns:
            --------
            
                - | ndf:bayesian_models.Data := A group of the original
                    Data
                
                - | warped_input:bayesian_models.Data := The output of
                    :code:`transform`
        '''
        if isinstance(row_indexer, np.ndarray):
            row_indexer = row_indexer.tolist()
        if isinstance(column_indexer, np.ndarray):
            column_indexer = column_indexer.tolist()
        selected = data[row_indexer, :][:, column_indexer]
        transformed = transform(selected)
        if unwrap:
            warped_input=transformed.values()
        else:
            warped_input=transformed
        return warped_input
    
    def _check_illegal_std(self, stds:list[np.typing.NDArray])->None:
        r'''
            Checks that the empirical standard deviations are valid
        
            Check against edge case where the empirical pooled std is
            invalid (0). This happens if:
                1. | There is a group in the data, such that at least
                   | one variable column has all the same values, i.e.
                   | x=[100,100,100,100]
                2. A group consists of exactly one observation
                
            Args:
            -----

                - | stds:list[NDArray] := A list of the calculated stds,
                    exactly as they would be passed to the model
        '''
        gather:list[bool] = [
            (std>.0).all() for std in stds
        ]
        invalid_idxs = filter(lambda e: e, gather)
        invalid_levels:list[str] = [
            self.levels[i] for i in invalid_idxs
            ]
        non_zero_sentinel = all(gather)
        if not non_zero_sentinel:
            raise ValueError((
                "Detected groups with 0 variance at levels "
                f"{invalid_levels}. This can happen is the group has"
                "only one observation or all observations have the same"
                " value (for at least one column)"
            ))
    
    
    def __call__(self, data, group_var:Union[str, tuple[str]]):
        r'''
            Initialize the full probability model

            Args:
            -----

                - | data:Any := Input information. The data is assumed
                    to have a single categorical column, defining the
                    variable to group by

                - | group_var:Union[str,tuple[str]] := The variable that
                    defines the groups (a label)

            Returns:
            --------
                
                - obj:BEST := The object
        '''
        self.group_var = group_var
        pdata = self._data_processor(data)
        self.nan_present_flag = pdata.missing_nan_flag()
        self._preprocessing_(pdata)
        core_dists = dict() if not self.common_shape else dict(
            ν = distribution(
                pm.Exponential, "ν_minus_one", BEST.ν_λ, 
                transform = lambda e: e+BEST.ν_offset
                )
        )
        likelihoods:list[LikelihoodComponent] = []
        functions:dict = dict()
        records:dict = dict()
        application_targets = dict()
        response_component = None
        pooled_stds:list = []
        
        for level in self.levels:
            
            pooled_std = BEST.warp_input(
                            pdata, self._groups[level], 
                            self.features,
                            BEST.std,
                            unwrap=False)
            pooled_stds.append(pooled_std)
            core_dists = core_dists| {
                
            f"obs_{level}" : distribution(
                pm.Data, f"y_{level}",
                BEST.warp_input(pdata, self._groups[level],
                    self.features, lambda df:df)
                , mutable=False
                ),
             f"μ_{level}": distribution(
                    pm.Normal, f"μ_{level}", 
                        mu = BEST.warp_input(
                            pdata, self._groups[level], 
                            self.features,
                            BEST.mean,
                            unwrap=False
                            ),   
                        sigma = pooled_std,
                        shape = self._ndims
                    ),
            f"σ_{level}" : distribution(
                pm.Uniform, f'σ_{level}', 
                lower = BEST.std_lower,
                upper = BEST.std_upper, 
                shape=self._ndims)
            }
            if not self.common_shape:
                core_dists = core_dists| {
                    f'ν_{level}' : distribution(
                        pm.Exponential, f"ν_minus_one_{level}",
                        BEST.ν_λ, shape=self._ndims,
                        transform = lambda d: d + BEST.ν_offset
                                            
                )}
            if not self.multivariate_likelihood:
                    like = LikelihoodComponent(
                        distribution = pm.StudentT,
                        observed = f"obs_{level}",
                        name = f"y_obs_{level}",
                        var_mapping = dict(
                            nu = 'ν' if self.common_shape else f'ν_{level}', 
                            mu = f'μ_{level}', 
                            lam = f'σ_star_{level}' 
                            )
                    )
                    fname:str = f'σ_star_{level}'
                    functions = functions|{
                        fname : lambda sigma: sigma**-2
                    }
                    application_targets = application_targets|{
                        fname : f'σ_{level}'
                    }
                    records = records|{
                        fname : False
                    }
                    likelihoods.append(like)
            else:
                fname = f"cov_{level}"
                functions = functions|{
                    fname : lambda sigma: pytensor.tensor.diag(
                        sigma**-2)
                }
                application_targets = application_targets|{
                    fname : f'σ_{level}'
                }
                records = records|{
                    fname : False
                }
                likelihood = LikelihoodComponent(
                    name = f"y_obs_{level}",
                    observed = f"obs_{level}",
                    distribution = pm.MvStudentT,
                    var_mapping = dict(
                        mu = f'μ_{level}',
                        scale = f'cov_{level}',
                        nu = f'ν_{level}' if not self.common_shape \
                            else 'ν'
                    )
                )
                likelihoods.append(likelihood)
            self._check_illegal_std(pooled_stds)
            self._group_distributions[level]=dict(
                mean = core_dists[f'μ_{level}'], 
                std = core_dists[f'σ_{level}'], 
                shape = core_dists[
                    f'ν_{level}'
                    ] if not self.common_shape else core_dists['ν']
                )
        response_component = ResponseFunctionComponent(
            ResponseFunctions(
                functions = functions,
                application_targets = application_targets,
                records = records
            )
        )
        core_component = BESTCoreComponent(
            distributions = core_dists,
            group_distributions = self._group_distributions,
            permutations = self._permutations,
            std_difference = self.std_difference ,
            effect_magnitude = self.effect_magnitude,
        )
        builder = ModelDirector(
            core_component = core_component,
            response_component = response_component,
            likelihood_component = likelihoods,
            coords = self.coords,
        )
        builder()
        self._model = builder.model
        self.var_names = core_component.variables
        self._io_handler._model = self._model
        self._io_handler._class = self
        self.initialized = True
        return self
    

    def fit(self, *args, sampler=pymc.sample , **kwargs)->az.InferenceData:
        r'''
            Perform inference by sampling from the posterior.
            :code:`infer` is an alias for :code:`fit`

            Args:
            -----

                - | sampler=pymc.sample := The :code:`pymc` sampler to
                    run MCMC with. Optional. Defaults to
                    :code:`pymc.sample`

                - | *args:tuple[Any] := Arguements to be forwarded to
                    the  sampler

                - | **kwargs:dict[str, Any] := Optional keyword
                    arguments to be forwarded to the sampler

            Returns:
            --------

                - idata:arviz.InferenceData := The results of inference
        

            Raises:
            -------

                - | RuntimeError := If called before :code:`fit` has
                    been called

            Warns:
            ------

                - If any divergences are detected
        '''
        if not self.initialized:
            raise RuntimeError(("Cannot run inference. Model is not "
                "initialized"))
        with self._model:
            self.idata=sampler(*args, **kwargs)
        self._divergence_handler(self.idata)
        self.trained = True
        return self.idata
    
    infer = fit

    def predict(self, var_names:typing.Sequence[str]=['Δμ'],
            ropes:typing.Sequence[tuple[float, float]]=[(-.1,.1)],
            hdis:typing.Sequence[float]=[.95],
            multilevel_on:str='[',  extend_summary:bool=True):
        r'''
            Render decisions on group differences, according to the
            :code:`ROPE+HDI` rule:
                
                - HDI in ROPE := Not Significant (All plausible values
                  are equivalent to 0)

                - HDI & ROPE == () := Significant (No plausible value is
                  equivalent to zero)

                - | HDI & ROPE != HDI :math::code:`\lor` () := Withhold
                    decision ('Indeterminate') (Some plausible values
                    are equivalent to zero, other are not)

            Aggregates results in a dictionary, mapping deterministic
            variable names to results dataframes. ROPE and HDI threshold limits are set on a dataframe wide level
            
            Example usage:
            
            .. code-block:: python
            
                from sklearn.datasets import load_iris
                from bayesian_models.models import BEST
                import pandas as pd
                
                X, y = load_iris(return_X_y=True, as_frame=True)
                names = load_iris().target_names
                Y = y.replace(
                    {i:name for i, name in enumerate(names)}
                    )
                df = pd.concat([X, Y], axis=1)
                df.columns=df.columns[:-1].tolist()+["species"]
                # Data
                #        sepal length (cm)   ...   species
                #    0                  5.1  ...   setosa
                #    1                  4.9  ...   setosa
                #    2                  4.7  ...   setosa
                #    3                  4.6  ...   setosa
                #    4                  5.0  ...   setosa
                #    ...                ...  ...     ...
                #    145                6.7  ...   virginica
                #    146                6.3  ...   virginica
                #    147                6.5  ...   virginica
                #    148                6.2  ...   virginica
                #    149                5.9  ...   virginica
                
                obj = BEST()(df, "species")
                obj.fit()
                results = obj.predict() # Only 'Δμ' by default
                print(results['Δμ'])
                # Output
                #                                            Significance
                # Δμ(setosa, versicolor)  sepal length (cm) Indeterminate
                #                        sepal width (cm)   Indeterminate
                #                        petal length (cm)  Indeterminate
                #                        petal width (cm)   Indeterminate
                # Δμ(setosa, virginica)  sepal length (cm)  Indeterminate
                #                        sepal width (cm)   Indeterminate
                #                        petal length (cm)  Indeterminate
                #                        petal width (cm)   Indeterminate
                # Δμ(versicolor, sepal length (cm)          Indeterminate
                #                       sepal width (cm)    Indeterminate
                #                        petal length (cm)  Indeterminate
                #                        petal width (cm)   Indeterminate

            Args:
            -----

                - | var_names:Iterable[str]=['Δμ'] := An iterable of
                    derived metrics to return. Optional. Defaults to
                    returning expected difference. Default names are
                    'Δμ' for the difference of means, 'Δσ' for the
                    difference of standard deviations and 'Effect_Size'
                    for Kruschkes effect size.

                - | ropes:Iterable[tuple[float, float]] := An Iterable
                    of length-2 tuples of floats, defining the Region Of
                    Practical Equivalence. Each rope is applied to all
                    features for every variable in :code:`var_names`.

                - | hdis:Iterable[float] := An iterable of non-negative
                    floats defining the probability threshold for the
                    credible interval. Is applied to all features for
                    each variable in :code:`var_names`

                - | multilevel_on:str='[' := A separator defining the
                    multilevel index. :code:`pymc` by default
                    concatinates the label according to the form:
                    {var_name}[{feature_label}]. The argument will
                    reindex them in a multilevel fashion of the form
                    (var_name, feature_label) in the resulting
                    dataframe. Set to None to disable this behavior.

                - | extend_summary:bool=True := If True the new
                    Significance column extends the summary dataframe.
                    Else return a new  dataframe containing only the
                    Significance results. Optional. Defaults to True and
                    returns an extended version of the summary.
                    
            Returns:
            --------
            
                - | results:dict[str,pandas.DataFrame] := Decision
                    results. Returned in the form of dictionary, whose
                    keys are the names of deterministic quantities (i.e.
                    'Δμ' 'Δσ' 'E' 'ν'). Items are
                    :code:`pandas.DataFrame` object containing summary
                    results (like those returned by
                    :code:`arviz.summary`) with an additional column,
                    named :code:`Significance` containing the decision
        '''
        if not self.trained:
            raise RuntimeError(("Cannot make predictions. Model has "
                "not been trained. Call the objects :code:`fit` method first")
                )
        if 'Δσ' in var_names and not self.std_difference:
            raise RuntimeError(("var_names contains the variable 'Δσ'"
                                " but :code:`std_difference` was not set to"
                                " True. Ensure you specify "
                                ":code:`std_differnce=True` when initializing"
                                " if you need to return standard "
                                "deviations" ))
        if 'Effect_Size' in var_names and not self.effect_magnitude:
            raise RuntimeError(("var_names contains the variable 'Δσ'"
                                " but :code:`effect_magnitude` was not set to"
                                " True. Ensure you specify "
                                ":code:`effect_magnitude=True` when "
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
        null_interval = interval(0,0)
        for var_name, rope,hdi in zip(var_names,ropes, hdis):
            raw_summary = az.summary(self.idata, var_names=[var_name],
            filter_vars='like', hdi_prob=hdi)
            rope=interval(*rope)
            out=[]
            for idx,row in raw_summary.iterrows():
                ci=interval(row[2], row[3])
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
        r'''
            Wrapper for :code:`arviz.summary(idata)`

            Args:
            -----

                - | *args:tuple[Any] := Arguments to be forwarded to
                    :code:`arviz.summary`

                - | **kwargs:dict[str, Any] := Keyword arguments to be
                    forwarded to :code:`arviz.summary`
            
            Returns:
            ---------

                - summary_results:xarray.DataArray := Summary results
        '''
        if not self.trained:
            raise RuntimeError(("Model has not been trained. Ensure "
                "you've called the objects :code:`fit` method first"))
        return az.summary(self.idata, *args, **kwargs)


    def plot_posterior(self, *args, **kwargs):
        r'''
            Wrapper for :code:`arviz.plot_posterior`
            
            Plot Posterior densities in the style of John K. Kruschke's
            book.
            
        Args:
        -----

            - | *args:tuple[Any] := Arguments to be forwarded to
                :code:`arviz.plot_posterior`

            - | **kwargs:dict[str, Any] := Keyword arguments to be
                forwarded to:code:`arviz.plot_posterior`
                
        Returns:
        --------
        
            - graph:Any := The posterior plot object
        '''
        if not self.trained:
            raise RuntimeError(
                "Cannot plot posterior. Model is untrained")
        return az.plot_posterior(self.idata, *args, **kwargs)


    def plot_trace(self, *args, **kwargs):
        r'''
            Wrapper for :code:`arviz.plot_trace`
        
        Plot distribution (histogram or kernel density estimates) and
        sampled values or rank plot. If divergences data is available in
        sample_stats, will plot the location of divergences as dashed
        vertical lines.
            
        Args:
        -----

            - | *args:tuple[Any] := Arguments to be forwarded to
                :code:`arviz.plot_trace`

            - | **kwargs:dict[str, Any] := Keyword arguments to be
                forwarded to :code:`arviz.plot_trace`
                
        Returns:
        --------
        
            - plot:Any := Trace plot object
        '''
        if not self.trained:
            raise RuntimeError("Cannot plot trace. Model is untrained")
        
        return az.plot_trace(self.idata, *args, **kwargs)
    
    def save(self, save_path:Optional[str]=None, 
             method:str='netcdf')->None:
        r'''
            Save the model object for later reuse
            
            Available save methods are 'netcdf' and 'pickle'. The latter
            is discouraged and has known problems. For the 'netcdf'
            method the posterior trace will be saved.
            
            .. caution::
            
                At present, no checks are being made to verify that the posterior is compatible with the model object
                
            Args:
            -----
            
                - | save_path:Optional[str] := The file path to save the
                    model to. If :code:`save_path` has been provided at
                    model initialization, need not be provided
                    
                - | method:str='netcdf' := Which method to use to save
                    the model. For the 'netcdf' method, only the
                    posterior trance is save and consequently reloaded.
                    For the 'pickle' method, attempts to serialize the
                    entire model. The latter case should be avoided
                    
            Returns:
            --------
            
                - None
        '''
        spath = save_path if save_path is not None else self.save_path
        self._io_handler.save(spath, method=method)
    
    def load(self, save_path:Optional[str]=None)->None:
        r'''
            Load a pre trained model from the disk
            
            Only meaningful for models saved with the 'netcdf' method.
            Otherwise, use 'pickle' directly instead
            
            .. caution::
            
                Not checks are being made that the posterior trace is compatible with model object
            
            Args:
            -----
            
                - | save_path:Optional[str] := File path to load the
                    model from. If :code:`save_path` has been provided
                    at object initialization, it can be ignored
                    
            Returns:
            --------
            
                - None
        '''
        self._io_handler.load(save_path)
