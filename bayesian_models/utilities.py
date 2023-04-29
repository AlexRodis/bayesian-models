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
#   This module contains miscellaneous helper functions used elsewhere

import pandas as pd
from collections.abc import Iterable
from typing import Callable, Any, Iterable
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from collections.abc import Iterable
import typing
from typing import Callable, Any, Type, Iterable, Sequence
import numpy as np
import xarray as xr
import functools
from pymc.distributions import Distribution

# Standard Scaling with label support
std_scale = lambda df: pd.DataFrame(
    data = StandardScaler().fit_transform(df), 
    columns = df.columns, index=df.index)


def rowwise_value_counts(df:pd.DataFrame):
    '''
        Returns row-wise counts of values for
        categorical variables.

        Args:
        ------

            - df:pandas.DataFrame := The dataframe to process
        
        Returns:
        ---------

            - counts:pandas.DataFrame := A new DataFrame of counts of
            distinct values in the input dataframe
    '''
    vcounts_df = pd.DataFrame(data = df.apply(
        lambda x: x.value_counts()).T.stack()).astype(int).T
    vcounts_df.index = ['']
    return vcounts_df


invert_dict = lambda e: {v:k for k, v in e.items()}

def flatten(obj:Iterable):
    '''
        Recursively flatten arbitary an arbitary input iterable, yielding
        values iteratively.

        Args:
        -----

            - obj:Iterable := The nested iterable to flatten

        Returns:
        --------

            - 
    '''
    PRIMITIVES=(str, bytes)
    for e in obj:
        if isinstance(e, Iterable) and not isinstance(e, PRIMITIVES):
            yield from flatten(e)
        else:
            yield e


def tidy_multiindex(df:pd.DataFrame, sep:str="."):
    '''
        Compress a hierarchically indexed dataframe to standardized tidy
        format. A unique sepperator `sep` is used to allow reversal. All
        levels of the index are appended together with a delimeter to allow
        reversals.
        
        Args:
        ----
        
            - df:pandas.DataFrame := A `pandas.DataFrame` hierarchically 
            indexed
            
            - sep:str='_._' := A delimenter delineating the different levels
            of the index. Ensure it is not present in any column name to avoid
            a malformed index
            
        Returns:
        --------
        
            - ndf:pandas.DataFrame := The DataFrame with a single-level index
    '''
    tidy_cols = df.columns
    tidy_rows = df.index
    import functools
    if isinstance(df.columns, pd.MultiIndex):
        tidy_cols = (functools.reduce(
            lambda e1,e2: str(e1)+sep+str(e2), col ) for col in df.columns)
    ndf = df.copy(deep=True)
    ndf.columns = tidy_cols
    if isinstance(df.index, pd.MultiIndex):
        tidy_rows = (functools.reduce(lambda e1,e2: str(e1)+sep+str(e2), col ) for col in df.index)
    ndf = ndf.copy(deep=True)
    ndf.index = tidy_rows
    return ndf


def reverse_tidy_multiindex(df:pd.DataFrame, sep="."):
    '''
        Reverses the tidying to a hierachical format. Different
        levels of the index are identified based on "sep"
        
        Args:
        -----
        
            - df:pandas.DataFrame := The dataframe to process
            
            - sep:str='_._' := The string delimeter, sepperating
            values for different levels of the index
            
        Returns:
        -------
        
            - ndf:pandas.DataFrame := The dataframe with hierarchical index
    '''
    h_cols = (tuple(col.split(sep)) for col in df.columns)
    ndf = df.copy(deep=True)
    ndf.columns = pd.MultiIndex.from_tuples(h_cols)
    return ndf

def undummify(df:pd.DataFrame,cols:list[str, tuple[str]],
    ncol_name:typing.Union[str,tuple[str]],
    sep:typing.Optional[str]=None,
    rmap:typing.Optional[dict[int, typing.Union[str, tuple[str]]]]=None
             )->pd.DataFrame:
    '''
        Reverses hot-encoded variables in the DataFrame. A series of 
        hot-encoded variable levels $(i_1, i2, \dots, i_k)$ is mapped to a 
        single new column $(k)$, whose name is specified by `ncol_name`, in 
        the new dataframe. Previous level columns are dropped.
        
        Args:
        ----
        
            - df:pandas.DataFrame := The DataFrame to operate upon
            
            - cols:list[str, tuple[str]] := A list of columns, representing 
            the levels of a categorical variable
            
            - sep:Optional[str] := sepperator for variable level. Currently 
            ignored
            
            - ncol_name:Union[str, tuple[str]] := Name of the new categorical 
            column
            
            - remap:Optional[dict[int, Union[str, tuple[str]]]] := A 
            dictionary mapping of categorical levels to values. Keys are the 
            assumed to be levels, values are assumed to be values 
            (i.e. strings). When provided, the previous levels will be 
            replaced by the specified mappings in the new DataFrame
            
        Returns:
        -------
        
            - ndf:pandas.DataFrame := The processed dataframe
     '''
    _df = df.loc[:, cols]
    for i, col in enumerate(cols, 1):
        _df.loc[:, col] = i*_df.loc[:, col]
    ndf = df.copy(deep=True)
    ndf.drop(cols, axis=1, inplace=True)
    ndf[ncol_name] = _df.max(axis=1)
    c1 = df.columns.tolist()
    i = c1.index(cols[0])
    swp = ndf.columns.tolist()[:i-1]+[ndf.columns.tolist()[-1]]+\
        ndf.columns.tolist()[i:-1]
    ndf = ndf.loc[:, swp]
    if rmap is not None:
        ndf = ndf.replace(rmap)
    return ndf

list_difference = lambda l1, l2: [e for e  in l1 if e not in set(l2)]


class SklearnDataFrameScaler:
    '''
        Simple wrapper for `sklearn.preprocessing` scalers.
        For labeled inputs, these return numpy arrays. This
        wrapper adds the labels back to the result
        
        Args:
        -----
        
            - scaler:Callable[[...], tuple[numpy.ndarray]] := 
            The scaler Callable. Must use the class based API
            
            - backend:str='pandas' := Which label matrix backend
            to use. Valid options are 'pandas' and 'xarray'
            
        Returns:
        --------
            -scaler_arrays:tuple[pd.DataFrame, xarray.DataArray] := 
            A tuple of rescaled and relabeled arrays
    '''
    
    def __init__(self, scaler:Callable[..., Any], backend:str="pandas",
        *args, **kwargs):
        
        self.scaler = scaler(*args, **kwargs)
        self.backend:str = backend
        
    def __call__(self, *arrs, **kwargs)->tuple[pd.DataFrame]:
        outputs = []
        return tuple([pd.DataFrame(
            data = self.scaler.fit_transform(arr,**kwargs),
            index = arr.index,
            columns = arr.columns) for arr in arrs ])

class DictTable(dict):
    '''
        Jupyter utility class that overrides the dicts' defaults
        __repr__ rendering the input dictionary to an HTML table
        for convenient jupyter rendering
    '''
    def _repr_html_(self):
        html = ["<table>"]
        for key, value in self.items():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def count_missing_nan(df:pd.DataFrame, axis:int=0):
    '''
        Return a new DataFrame with the counts of missing
        and invalid values across specified axis.
        
        Args:
        -----
        
            - df:pandas.DataFrame := The data
            
            - axis:int=0 := The axis across which missing
            values will be enumerated. Defaults to 0 (show
            missing values in each column)
            
        Returns:
        -------
        
            - missing_df:pd.DataFrame := A DataFrame containing
            counts of missing values
    '''
    ndf = pd.DataFrame(df.isna().sum(axis=axis)).T
    ndf.index = ['']
    return ndf

# Multiple query utility
query_multiple = lambda df, col, multiquery : functools.reduce(
    lambda p,n: "{col} == '{p}' | ".format(p=p, col=col
    ) + "{col} == '{n}".format(n=n, col=col), multiquery).split(
        " == '",1)[1]+"'"

def select_subarray(df:pd.DataFrame, 
    targets:list[tuple[str,str,str]], 
    indicators:list[tuple[str,str,str]], scaleX:bool=True, 
    train_split:bool=False,
    mappings:typing.Optional[dict[str, typing.Union[str,int, float]]]=None,
    dummify:bool=False, )->tuple[pd.DataFrame, pd.DataFrame]:
    import sklearn
    X = df.loc[:,indicators]
    X = std_scale(X) if scaleX else X
    Y = df.loc[:, targets]
    m = Y.loc[~Y.isna().any(axis=1),:].index.intersection(X.loc[~X.isna().any(axis=1),:].index)
    X = X.loc[m, :]
    Y = Y.loc[m,:]
    Y=Y.replace(mappings) if mappings is not None else Y
    Y = pd.get_dummies(Y) if dummify else Y
    if train_split:
        X_train, X_test, Y_train, Y_test = \
            sklearn.model_selection.train_test_split(X, 
            Y,test_size=.1 ,random_state=44)
        return X_train, X_test, Y_train, Y_test
    else:
        return X, Y 

def dataarray_from_pandas(df:pd.DataFrame,dim_names=["dim_0", "dim_1"],
    **kwargs ):
    '''
        Convert a `pandas.DataFrame` to an equivalent `xarray.DataArray`
        
        Args:
        -----
        
            - df:pandas.DataFrame := The `pandas.DataFrame` to convert
            
            - dim_names:Sequence[Hashable] := A sequence of `Hashable`
            names to be used for the two axis. Optional. Defaults to
            the names 'dim_0' and 'dim_1'.
            
            - **kwargs:dict[Any,Any] := Keyword arguements to be 
            forwarded to `xarray.DataArray` constructor. Optional
            
        Returns:
        --------
        
            - converted:xarray.DataArray := The converted dataframe
    '''
    dims={k:v for k,v in zip(dim_names,(df.index,df.columns))}
    return xr.DataArray(df.values, coords=dims, **kwargs)

def package_dirichlet_predictions(raw_preds,outputs, 
    inputs=None,model=None)->xr.DataArray:
    '''
        Converts the raw numpy tensor output of an A.N.N
        to a human-readable `xarray.DataArray`, with optional
        post-processing
        
        Args:
        -----
        
            - raw_preds:numpy.ndarray := The output of the
            `tf.model.predict`
            
            - outputs:xarray.DataArray := The test set of
            the N.N., from which names and labels will be
            infered.
            
            - inputs:Optional[xarray.DataArray] := The inputs
            to the model. Should only be prodived if the
            inputs were masked, and this should be the
            unmasked tensor, from which appropriate reshape
            will be infered. First two dimentions are
            assumed to be of `permutation x sample`. 
            Optional. Defaults to None (ignored and no
            reshaping will be attempted).
            
            model:Optional[tf.keras.model]=None := The
            model, whose metadata will be extracted and
            added to the resulting DataArray's attributes.
            Optional. Currently not implemented.
    '''
    import datetime
    # Add more model characteristics
    attributes=dict(
        created_on=datetime.datetime.now(),
        masked_inputs=inputs is not None,
        )
    output_coords={dim:outputs.coords[dim] for dim in outputs.dims[1:]}
    output_coords[list(output_coords.keys())[0]]=np.concatenate(
        [np.asarray(['Unmasked']), output_coords[list(
            output_coords.keys())[0]]],axis=0)
    if inputs is not None:
        input_coords={dim:inputs.coords[dim] for dim in inputs.dims[:2]}
        coords=input_coords|output_coords
        processed_data=np.roll(np.concatenate(
        [raw_preds.reshape(list(inputs.shape[:2])+list(raw_preds.shape[1:])),
        inputs[:,:,[0]].values],axis=2),1, axis=2)       
    else:
        coords={outputs.dims[0]:outputs.coords[outputs.dims[0]]
        }|output_coords
        processed_data=raw_preds
    predictions = xr.DataArray(processed_data, coords=coords,
                              attrs=attributes)
    return predictions

def dirichlet_moments(a:np.ndarray,standardize:bool=True):
    '''
        Calculate the Dirichlet distributions'
        critical moments. A dirichlet is fully
        specified by it's first two moments,
        i.e. mean and variance
        
        Args:
        -----
            - shape:Sequence:= The shape parameter(α) of the
            Dirichlet. First dimention is assumed to by
            sepperate Dirichlets and the last dimention
            defines the shape
            
            - standardize:bool=True := If true standardizes
            the variance, returning standard deviation. Else
            returns plain second order central moment i.e. the
            variance. Optional. Defalts to True.
        
        Returns:
        -------
        
            - dirich_moments:xarray.Dataset := An `xarray.Dataset`
            containing the calculcate moments. Always has the
            'variance' and 'mean' data variables, and if
            `standardize=True` also contains the 'std_dev'
            data variable
            
        Raises:
        -------
            
            - ValueError:= If the input a cannot be coerced into
            an array-like structure, via a.values
    '''
    α=a
    α0=α.sum(axis=-1,keepdims=True)
    μ=(α/α0)
    var=α*(α0-α)/(α0**2*(α0+1))
    if standardize:
        out=np.stack([μ,var,np.sqrt(var)] )
    else:
        out=np.stack([μ,var],axis=len(μ.shape)+1 )
    return out


def mean_squared_error(true:np.typing.NDArray, 
    predicted:np.typing.NDArray, sample_weights=None, squared:bool=True, 
    mean:bool=True, average:bool=True, sample_dim:int=0):
    '''
        Squared Error implementation, based  on 
        `sklearn.metrics.mean_squared_error` with more options.

        Args:
        -----

            - true:numpy.typing.NDArray := Array of true values

            - predicted:numpy.typing.NDArray := Array of predicted values

            - mean:bool=True := Selects whether to calculate the mean of
            error across all samples or return indevidual errors. Optional
            Defaults to True (and return Mean Squared Error)

            - average:bool=True := Selects how to handle multidimentional
            inputs. If True (default) averages over the outputs axis. With
            The averaging behavior is determined by `sample_weights`. If `None`
            then performs uniform averaging, otherwise returns a weighted
            average. If `False` averaging is done, and errors are returned
            for each output. Optional. Defaults to `True` and returns uniform
            average `MSE`.

            - sample_weights:Optional[np.typing.NDArray] := An array of 
            weights to be used during averaging over the outputs dimention.
            Must be of appropriate length and is ignored is `average=False`.
            If `None` the average is uniform. Optional. Defaults to `None` and
            returns uniform averaging.

            - sample_dim:int=0 := Specifies the sample dimention. Optional.
            Defaults to 0.

        Returns:
        --------

            - errors:numpy.ndarray := An array of errors. Either mse, if
            `mean=True` and `squared=True`, rmse if `squared=False`, 
            squared error if `mean=False` and `squared=True` or error
            if `mean=False` and `squared=False`:

                - `mean=True`and `average=True` := Scalar mse

                - `mean=True` and `average=False` := Output-length vector
                of errors for each output

                - `mean=False` and `average=True` := observations-length
                vector of averaged error for each test sample

                - `mean=False` and `average=False` := 2D array, of the same
                shape as the inputs

    '''
    averaged = None
    meaned = None
    se = (true-predicted)**2
    if average:
        averaged = np.average(se, axis=-1, weights=sample_weights)
    averaged = None
    meaned = None
    se = (true-predicted)**2
    if average:
        averaged = np.average(se, axis=-1, weights=sample_weights)
    else:
        averaged=se
    if mean:
        meaned = np.average(averaged, weights=None)
        averaged=se
    if mean:
        meaned = np.average(averaged, weights=None,axis=sample_dim)
    else:
        meaned = averaged
    return meaned if squared else np.sqrt(meaned)

    
def gen_masked_predictions(model, masked_generator,
                           baseline_truth,
                           metrics:list[
                            typing.Callable[..., typing.Any]
                            ]=[functools.partial(mean_squared_error,
                            squared=False)],
                            return_elements:bool=False, **kwargs):
    '''
        Generate performance metric evaluations on masked
        inputs.
        
        Args:
        -----
            - model:= The model to generate predictions from
        
            - masked_generator:= Masked inputs generator. Firstmost
            element must be ground truth, i.e. fully unmasked inputs
            
            - baseline_truth:np.ndarray := Array of values to be used
            as ground truths. Will be expanded during evaluation
            process
            
            - metrics:list[Callable[...,Any]] := Metrics to be used in
            the assessment of the A.N.N. predictions. Currently only a
            single metric is supported. The metric should be a Callable
            accepting a `baseline truth` tensor, a `predictions` tensor and
            a sample_dim arguement. Currently only single rank inputs are
            supported and the `sample_dim` determines the features axis.

           - return_inputs:bool=True := Selects whether to return detected 
            inputs or just the absolute indices of the masks.
            
            - kwargs:dict[str,Any]:= Keyword arguments to be forwarded
            to, among other places `model.predict`


        Returns:
        ---------

            - Measures:collections.namedtuple := A namedtupe containing
            the metric evaluation for the masks and optionally the 
            corresponding inputs. Has three fields:

                - iteration:int>=1 := Enumerates the current iteration

                - measures:numpy.ndarray := Array of `metric` evaluations
                for the masks. For code consistency, is a rank-4 tensor
                of the form `distribution_moment x permutation x sample x features`
                of the general form `1 x batch_size x 1 x 1`.

                - inputs:Optional[xarray.DataArray]=None := If 
                `return_elements=True` if an `xarray.DataArray` containing
                the masked inputs, otherwise is None. The array is a rank-3 
                tensor of `permutations x samples x features`. Defaults to
                non-empty field.
    '''
    import numpy as np
    from collections import namedtuple
    Measures = namedtuple('Measures', ["iteration", "measures",
        "inputs"])
    i=0
    while True:
        masked=next(masked_generator)
        preds= model.predict(masked.stack(
            dict(train_samples=["permutation", 'samples'])).T.values,
                                     verbose=kwargs['verbose'])
        unstacked_shape=list(masked.shape[:2])+list(preds.shape[1:])
        stacked_shape=preds.shape
        dirichlet=dirichlet_moments(preds.reshape(*unstacked_shape))
        expectations_only=dirichlet[0,...]
        expectations_only[expectations_only<.05]=.0
        dirichlet_reset_baseline=expectations_only/expectations_only.sum(
            axis=-1,keepdims=True)
        i+=1
        measures=[]
        for metric in metrics:
            measures.append(
                metric(baseline_truth, dirichlet_reset_baseline,
                sample_dim=1))
        measure_array = np.stack(measures) if len(metrics)>1 else \
            measures[0][None,...]
        yield Measures(
            iteration=i, measures=measure_array, inputs=masked)
        
        
def extract_dist_shape(dist:Type[Distribution])->list[str]:
    from inspect import signature
    '''
        Extracts the names of a distributions' shape parameters,
        returning them as strings. For example:
        .. code-block::
            extract_dist_shape(pymc.StudentT)
            ['mu', 'sigma', 'nu']
    '''
    return [e for e in signature(dist.logp) if e != 'value']


def powerset(sequence:Sequence)->Iterable:
    '''
        Powerset implementation in pure python. Returns all possible
        'subsets' of input sequence, including the empty set. Evaluation
        is lazy, and each element returned is a tuple of the elements
        of the original iterable. Example:
        
        .. code-block::
            # The input here are the keys
            ittr = dict(one=1, two=2, three=3, four=4)
            In [7]: list(powerset(ittr))
            Out[7]: 
            [(),
            ('one',),
            ('two',),
            ('three',),
            ('four',),
            ('one', 'two'),
            ('one', 'three'),
            ('one', 'four'),
            ('two', 'three'),
            ('two', 'four'),
            ('three', 'four'),
            ('one', 'two', 'three'),
            ('one', 'two', 'four'),
            ('one', 'three', 'four'),
            ('two', 'three', 'four'),
            ('one', 'two', 'three', 'four')]

            # Another example, with the iterable being tuples of key/value
            # pairs
            In [8]: list(powerset(ittr.items()))
            Out[8]: 
            [(),
            (('one', 1),),
            (('two', 2),),
            (('three', 3),),
            (('four', 4),),
            (('one', 1), ('two', 2)),
            (('one', 1), ('three', 3)),
            (('one', 1), ('four', 4)),
            (('two', 2), ('three', 3)),
            (('two', 2), ('four', 4)),
            (('three', 3), ('four', 4)),
            (('one', 1), ('two', 2), ('three', 3)),
            (('one', 1), ('two', 2), ('four', 4)),
            (('one', 1), ('three', 3), ('four', 4)),
            (('two', 2), ('three', 3), ('four', 4)),
            (('one', 1), ('two', 2), ('three', 3), ('four', 4))]

    
    '''
    from itertools import chain, combinations
    return chain.from_iterable(
        combinations(sequence, r) for r in range(len(sequence)+1)
        )

def dict_powerset(dictionary:dict, keys_only:bool=False)->Iterable:
    '''
        Dictionary powerset function. Lazily returns all possible 
        'sub-dictionaries' from an input dict - including an emtpy
        dict. Returns entire dicts if `keys_only=True` or tuples of
        keys otherwise. Examples:
        
        .. code-block::
            In [1]: ittr = dict(one=1, two=2, three=3, four=4)
            In [2]: list(dict_powerset(ittr))
            Out[2]: 
            [{},
            {'one': 1},
            {'two': 2},
            {'three': 3},
            {'four': 4},
            {'one': 1, 'two': 2},
            {'one': 1, 'three': 3},
            {'one': 1, 'four': 4},
            {'two': 2, 'three': 3},
            {'two': 2, 'four': 4},
            {'three': 3, 'four': 4},
            {'one': 1, 'two': 2, 'three': 3},
            {'one': 1, 'two': 2, 'four': 4},
            {'one': 1, 'three': 3, 'four': 4},
            {'two': 2, 'three': 3, 'four': 4},
            {'one': 1, 'two': 2, 'three': 3, 'four': 4}]
            
            
            In [3]: list(dict_powerset(ittr, keys_only=True))
            Out[3]: 
            [(),
            ('one',),
            ('two',),
            ('three',),
            ('four',),
            ('one', 'two'),
            ('one', 'three'),
            ('one', 'four'),
            ('two', 'three'),
            ('two', 'four'),
            ('three', 'four'),
            ('one', 'two', 'three'),
            ('one', 'two', 'four'),
            ('one', 'three', 'four'),
            ('two', 'three', 'four'),
            ('one', 'two', 'three', 'four')]

    '''
    expr = dictionary.keys if keys_only else dictionary.items
    if keys_only:
        return powerset(expr())
    else:
        return map(dict, powerset(expr()))

invert_dict = lambda e: {v:k for k, v in e.items()}

def flatten(obj:Iterable):
    r'''
        Flatten a nested iterable
    
        Recursively flatten arbitrary an arbitrary input iterable,
        yielding values iteratively.

        Args:
        -----

            - obj:Iterable := The nested iterable to flatten

        Yields:
        -------

            - | element:Any := Each non-iterable element in the array
    '''
    PRIMITIVES=(str, bytes)
    for e in obj:
        if isinstance(e, Iterable) and not isinstance(e, PRIMITIVES):
            yield from flatten(e)
        else:
            yield e


def tidy_multiindex(df:pd.DataFrame, sep:str="."):
    r'''
        Convert a hierarchically indexed :code:`pandas.DataFrame` to
        tidy formated one
    
        Compress a hierarchically indexed dataframe to standardized tidy
        format. A unique sepperator `sep` is used to allow reversal. All
        levels of the index are appended together with a delimeter to
        allow reversals.
        
        Args:
        ----
        
            - | df:pandas.DataFrame := A `pandas.DataFrame`
                hierarchically indexed
            
            - | sep:str='_._' := A delimenter delineating the different
                levels of the index. Ensure it is not present in any
                column name to avoid a malformed index
            
        Returns:
        --------
        
            - | ndf:pandas.DataFrame := The DataFrame with a
                single-level index
    '''
    tidy_cols = df.columns
    tidy_rows = df.index
    import functools
    if isinstance(df.columns, pd.MultiIndex):
        tidy_cols = (functools.reduce(
            lambda e1,e2: str(e1)+sep+str(e2), col ) for col in df.columns)
    ndf = df.copy(deep=True)
    ndf.columns = tidy_cols
    if isinstance(df.index, pd.MultiIndex):
        tidy_rows = (functools.reduce(lambda e1,e2: str(e1)+sep+str(e2), col ) for col in df.index)
    ndf = ndf.copy(deep=True)
    ndf.index = tidy_rows
    return ndf


def reverse_tidy_multiindex(df:pd.DataFrame, sep="."):
    r'''
        Convert a tidy dataframe to hierarchically indexed one based on
        separator delimiters
    
        Reverses the tidying to a hierarchical format. Different
        levels of the index are identified based on "sep"
        
        Args:
        -----
        
            - df:pandas.DataFrame := The dataframe to process
            
            - | sep:str='_._' := The string delimiter, separating
                values for different levels of the index
            
        Returns:
        -------
        
            - ndf:pandas.DataFrame := The dataframe with hierarchical index
    '''
    h_cols = (tuple(col.split(sep)) for col in df.columns)
    ndf = df.copy(deep=True)
    ndf.columns = pd.MultiIndex.from_tuples(h_cols)
    return ndf


class SklearnDataFrameScaler:
    r'''
        Extend the functionality of sklearn scalers, allowing for
        labeled inputs and outputs
        
        Adds labels to the result of sklearn scalers
        
        Args:
        -----
        
            - | scaler:Callable[[...], tuple[numpy.ndarray]] := The
                scaler Callable. Must use the class based API
            
            - | backend:str='pandas' := Which label matrix backend to
                use. Valid options are 'pandas' and 'xarray'
            
        Returns:
        --------
        
            - | scaler_arrays:tuple[pd.DataFrame, xarray.DataArray] := A
                tuple of rescaled and relabeled arrays
    '''
    
    def __init__(self, scaler:Callable[..., Any], backend:str="pandas",
        *args, **kwargs):
        
        self.scaler = scaler(*args, **kwargs)
        self.backend:str = backend
        
    def __call__(self, *arrs, **kwargs)->tuple[pd.DataFrame]:
        outputs = []
        return tuple([pd.DataFrame(
            data = self.scaler.fit_transform(arr,**kwargs),
            index = arr.index,
            columns = arr.columns) for arr in arrs ])
