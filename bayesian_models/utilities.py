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

def merge_dicts(*args:tuple[dict])->dict:
    r'''
        Merge multiple dictionaries into one
        
        Backwards compatible version of dict merging, alternative to the
        3.10 operator
        
        .. code-block:: python
            d1:dict = dict(foo=1)
            d2:dict = dict(bar=2)
            d3:dict = dict(baz='hi')
            
            print(merge_dicts(d1, d2, d3))
            # Output
            # {'foo': 1, 'bar': 2, 'baz': 'hi'}
            
            print(merge_dicts(d1, d2, d3) == d1|d2|d3)
            # Output
            # True
    '''
    from itertools import chain
    return {k:v for k, v in chain.from_iterable(
        map(lambda e: e.items(), args)
    ) }
    
def get_wnulldict(dictionary:dict[str,Any], 
                    lookupstr:str)->dict:
    r'''
        Drop in replacement for :code:`dictionary.get(...)` that returns
        an empty dictionary instead of :code:`None` on a non existent
        key lookup
    '''
    fetch = dictionary.get(lookupstr)
    return fetch if fetch is not None else dict()

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

# This module contains redundant and irrelevant definitions, which
# will be removed in the future
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
    r'''
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

def undummify(df:pd.DataFrame,cols:list[str, tuple[str]],
    ncol_name:typing.Union[str,tuple[str]],
    sep:typing.Optional[str]=None,
    rmap:typing.Optional[dict[int, typing.Union[str, tuple[str]]]]=None
             )->pd.DataFrame:
    r'''
        Reverses hot-encoded variables in the DataFrame. 
        
        A series of hot-encoded variable levels :math:`(i_1, i2, \dots,
        i_k)` is mapped to a  single new column :math:`(k)`, whose name
        is specified by :code:`ncol_name`, in the new dataframe.
        Previous level columns are dropped.
        
        Args:
        ----
        
            - df:pandas.DataFrame := The DataFrame to operate upon
            
            - | cols:list[str, tuple[str]] := A list of columns,
                representing  the levels of a categorical variable
            
            - | sep:Optional[str] := Separator for variable level.
                Currently ignored
            
            - | ncol_name:Union[str, tuple[str]] := Name of the new
                categorical column
            
            - | remap:Optional[dict[int, Union[str, tuple[str]]]] := A
                dictionary mapping of categorical levels to values. Keys
                are the  assumed to be levels, values are assumed to be
                values (i.e. strings). When provided, the previous
                levels will be  replaced by the specified mappings in
                the new DataFrame
            
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
        
def extract_dist_shape(dist:Type[Distribution])->list[str]:
    from inspect import signature
    r'''
        Extracts the names of a distributions' shape parameters,
        returning them as strings. 
        
        Example:
        
        .. code-block:: python
        
            extract_dist_shape(pymc.StudentT)
            ['mu', 'sigma', 'nu']
            
        Args:
        -----
        
            - | dist:Type[pymc.Distribution] := A
                :code:`pymc.Distribution` object
                
        Returns:
        --------
        
            - | shape:list[str] := Symbols for the distributions' shape
                parameters
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