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
