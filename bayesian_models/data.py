from __future__ import annotations
import xarray as xr
import  pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Hashable, Iterable, Type
from .typing import ndarray, InputData, SHAPE, DIMS, COORDS, \
    AXIS_PERMUTATION
from dataclasses import dataclass





# Data Types Bridge
class DataStructure(ABC):
    '''
        Abstract Base Class for Data Structure implementations
        
        Properties:
        ------------
            Common properties exposed by the underlying object
        
            - obj:DataStructure := The wrapped, underlying object
            
            - shape:SHAPE := The shape property of the wrapped object
            
            - dims:DIMS := Labels for the dimentions of the object - the
            axii
            
            - coords:COORDS := Labels for each element in each axis (i.e
            ) distinct row labels, column labels etc
            
            - rank:int := The tensors rank
            
        Methods:
        ---------
        
            Methods exposed by the tensor
        
            - transpose(axis:Optional[AXES_PERMUTATION] = None) := 
            Return a tranposed version of the object. Signature is the
            same as numpy and must return the same default. Should
            always return the same type of object. The T attribute is
            an alias for this methods
            
            - isna(axis:Optional[int] = None) := Elementwise `isnan`.
            Should default to returning the a boolean tensor of the same
            shape as the original tensor. When `axis` is provided should
            this is equivalent to an `any` operation over this axis. The
            axis should be preseved in the return
            
            - any(axis:Optional[int] = None) := When `axis=None` perform
            `any` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - all(axis:Optional[int] = None) := When `axis=None` perform
            `all` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - iterrows() := Iterate over the first axis of the structure
            Similar to `pandas.DataFrame.iterrows()`
            
            - itercolumns() := Iterate over the second axis of the 
            structure. Similar to `pandas.DataFrame.itercolumns`
            
            
    '''
    
    @property
    def obj(self)->DataStructure:
        return self._obj
    
    @obj.setter
    def obj(self, val:DataStructure)->None:
        self._obj = val
    
    @property
    def values(self)->ndarray:
        return self.obj.values
    @values.setter
    def values(self, *args:tuple[Any, ...], **kwargs:dict[Hashable,
                                                          Any])->None:
        raise RuntimeError(("Updating object values not allowed"))
    
    @property
    def shape(self)->SHAPE:
        return self._shape
    @shape.setter
    def shape(self, val:SHAPE)->None:
        self._shape = val
    @property
    def dims(self)->DIMS:
        return self._dims
    @dims.setter
    def dims(self, val:DIMS)->None:
        self._dims = val
    @property
    def coords(self)->COORDS:
        return self._coords
    @coords.setter
    def coords(self, val:COORDS)->None:
        self._coords = val
    @property
    def rank(self)->int:
        return self._rank
    @rank.setter
    def rank(self, val:int)->None:
        self._rank = val
    
    @abstractmethod
    def T(self, axes: Optional[AXIS_PERMUTATION] = None):
        raise NotImplementedError
    
    @abstractmethod
    def any(self, axis:Optional[int] = None)->Union[bool, 
                                                    DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def all(self, axis:Optional[int] = None)->Union[bool, 
                                                    DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def isna(self)->DataStructure:
        raise NotImplementedError()
    
    @abstractmethod
    def transpose(self, axes:Optional[AXIS_PERMUTATION] = None)->\
        DataStructure:
        raise NotImplementedError()

    @abstractmethod
    def iterrows(self)->DataStructure:
        raise NotImplementedError()
    
    @abstractmethod
    def itercolumns(self)->DataStructure:
        raise NotImplementedError()
    

class UtilityMixin:
    
    def _cut_dims_(self, axis:Optional[int])->tuple[DIMS, COORDS]:
        if axis is None:
            return self.dims, self.coords
        from copy import copy
        ndims = copy(list(self.dims))
        ndims.pop(axis)
        ncoords = {
            k:v  for i,(k,v) in enumerate(
                self._coords.items()
                ) if i!=axis}
        return tuple(ndims), ncoords
    
    def _dimshuffle_(self,
                    axes:AXIS_PERMUTATION=None):
        perm = axes if axes is not None else reversed(range(
            len((self.dims))))
        permuted_dims = [self.dims[i] for i in perm]
        permuted_coords = {
            pdim:self.coords[pdim] for pdim in permuted_dims
                        }
        return permuted_dims, permuted_coords

class NDArrayStructure(DataStructure, UtilityMixin):
    
    def __init__(self, obj:ndarray, dims:Optional[DIMS] = None,
                 coords:Optional[COORDS] = None)->None:
        
        self._obj = obj if len(obj.shape)>=2 else obj[None,:]
        self._shape:tuple[int] = obj.shape
        self._dims = np.asarray([
            f"dim_{i}" for i in range(len(obj.shape))]) if dims is \
                None else dims
        self._coords = {i:np.asarray(range(self.obj.shape[k])) for k,i in enumerate(
            self._dims)} if coords is None else coords
        self._rank = len(self.obj.shape)       
        
    @property
    def values(self)->ndarray:
        return self.obj
    
    def isna(self):
        return NDArrayStructure(np.isnan(self.obj),
                                coords = self.coords,
                                dims = self.dims)
        
        
    def any(self, axis:Optional[int] = None, **kwargs):

        if axis is None:
            return self.obj.any(axis=axis, **kwargs)
        else:
            ndims, ncoords = self._cut_dims_(axis)
            return NDArrayStructure(self.obj.any(axis=axis),
                                    dims = ndims,
                                    coords = ncoords)

    def all(self, axis: Optional[int] = None, **kwargs):
        
        if axis is None:
            return self.obj.all(**kwargs)
        else:
            ndims, ncoords = self._cut_dims_(axis)
            return NDArrayStructure(self.obj.any(axis=axis),
                                    dims = ndims,
                                    coords = ncoords)
    
    def transpose(self, 
                  axes:AXIS_PERMUTATION = None):
        tobj = self.obj.transpose(axes)
        permuted_dims, permuted_coords = self._dimshuffle_(axes)
        return NDArrayStructure(tobj,
                              dims = permuted_dims, 
                              coords = permuted_coords
                              )
    T = transpose
    
    def _warn_multidim_iter(self):
        from warnings import warn
        if self.rank >2:
            warn(("Warning! Attempting to iterate over multidimentional"
                  " matrix. If this is intentional, you can ignore this"
                  "warning"))
    
    def iterrows(self):
        self._warn_multidim_iter()
        ndims, ncoords = self._cut_dims_(0)
        for idx, row in enumerate(self.obj):
            yield idx, NDArrayStructure(
                row, dims = ndims, coords = ncoords
            )       
    
    def itercolumns(self):
        self._warn_multidim_iter()
        ndims, ncoords = self._cut_dims_(1)
        swap = [1,0]+[i for i in range(2,self.rank)]
        this = self.obj.transpose(tuple(swap))
        for idx, col in enumerate(this):
            yield idx, NDArrayStructure(
                col, dims = ndims, coords = ncoords
            )
            

class DataFrameStructure(DataStructure, UtilityMixin):
    
    def __init__(self, obj:pd.DataFrame, dims:Optional[DIMS] = None
                 , coords: Optional[COORDS] = None)->None:
        self._obj = obj
        self._shape:tuple[int] = obj.shape
        self._dims = np.asarray(["dim_0", "dim_1"])
        self._coords = dict(dim_0 = self.obj.index, 
                            dim_1 =self.obj.columns)
        self._rank:int = 2
    
    def isna(self):
        return DataFrameStructure( self.obj.isna(), coords=self.coords,
                                dims=self.dims)  
        
    def any(self, axis: Optional[int] = None, **kwargs):
        if axis is None:
            return self.obj.any(axis = axis)         
        elif axis == 0:
            return DataFrameStructure(
                pd.DataFrame(self.obj.any(axis=0).values[None, :],
                             columns = self.coords['dim_1'],
                             index = ["0"]),
            )
        elif axis == 1:
            return DataFrameStructure(
                pd.DataFrame(self.obj.any(axis=1).values[:,None],
                             index = self.coords['dim_0'],
                             columns = ["0"]
                             )
                )
        else:
            raise ValueError(("Pandas DataFrame have exactly two axii."
                              f"Received value {axis} is out of bounds."))

    
    def all(self, axis: Optional[int] = None, **kwargs):
        if axis is None:
            return self.obj.any(axis = axis)         
        elif axis == 0:
            return DataFrameStructure(
                pd.DataFrame(
                    self.obj.all(axis=0).values[None, :],
                    columns = self.coords['dim_1'],
                    index = ["0"]
                )
            )
        elif axis == 1:
            return DataFrameStructure(
                pd.DataFrame(self.obj.all(axis=1).values[:,None],
                             index = self.coords['dim_0'],
                             columns = ["0"]
                             )
                )
        else:
            raise ValueError(("Pandas DataFrame have exactly two axii."
                              f"Received value {axis} is out of bounds."))
            
    
    def transpose(self, axes: Optional[AXIS_PERMUTATION] = None):
        return DataFrameStructure(self.obj.transpose(),
                                dims = [e for e in reversed(self._dims)],
                                coords = {k:v for k,v in reversed(
                                    self._coords.items()
                                )})
    
    T = transpose
    
    def itercolumns(self):
        for i, col in self.obj.iteritems():
            yield i, DataFrameStructure(pd.DataFrame(
                col.values[:,None], index = col.index,
            ))
            
    def iterrows(self):
        for i, row in self.obj.iterrows():
            yield i, DataFrameStructure(
                pd.DataFrame(
                  row.values[None,:], columns = row.index
                )
            )
        

class DataArrayStructure(DataStructure, UtilityMixin):
    
    def __init__(self, obj:xr.DataArray, dims:Optional[DIMS] = None
                , coords: Optional[COORDS] = None)->None:
        self._obj:xr.DataArray = obj
        self._shape:SHAPE = obj.shape
        self._dims = dims if dims is not None else self.obj.dims
        self._coords = coords if coords is not None else {
            k:v.values for k,v in dict(self.obj.coords).items()
            }
        self._rank:int = len(self._coords)        
        
    def all(self, axis: Optional[int] = None, **kwargs)->Union[bool,
                                                DataArrayStructure]:
        ndims, ncoords = self._cut_dims_(axis)
        return DataArrayStructure(self.obj.values.all(axis = axis,
                                                      **kwargs),
                                  dims=ndims, coords = ncoords)
    
    def any(self, axis: Optional[int] = None, **kwargs):
        ndims, ncoords = self._cut_dims_(axis)
        return DataArrayStructure(self.obj.values.any(axis=axis,
                                                      **kwargs),
                                  dims=ndims, coords=ncoords
                                  )
    
    def isna(self):
        return DataArrayStructure( np.isnan(self.obj.values),
                                coords = self._coords,
                                dims = self._dims)
    
    def transpose(self,
                  axes: Optional[AXIS_PERMUTATION] = None):
        permuted_dims, permuted_coords = self._dimshuffle_(axes)
        return DataArrayStructure(
            xr.DataArray(self.obj.values.transpose(axes),
                         coords = permuted_coords),
            dims = permuted_dims, coords = permuted_coords
        )
    T = transpose
    
    def iterrows(self):
        ndims, ncoords = self._cut_dims_(0)
        for i, row in enumerate(self.obj):
            yield i, DataArrayStructure(
                xr.DataArray(row.values, dims=ndims, coords=ncoords),
                                      )
            
    def itercolumns(self):
        ndims, ncoords = self._cut_dims_(1)
        swp = [1,0]+[i for i in range(2, self.rank)]
        for i, col in self.obj.T(tuple(swp)):
            yield i, DataArrayStructure(
                xr.DataArray(
                    col.values, dims=ndims, coords=ncoords
                )
            )


class DataStructureInterface(ABC):
    '''
        Abstract Base Class for the external interface (The bridge 
        Abstraction)
        
        Properties:
        ------------
        
            - data_structure:DataStructure := The core data structure
            implementation
            
        Methods:
        ---------
        
            Methods exposed by the tensor
        
            - transpose(axis:Optional[AXES_PERMUTATION] = None) := 
            Return a tranposed version of the object. Signature is the
            same as numpy and must return the same default. Should
            always return the same type of object. The T attribute is
            an alias for this methods
            
            - isna(axis:Optional[int] = None) := Elementwise `isnan`.
            Should default to returning the a boolean tensor of the same
            shape as the original tensor. When `axis` is provided should
            this is equivalent to an `any` operation over this axis. The
            axis should be preseved in the return
            
            - any(axis:Optional[int] = None) := When `axis=None` perform
            `any` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - all(axis:Optional[int] = None) := When `axis=None` perform
            `all` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - iterrows() := Iterate over the first axis of the structure
            Similar to `pandas.DataFrame.iterrows()`
            
            - itercolumns() := Iterate over the second axis of the 
            structure. Similar to `pandas.DataFrame.itercolumns`
    '''
    
    @property
    @abstractmethod
    def data_structure(self)->DataStructure:
        raise NotImplementedError()
    
    @abstractmethod
    def transpose(self):
        raise NotImplementedError()
    
    @abstractmethod
    def values(self):
        raise NotImplementedError()
    
    @abstractmethod
    def rank(self):
        raise NotImplementedError()
    
    @abstractmethod
    def shape(self):
        raise NotImplementedError()
    
    @abstractmethod
    def dims(self):
        raise NotImplementedError()
    
    @abstractmethod
    def coords(self):
        raise NotImplementedError()
    
    @abstractmethod
    def any(self):
        raise NotImplementedError()
    
    @abstractmethod
    def all(self):
        raise NotImplementedError()
    
    @abstractmethod
    def isna(self):
        raise NotImplementedError()
    
    @abstractmethod
    def itercolumns(self):
        raise NotImplementedError()
    
    @abstractmethod
    def iterrows(self):
        raise NotImplementedError()

@dataclass(kw_only=True)
class CommonDataStructureInterface(DataStructureInterface):
    '''
        Core interface for supported data structures. Should be the only
        'abstraction' provided
        
        Properties:
        ------------
        
            - data_structure:DataStructure := The core data structure
            implementation
            
        Methods:
        ---------
        
            Methods exposed by the tensor
        
            - transpose(axis:Optional[AXES_PERMUTATION] = None) := 
            Return a tranposed version of the object. Signature is the
            same as numpy and must return the same default. Should
            always return the same type of object. The T attribute is
            an alias for this methods
            
            - isna(axis:Optional[int] = None) := Elementwise `isnan`.
            Should default to returning the a boolean tensor of the same
            shape as the original tensor. When `axis` is provided should
            this is equivalent to an `any` operation over this axis. The
            axis should be preseved in the return
            
            - any(axis:Optional[int] = None) := When `axis=None` perform
            `any` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - all(axis:Optional[int] = None) := When `axis=None` perform
            `all` over the entire array and return a boolean. Otherwise
            perform the operation over the specified axis, preserving
            the axis
            
            - iterrows() := Iterate over the first axis of the structure
            Similar to `pandas.DataFrame.iterrows()`
            
            - itercolumns() := Iterate over the second axis of the 
            structure. Similar to `pandas.DataFrame.itercolumns`
    '''
    
    
    _data_structure:Optional[DataStructure] = None
    
    @property
    def data_structure(self) -> DataStructure:
        return self._data_structure # type: ignore
    
    @data_structure.setter
    def data_structure(self, val:DataStructure) -> None:
        self._data_structure = val
    
    def values(self)->ndarray:
        return self.data_structure.values
    
    def shape(self)->SHAPE:
        return self.data_structure.shape
    
    def dims(self)->DIMS:
        return self.data_structure.dims
    
    def coords(self)->COORDS:
        return self.data_structure.coords
    
    def rank(self)->int:
        return self.data_structure.rank
    
    def transpose(self, axes: Optional[AXIS_PERMUTATION] = None):
        return self.data_structure.transpose(axes = axes)
    
    T = transpose
    
    def iterrows(self):
        return self.data_structure.iterrows()
    
    def itercolumns(self):
        return self.data_structure.itercolumns()
    
    def isna(self):
        return self.data_structure.isna()
    
    def any(self, axis: Optional[int] = None, **kwargs):
        return self.data_structure.any(axis = axis, **kwargs)
    
    def all(self,axis: Optional[int] = None, **kwargs):
        return self.data_structure.all(axis = axis, **kwargs)
        
    
