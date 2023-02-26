import xarray as xr
import  pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Hashable, Iterable
from .typing import ndarray, InputData



class DataStructure(ABC):
    
    @abstractmethod
    def T(self):
        raise NotImplementedError
    
    @abstractmethod
    def obj(self):
        raise NotImplementedError
    
    @abstractmethod
    def coords(self):
        raise NotImplementedError
    
    @abstractmethod
    def dims(self):
        raise NotImplementedError
    
    @abstractmethod
    def shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def any(self):
        raise NotImplementedError
    
    @abstractmethod
    def all(self):
        raise NotImplementedError
    
    @abstractmethod
    def isna(self):
        raise NotImplementedError
    
    @abstractmethod
    def transpose(self):
        raise NotImplementedError

    @abstractmethod
    def iterrows(self):
        raise NotImplementedError
    
    @abstractmethod
    def itercolumns(self):
        raise NotImplementedError
    

class UtilityMixin:
    
    @property
    def obj(self):
        return self._obj
    @obj.setter
    def obj(self, val:xr.DataArray):
        self._obj = val
    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, val:tuple[int]):
        self._shape = val
    @property
    def dims(self):
        return self._dims
    @dims.setter
    def dims(self, val:Iterable[Hashable]):
        self._dims = val
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, val:dict[Hashable, Iterable]):
        self._coords = val
    @property
    def rank(self)->int:
        return self._rank
    @rank.setter
    def rank(self, val:int):
        self._rank = val
    
    def _cut_dims_(self, axis:int):
        from copy import copy
        ndims = copy(list(self.dims))
        ndims.pop(axis)
        ncoords = {
            k:v  for i,(k,v) in enumerate(
                self._coords.items()
                ) if i!=axis}
        return ndims, ncoords
    
    def _dimshuffle_(self,
                    axes:Optional[Union[list[int], tuple[int]]]=None):
        perm = axes if axes is not None else reversed(range(
            len((self.dims))))
        permuted_dims = [self.dims[i] for i in perm]
        permuted_coords = {
            pdim:self._coords[pdim] for pdim in permuted_dims
                        }
        return permuted_dims, permuted_coords

class NDArrayAdaptor(UtilityMixin):
    
    def __init__(self, obj:ndarray, dims:Optional[list[str]] = None,
                 coords:Optional[dict[str,Any]] = None)->None:
        
        self._obj = obj if len(obj.shape)>=2 else obj[None,:]
        self._shape:tuple[int] = obj.shape
        self._dims = np.asarray([
            f"dim_{i}" for i in range(len(obj.shape))]) if dims is \
                None else dims
        self._coords = {i:np.asarray(range(self.obj.shape[k])) for k,i in enumerate(
            self._dims)} if coords is None else coords
        self._rank = len(self.obj.shape)       
        
    def isna(self):
        return NDArrayAdaptor(np.isnan(self.obj), coords = self.coords,
                              dims=self.dims)
        
        
    def any(self, axis:Optional[int] = None, **kwargs):

        if axis is None:
            return self.obj.any(axis=axis, **kwargs)
        else:
            ndims, ncoords = self._cut_dims_(axis)
            return NDArrayAdaptor(self.obj.any(axis=axis), dims = ndims,
                                  coords = ncoords)

    def all(self, axis: Optional[int] = None, **kwargs):
        
        if axis is None:
            return self.obj.all(**kwargs)
        else:
            ndims, ncoords = self._cut_dims_(axis)
            return NDArrayAdaptor(self.obj.any(axis=axis), dims = ndims,
                                  coords = ncoords)
    
    def transpose(self, 
                  axes:Optional[Union[list[int], tuple[int]]] = None):
        tobj = self.obj.transpose(axes)
        permuted_dims, permuted_coords = self._dimshuffle_(axes)
        return NDArrayAdaptor(tobj,
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
            yield idx, NDArrayAdaptor(
                row, dims = ndims, coords = ncoords
            )       
    
    def itercolumns(self):
        self._warn_multidim_iter()
        ndims, ncoords = self._cut_dims_(1)
        swap = [1,0]+[i for i in range(2,self.rank)]
        this = self.obj.transpose(tuple(swap))
        for idx, col in enumerate(this):
            yield idx, NDArrayAdaptor(
                col, dims = ndims, coords = ncoords
            )
            

class DataFrameAdaptor(UtilityMixin):
    
    def __init__(self, obj:pd.DataFrame, dims:Optional[list[str]] = None
                 , coords: Optional[dict[str, Any]] = None)->None:
        self._obj = obj
        self._shape:tuple[int] = obj.shape
        self._dims = np.asarray(["dim_0", "dim_1"])
        self._coords = dict(dim_0 = self.obj.index, 
                            dim_1 =self.obj.columns)
        self._rank:int = 2
    
    def isna(self):
        return DataFrameAdaptor( self.obj.isna(), coords=self.coords,
                                dims=self.dims)  
        
    def any(self, axis: Optional[int] = None, **kwargs):
        if axis is None:
            return self.obj.any(axis = axis)         
        elif axis == 0:
            return DataFrameAdaptor(
                pd.DataFrame(self.obj.any(axis=0).values[None, :],
                             columns = self.coords['dim_1'],
                             index = ["0"]),
            )
        elif axis == 1:
            return DataFrameAdaptor(
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
            return DataFrameAdaptor(
                pd.DataFrame(
                    self.obj.all(axis=0).values[None, :],
                    columns = self.coords['dim_1'],
                    index = ["0"]
                )
            )
        elif axis == 1:
            return DataFrameAdaptor(
                pd.DataFrame(self.obj.all(axis=1).values[:,None],
                             index = self.coords['dim_0'],
                             columns = ["0"]
                             )
                )
        else:
            raise ValueError(("Pandas DataFrame have exactly two axii."
                              f"Received value {axis} is out of bounds."))
            
    
    def transpose(self, *args):
        return DataFrameAdaptor(self.obj.tranpose(),
                                dims = [e for e in reversed(self._dims)],
                                coords = {k:v for k,v in reversed(
                                    self._coords.items()
                                )})
    
    T = transpose
    
    def itercolumns(self):
        for i, col in self.obj.iteritems():
            yield i, DataFrameAdaptor(pd.DataFrame(
                col.values[:,None], index = col.index,
            ))
            
    def iterrows(self):
        for i, row in self.obj.iterrows():
            yield i, DataFrameAdaptor(
                pd.DataFrame(
                  row.values[None,:], columns = row.index
                )
            )
        

class DataArrayAdaptor(UtilityMixin):
    
    def __init__(self, obj:xr.DataArray, dims:Optional[list[str]] = None
                , coords: Optional[dict[str, Any]] = None)->None:
        self._obj = obj
        self._shape:tuple[int] = obj.shape
        self._dims = obj.dims
        self._coords = {k:v.values for k,v in dict(self.obj.coords
                                                   ).items()}
        self._rank:int = len(self._coords)        
        
    def all(self, axis: Optional[int] = None, **kwargs)->Union[
        ndarray,bool]:
        
        return self.obj.values.all(axis = axis, **kwargs)
    
    def any(self, axis: Optional[int] = None, **kwargs):
        return self.obj.values.any(axis=axis, **kwargs)
    
    def isna(self):
        return DataArrayAdaptor( np.isnan(self.obj.values),
                                coords = self._coords,
                                dims = self._dims)
    
    def transpose(self,
                  axes: Optional[Union[list[int], tuple[int]]] = None):
        permuted_dims, permuted_coords = self._dimshuffle_(axes)
        return DataArrayAdaptor(
            xr.DataArray(self.obj.values.transpose(axes),
                         coords = permuted_coords),
            dims = permuted_dims, coords = permuted_coords
        )
    T = transpose
    
    def iterrows(self):
        ndims, ncoords = self._cut_dims_(0)
        for i, row in enumerate(self.obj):
            yield i, DataArrayAdaptor(
                xr.DataArray(row.values, dims=ndims, coords=ncoords),
                                      )
            
    def itercolumns(self):
        ndims, ncoords = self._cut_dims_(1)
        swp = [1,0]+[i for i in range(2, self.rank)]
        for i, col in self.obj.T(tuple(swp)):
            yield i, DataArrayAdaptor(
                xr.DataArray(
                    col.values, dims=ndims, coords=ncoords
                )
            )
            
    
        
    
        

        

class DataAdaptorFacade:
    
    def __init__(self,obj)->None:
        pass
