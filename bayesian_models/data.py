import xarray as xr
import  pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from .typing import ndarray, InputData

class DataStructure(ABC):
    
    @abstractmethod
    def T(self):
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
    

class NDArrayAdaptor:
    
    def __init__(self, obj:ndarray, dims:Optional[list[str]] = None,
                 coords:Optional[dict[str,Any]] = None)->None:
        
        self.obj = obj if len(obj.shape)>=2 else obj[None,:]
        self.shape:tuple[int] = obj.shape
        self._dims = np.asarray([
            f"dim_{i}" for i in range(len(obj.shape))]) if dims is \
                None else dims
        self._coords = {i:np.asarray(range(self.obj.shape[k])) for k,i in enumerate(
            self._dims)} if coords is None else coords
        
    def isna(self)->ndarray:
        return np.isnan(self.obj)
        
    def any(self, axis:Optional[int] = None, **kwargs)->ndarray:
        return self.obj.any(axis=axis, **kwargs)

    def all(self, axis: Optional[int] = None, **kwargs):
        return self.obj.all(axis=axis, **kwargs)

class DataFrameAdaptor:
    
    def __init__(self, obj:pd.DataFrame, dims:Optional[list[str]] = None
                 , coords: Optional[dict[str, Any]] = None)->None:
        self.obj = obj
        self.shape:tuple[int] = obj.shape
        self._dims = np.asarray(["dim_0", "dim_1"])
        self._coords = dict(dim_0 = self.obj.index, 
                            dim_1 =self.obj.columns)
    
    def isna(self):
        return self.obj.isna().values
    
    def any(self, axis: Optional[int] = None, **kwargs):
        return self.obj.any(axis = axis)
    
    def all(self, axis: Optional[int] = None, **kwargs)->Union[
        ndarray,bool]:
        if axis == None:
            return self.obj.all(axis = axis)
        else: 
            return self.obj.all(axis = axis).values
        

class DataArrayAdaptor:
    
    def __init__(self, obj:xr.DataArray, dims:Optional[list[str]] = None
                , coords: Optional[dict[str, Any]] = None)->None:
        self.obj = obj
        self.shape:tuple[int] = obj.shape
        self._dims = obj.dims
        self._coords = {k:v.values for k,v in dict(self.obj.coords
                                                   ).items()}
        
    def all(self, axis: Optional[int] = None, **kwargs)->Union[
        ndarray,bool]:
        
        return self.obj.values.all(axis = axis, **kwargs)
    
    def any(self, axis: Optional[int] = None, **kwargs):
        return self.obj.values.any(axis=axis, **kwargs)
    
    def isna(self)->ndarray:
        return np.isnan(self.obj.values)
        


class DataAdaptorFacade:
    
    def __init__(self,obj)->None:
        pass
