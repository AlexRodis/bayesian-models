from __future__ import annotations
import xarray as xr
import  pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Hashable, Iterable, Type
from .typing import ndarray, InputData, SHAPE, DIMS, COORDS
from .typing import AXIS_PERMUTATION
from dataclasses import dataclass, field

# TODO: Impute missing data. Maybe add slicing to the common data 
# interface



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
            
            - dtype := The datatype for the elements. For consistancy
            all `DataStructure` are coerced into homogenous types
            
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
            
            - cast(dtype, **kwargs) := Attemp to cast tensor elements to
            to `dtype`. All kwargs are forwarded to `numpy`. Returns a
            new copy of the tensor (as a DataStructure object) with the
            update data type
            
            
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
    
    @property
    def dtype(self)->Any:
        return self._dtype
    @dtype.setter
    def dtype(self, val:Any)->None:
        self._dtype = val
        
    @property
    def missing_nan_flag(self)->Optional[bool]:
        return self._missing_nan_flag
    @missing_nan_flag.setter
    def missing_nan_flag(self, val:bool)->None:
        self._missing_nan_flag = val
    
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
    
    @abstractmethod
    def cast(self, typ_spec)->DataStructure:
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
    
    def __init__(self, obj:Union[ndarray, DataStructure],
                 dims:Optional[DIMS] = None,
                 coords:Optional[COORDS] = None,
                 dtype:Optional[Any] = None)->None:
        
        self._obj = obj if len(obj.shape)>=2 else obj[:, None]
        self._shape:tuple[int] = self.obj.shape
        self._dims = np.asarray([
            f"dim_{i}" for i in range(len(obj.shape))]) if dims is \
                None else dims
        self._coords = {i:np.asarray(range(self.obj.shape[k])
                                     ) for k,i in enumerate(
            self._dims)} if coords is None else coords
        self._rank = len(self.obj.shape)
        unpacked = obj if isinstance(obj, np.ndarray) else obj.values
        self._dtype = unpacked.dtype if dtype is None else dtype
        self._missing_nan_flag:Optional[bool] = None 
        
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
            temp = self.obj.any(axis=axis)
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
            
            
    def cast(self, dtype, **kwargs):
        return NDArrayStructure(
            self.obj.astype(dtype, **kwargs),
            dims = self.dims, coords = self.coords, dtype=dtype
        )

class DataFrameStructure(DataStructure, UtilityMixin):
    
    accepted_inputs:set=[pd.DataFrame, pd.Series, np.ndarray]
    
    def __init__(self, obj:pd.DataFrame, dims:Optional[DIMS] = None
                 , coords: Optional[COORDS] = None, dtype = None)->None:
        if len(obj.shape) not in set([1,2]):
            raise ValueError(("Unable to coerce input to a DataFrame. "
                              "Valid objects must be 1D or 2D objects,"
                              f" but received {len(obj.shape)}D object"
                              " instead"))
        elif len(obj.shape) == 1:
            self._obj = pd.DataFrame(
                data=obj.values[None, :], columns=obj.index
            )
        else:
            if obj.shape[1] == 1:
                self._obj = pd.DataFrame(data = obj.values)
            self._obj = obj
        self._shape:tuple[int] = self.obj.shape
        self._dims = np.asarray(["dim_0", "dim_1"])
        self._coords = dict(dim_0 = self.obj.index, 
                            dim_1 =self.obj.columns)
        self._rank:int = 2
        self._dtype = obj.values.dtype if dtype is None else dtype
        self._missing_nan_flag:Optional[bool] = None
    
    def isna(self):
        return DataFrameStructure( self.obj.isna(), coords=self.coords, #type:ignore
                                dims=self.dims)  
        
    def any(self, axis: Optional[int] = None, **kwargs):
        if axis is None:
            return self.obj.any(axis = axis)         
        elif axis == 0:
            return DataFrameStructure(
                pd.DataFrame(self.obj.any(axis=0).values[None, :], #type:ignore
                             columns = self.coords['dim_1'],
                             index = ["0"]),
            )
        elif axis == 1:
            return DataFrameStructure(
                pd.DataFrame(self.obj.any(axis=1).values[:,None], #type:ignore
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
                    self.obj.all(axis=0).values[None, :], #type:ignore
                    columns = self.coords['dim_1'],
                    index = ["0"]
                )
            )
        elif axis == 1:
            return DataFrameStructure(
                pd.DataFrame(self.obj.all(axis=1).values[:,None], #type:ignore
                             index = self.coords['dim_0'],
                             columns = ["0"]
                             )
                )
        else:
            raise ValueError(("Pandas DataFrame have exactly two axii."
                              f"Received value {axis} is out of bounds."))
            
    
    def transpose(self, axes: Optional[AXIS_PERMUTATION] = None):
        return DataFrameStructure(self.obj.transpose(), #type:ignore
                                dims = [e for e in reversed(self._dims)], #type:ignore
                                coords = {k:v for k,v in reversed(
                                    self._coords.items()
                                )})
    
    T = transpose
    
    def itercolumns(self):
        for i, col in self.obj.iteritems(): #type:ignore
            yield i, DataFrameStructure(pd.DataFrame(
                col.values[:,None], index = col.index,
            ))
            
    def iterrows(self):
        for i, row in self.obj.iterrows(): #type:ignore
            yield i, DataFrameStructure(
                pd.DataFrame(
                  row.values[None,:], columns = row.index
                )
            )
            
    def cast(self, dtype, **kwargs):
        return DataFrameStructure(pd.DataFrame(
            self.obj.values.astype(dtype, **kwargs),
                            index= self.coords["dim_0"],
                            columns= self.coords["dim_1"]),
                                  dims=self.dims, coords=self.coords,
                                  dtype=dtype)
        

class DataArrayStructure(DataStructure, UtilityMixin):
    
    accepted_inputs:set = set([np.ndarray, pd.DataFrame, pd.Series,
                               xr.DataArray])
    
    def __init__(self, obj:xr.DataArray, dims:Optional[DIMS] = None
                , coords: Optional[COORDS] = None, dtype=None)->None:
        _t = type(obj)
        if _t not in DataArrayStructure.accepted_inputs:
            raise ValueError(("Received invalid input type. Expected "
                              "one of `numpy.ndarray`, `pandas.Series`,"
                              " `pandas.DataArray, or "
                              "`xarray.DataArray`, but received "
                              f"{type(obj)} instead"))
        elif _t == pd.DataFrame:
            self._obj = xr.DataArray(obj.values, coords =dict(
                dim_0 = obj.index, dim_1 = obj.columns
            ))
            self._dtype = dtype if dtype is not None else \
                obj.values.dtype
        elif _t == pd.Series:
            self._obj = xr.DataArray(obj.values[None,:], coords =dict(
                dim_0 = ["0"], dim_1 = obj.index 
            ))
            self._dtype = dtype if dtype is not None else \
                obj.values.dtype
            
        elif _t == np.ndarray:
            self._obj = xr.DataArray(obj, coords = {
                f"dim_{i}": np.asarray(range(axis)) for i, axis in \
                    enumerate(obj.shape)
            })
            self._dtype = dtype if dtype is not None else obj.dtype
        else:
            self._obj:xr.DataArray = obj
            self._dtype = obj.dtype
        self._shape:SHAPE = self._obj.shape
        self._dims = dims if dims is not None else np.asarray(
            self.obj.dims)
        self._coords = coords if coords is not None else {
            k:v.values for k,v in dict(self.obj.coords).items()
            }
        self._rank:int = len(self._coords)
        self._missing_nan_flag:Optional[bool] = None      
        
    def all(self, axis: Optional[int] = None, **kwargs)->Union[bool,
                                                DataArrayStructure]:
        ndims, ncoords = self._cut_dims_(axis)
        core_obj = self.obj.values.any(axis=axis,**kwargs)
        if axis is None:
            return core_obj
        else:
            return DataArrayStructure(core_obj, dims=ndims,
                                      coords = ncoords)
    
    def any(self, axis: Optional[int] = None, **kwargs):
        ndims, ncoords = self._cut_dims_(axis)
        core_obj = self.obj.values.any(axis=axis,**kwargs)
        if axis is None:
            return core_obj
        else:
            return DataArrayStructure(
        core_obj if len(core_obj.shape)>=2 else core_obj[:,None],
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
    def cast(self, dtype, **kwargs):
        return DataArrayStructure(
            xr.DataArray(self.obj.values.astype(dtype, **kwargs),
                         dims = self.dims, coords = self.coords),
            dims = self.dims, coords = self.coords, dtype = dtype
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
    def transpose(self)->DataStructureInterface:
        raise NotImplementedError()
    
    @abstractmethod
    def values(self)->np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def rank(self)->int:
        raise NotImplementedError()
    
    @abstractmethod
    def shape(self)->SHAPE:
        raise NotImplementedError()
    
    @abstractmethod
    def dims(self)->DIMS:
        raise NotImplementedError()
    
    @abstractmethod
    def coords(self)->COORDS:
        raise NotImplementedError()
    
    @abstractmethod
    def dtype(self)->Any:
        raise NotImplementedError()
    
    @abstractmethod
    def any(self)->Union[DataStructureInterface, bool, np.bool_]:
        raise NotImplementedError()
    
    @abstractmethod
    def all(self)->Union[DataStructureInterface, bool, np.bool_]:
        raise NotImplementedError()
    
    @abstractmethod
    def isna(self)->Union[DataStructureInterface, bool, np.bool_]:
        raise NotImplementedError()
    
    @abstractmethod
    def itercolumns(self)->DataStructureInterface:
        raise NotImplementedError()
    
    @abstractmethod
    def iterrows(self)->DataStructureInterface:
        raise NotImplementedError()
    
    @abstractmethod
    def astype(self, dtype, kwargs)->DataStructureInterface:
        raise NotImplementedError()

@dataclass(kw_only=True)
class CommonDataStructureInterface(DataStructureInterface):
    '''
        Core interface for supported data structures. Should be the only
        'refined abstraction' provided
        
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
    _implementor:Optional[Type[DataStructure]] = None
    
        
    def __post_init__(self):
        self._implementor = type(self._data_structure) #type: ignore
    
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
    
    def transpose(self, axes: Optional[AXIS_PERMUTATION] = None
                 )->CommonDataStructureInterface:
        return CommonDataStructureInterface(
            _data_structure = self.data_structure.transpose(axes = axes)
            )
    
    T = transpose
    
    def dtype(self):
        return self.data_structure.dtype
    
    def iterrows(self):
        return CommonDataStructureInterface(
            _data_structure = self.data_structure.iterrows()
            )
    
    def itercolumns(self):
        return CommonDataStructureInterface(
            _data_structure = self.data_structure.itercolumns()
            )
    
    def isna(self):
        struct = self.data_structure.isna()
        if isinstance(struct, np.bool_):
            return bool(struct)
        return CommonDataStructureInterface(
            _data_structure = struct)
    
    def any(self, axis: Optional[int] = None, **kwargs)->Union[
        CommonDataStructureInterface, np.bool_]:
        struct = self.data_structure.any(axis = axis, **kwargs)
        if isinstance(struct, np.bool_):
           return bool(struct) 
        return CommonDataStructureInterface(
            _data_structure = struct
        )
    
    def all(self,axis: Optional[int] = None, **kwargs):
        struct = self.data_structure.all(axis = axis, **kwargs)
        if isinstance(struct, np.bool_):
            return bool(struct)
        return CommonDataStructureInterface(
            _data_structure = struct
        )
    
    def missing_nan_flag(self)->Optional[bool]:
        return self.data_structure.missing_nan_flag
    
    def astype(self, dtype, **kwargs):
        return CommonDataStructureInterface(
            _data_structure = self.data_structure.cast(dtype, **kwargs)
        )
    

class NANHandler(ABC):
    '''
        Abstract Base Class for missing value handlers
    '''
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        raise NotImplementedError()

@dataclass
class ImputeMissingNAN(NANHandler):
    '''
        Core class from missing data imputation strategy. Currently not
        implemented and will raise
    '''
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        raise NotImplementedError(("Data Imputation not yet "
                                   "implemented"))

@dataclass
class ExcludeMissingNAN(NANHandler):
    '''
        Common use-case for missing value handling. Discards all coordinates
        on the first dimention (i.e. rows) along which there are any missing 
        values - updating the objects' metadata
        
        Object Attributes:
        --------------------
        
            - new_coords:Optional[COORDS]=None := Updated object coordinates
            
            - new_dims:Optional[DIMS]=None := Updated object dimentions
            
            - axis:int=0 := The dimention along which to exclude. Optional.
            Defaults to 0 ('rows'). Currently no other options are implemeted
            and other values are ignored.
            
            - constructor:Optional[DataStructure]=None := The DataStructure of
            the object to convert after processing
            
            
        Object Methods:
        ----------------
             - __call__(data:DataStructureInterface)->DataStructureInterface
             := Handle missing values and return the updated object
    
    '''
    
    new_coords:Optional[COORDS] = None
    new_dims:Optional[DIMS] = None
    axis:int = 0 # Unused
    constructor:Optional[DataStructure] = None
    
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        from copy import copy
        
        self.constructor = type(data._data_structure)
        
        indices:CommonDataStructureInterface = data.isna()
        
        for i,_ in enumerate(data.dims()[1:], 1):
            indices = indices.any(axis=1) # type:ignore
        
        # Try reshape
        not_nan = np.logical_not(indices.values()[:,0])
        
        clean_data= data._data_structure._obj[not_nan]
        
        self.new_coords = copy(data.coords())
        self.new_coords[data.dims()[0]] = np.asarray([
            coord  for i, coord in enumerate(
                data.coords()[data.dims()[0]]
                ) if i in np.where(not_nan)[0]
        ])
        self.new_dims = data.dims()
        this = CommonDataStructureInterface(
            _data_structure = self.constructor(clean_data,
                                    coords=self.new_coords,
                                    dims = self.new_dims
                                    )
            )
        return this

@dataclass
class IgnoreMissingNAN(NANHandler):
    '''
        Identity strategy for nan handling the does nothing. Only included
        for completeness' sake. Returns the object unmodified
        
        Object Methods:
        ---------------
        
            - __call__(data:DataStructureInterface)->DataStructureInterface :=
            Returns the data unchanged
    '''
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        return data


@dataclass(kw_only = True)
class NANHandlingContext:
    '''
        Composite for missing values handlin. Defines the external interface
        
        Object Properties:
        --------------------
        
            - nan_handler:NANHandler := The nan handling strategy to apply
            
        Objcet Methods:
        ------------------
        
            - __call__(data:DataStructureInterface))->DataStructureInterace :=
            Delegate missing value handling to the handler and return the
            results
    '''
    _nan_strategy:Type[NANHandler] = ExcludeMissingNAN
    nan_handler:Optional[NANHandler] = None
    kwargs:dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.nan_handler = self._nan_strategy(**self.kwargs)# type:ignore
        
    def __call__(self, data:DataStructureInterface
                 )->DataStructureInterface:
        return self.nan_handler(data) #type:ignore


class DataProcessor(ABC):
    '''
        Abstract base class for Data Processors
    '''
    
    @abstractmethod
    def __call__(self, data: DataStructure)->DataStructureInterface:
        raise NotImplementedError()

@dataclass(kw_only = True)
class CommonDataProcessor(DataProcessor):
    '''
        Common use-case data pre processor. Will handle:
        
            - converting the data structure to a common internal interface
            
            - handling of missing values
            
            - casting to data type
            
            - validate data types
        
        Can be subclassed for extended functionality or overriden.
        
        Object Attributes:
        --------------------
        
            - nan_handler:NANHandlerContext := The missing values handler.
            Optional. Defaults to ExcludeMissingNAN. Initially a ref to the 
            context class, will be replaced by a instance of that class.
            
            - cast:Optional[np.dtype]=None := Attempt to forcefully cast all
            inputs to the specified type. Optional. Defaults to `np.float32`.
            Set to `None` will disable typecasting
            
            - type_spec := Schema to validate. Not implemented and will be 
            ignored
            
            - casting_kwargs:dict={} := Keyword arguements to be forwarded to
            the underlying typecaster. See numpy for details. Defaults to an
            empty dict. Warning! Poorly tested!
            
        Object Methods:
        ----------------
        
            - __call__(data:InputData)->CommonDataStructureInterface :=
            Preprocess the data according the set options and return the
            result.
            
                   
    '''
    
    nan_handler_context:NANHandlingContext = NANHandlingContext(
        _nan_strategy = ExcludeMissingNAN
    )
    cast:Optional[np.dtype] = None
    type_spec:Any = None
    casting_kwargs:dict = field(default_factory = dict)
    
    def _convert_structure(self, data: InputData
                           )->DataStructureInterface:
        '''
            Convert the input structure to a common interface by wrapping
            it in the appropriate implementation class
            
            Args:
            -----
            
                - data:InputData := The data to process
                
            Returns:
            --------
            
                - common:CommonDataStructureInterface := The object, converted
                to a common interface
                
            Raises:
            -------
            
                - RuntimeError := When unable to identify the data type. This
                is a last-resort exception. Should be handled elsewhere
        '''
        core_type:str = str(type(data)).split(".")[-1][:-2]
        struct:Optional[Type[DataStructure]] = None
        if core_type == "ndarray":
            struct = NDArrayStructure
        elif core_type == "DataFrame":
            struct = DataFrameStructure
        elif core_type == "DataArray":
            struct = DataArrayStructure
        else:
            raise RuntimeError("Unable to convert data type")
            
        return CommonDataStructureInterface(
            _data_structure = struct(data) ) # type: ignore
    
    def _handle_nan(self, data:DataStructureInterface
                    )->DataStructureInterface:
        return self.nan_handler_context(data)
    
    def _cast_data(self, data:DataStructureInterface,
                   )->DataStructureInterface:
        if self.cast is None:
            return data
        else:
            return data.astype(self.cast, **self.casting_kwargs)
            
    def _validate_dtypes(self, data:DataStructureInterface
                         )->DataStructureInterface:
        if self.type_spec is None:
            return data
        else:
            return data
        
    def _detect_missing_nan(self, data:DataStructureInterface)->bool:
        return data.isna() #type:ignore
    
    def __call__(self, data: InputData)->DataStructureInterface:
        '''
            Preprocess the object according to set options and return the
            results
            
            Args:
            -----
            
                data:InputData := The data to process
                
            Returns:
            ---------
            
                - processed:CommonDataStructureInterface := The processed data
        '''
        _data = self._convert_structure(data)
        _data.data_structure._missing_nan_flag = \
            self._detect_missing_nan(_data)
        _data = self._handle_nan(_data)
        _data = self._cast_data(_data)
        return _data
    


@dataclass(kw_only=True)
class DataProcessingDirector:
    '''
        Master composite for data pre processing
        
        Object Attributes:
        ------------------
        
            - processor:CommonDataProcessor := A reference to the class
            that represents the data processor. Converted to an instance
            on said class. Optional. Defaults to `CommonDataProcessor`
            
            - nan_handler_context:NANHanderContext := The missing nan handler
            
            
            - processor_kwargs:dict = Keyword arguments to be forwarded to
            the processor instance
            
        Object Methods:
        ----------------
        
            - __call__(data:InputeData)->CommonDataStructureInterface := Call
            the processor and return the pre processed data
    '''
    
    processor:Union[Type[DataProcessor],
                    DataProcessor] = CommonDataProcessor
    nan_handler_context:NANHandlingContext = NANHandlingContext(
        _nan_strategy = ExcludeMissingNAN
    )
    processor_kwargs:dict = field(default_factory = dict)
    
    def __post_init__(self)->None:
        self.processor = self.processor(
            nan_handler_context = self.nan_handler_context,
            **self.processor_kwargs) #type:ignore
    
    def __call__(self, data: InputData)->CommonDataStructureInterface:
        if self.processor is not None:
            return self.processor(data) #type:ignore


class Data:
    '''
        Container for model data with optional preprocessing
        functionality.
        
        Class Attributes:
        ------------------
        
            - nan_handlers:set[str]=['exlude', 'impute', 'ignore'] := 
            Valid strategies for missing value handling
            
            - input_types:set[str]=['ndarray', 'DataFrame', 'DataArray'] := 
            Supported input data structures
            
        Object Attributes:
        -------------------
            
            - nan_handling:str='exclude' := The missing data handling 
            strategy. Has to be one of Data.nan_handlers. Optional. Defaults
            to 'exclude' and discards all axis=0 coordinates with missing
            values (i.e rows).
            
            - cast:Any := A data type to force-cast the data to. Optional.
            Defaults to `numpy.float32`. Set to `None` to disable casting
            
            - type_spec:dict={} := Dictionary specification data validation
            across the second dimention. Keys should be coordinates (labels)
            along the second axis(=1) and values should be valid numpy dtypes.
            Currently ignored
            
            - casting_kwargs:dict={} := Optional keyword arguments to be
            forwarded to the type caster. Optional. Defaults to an empty dict.
            See the `numpy` documentation for further details. Ignored if
            `cast=None`
            
            - processor:Type[DataProcessor]=CommonDataProcessor := The 
            processor to be used for data processing. Optional and defaults to
            the generic data processor. Can be overriden to customized
            with a user specified processor that subclasses `DataProcessor` or
            `CommonDataProcessor`
            
            - process_director:Optional[DataProcessDirector] := The 
            director for data processing. Optional
            
            - nan_handler:Optional[NANHandler]=None := The class that
            handles missing values. None only when unset
                
        Object Methods:
        ----------------
        
            - __call__(data:InputData) := Process the data and return
            the result
    
    '''

    nan_handlers:set[str] = set(["exclude", "impute", "ignore"])
    input_types:set[str] = set(["ndarray", "DataFrame", "DataArray"])

    def __init__(self, nan_handling:str='exclude',
                 processor:Type[DataProcessor] = \
                     CommonDataProcessor,
                 cast:Any = np.float32, type_spec:dict = {},
                 casting_kwargs:dict = {}, 
                 )->None:
        
        if nan_handling not in Data.nan_handlers:
            raise ValueError(("Receive illegal value for 'nan_handling'"
                              " argument. Expected on of 'ignore' "
                              "'impute' or 'ignore', received "
                             f"{nan_handling} instead"))
        self.nan_handling:str = nan_handling
        self.nan_handler:Optional[Type[NANHandler]] = None
        self.data_processor:Type[DataProcessor] = processor
        self.cast = cast
        self.process_director:Optional[DataProcessingDirector] = None
        self.type_spec = type_spec
        self.casting_kwargs = casting_kwargs
    
    def __call__(self,data:InputData)->CommonDataStructureInterface:
        '''
            Process input data according to specifications
            
            Args:
            -----
            
                - data:InputData := The data to process
                
            Returns:
            -------
            
                - processed:DataStructureInterface := Container for the
                processed and harmonized data
                
            Raises:
            -------
            
                - ValueError := (1) If the objects' type is not included
                as one of the valid options, (2) if `nan_handling` is not a
                valid option
        '''
        inpt_type:str = str(type(data)).split(".")[-1].strip(">`'")    
        if inpt_type not in Data.input_types:
            raise ValueError(("Uknown data type received. Expected "
                              "one of `numpy.ndarray`, "
                              "`pandas.DataFrame` or `xarray.DataArray`"
                              f" but received {inpt_type} instead"))
        if self.nan_handling == "exclude":
            self.nan_handler = NANHandlingContext(
                _nan_strategy = ExcludeMissingNAN)
        elif self.nan_handling == "impute":
            self.nan_handler = NANHandlingContext(
                _nan_strategy = ImputeMissingNAN
            )
        elif self.nan_handling == "ignore":
            self.nan_handler = NANHandlingContext(
                _nan_strategy = IgnoreMissingNAN
            )
        else:
            raise ValueError(("Unrecognized strategy for missing nan."
                              "Valid options are 'ignore', 'impute' "
                              "and 'exclude'. Received "
                              f"{self.nan_handling} instead"))
            
        self.process_director = DataProcessingDirector(
            processor = self.data_processor,
            nan_handler_context = self.nan_handler,
            processor_kwargs = dict(
                cast = self.cast,
                type_spec = self.type_spec,
                casting_kwargs = self.casting_kwargs
                               )
            )
        return self.process_director(data)
        