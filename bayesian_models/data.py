from __future__ import annotations
import xarray as xr
import  pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Hashable, Iterable, Type
from typing import Callable, NamedTuple
from .typing import ndarray, InputData, SHAPE, DIMS, COORDS
from .typing import AXIS_PERMUTATION
from dataclasses import dataclass, field

# TODO: 
# * Impute missing data. Maybe add slicing to the common data 
# interface
# * Much of this code is inefficient. Refactor
# * Replace long if/else statements with structural pattern matching
# * xarray constructor needs refactoring
# * Lots of repetition in both __getitem__ and unique implementations



# Data Types Bridge
class DataStructure(ABC):
    r'''
        Abstract Base Class for Data Structure implementations
        
        Properties:
        ------------
            Common properties exposed by the underlying object
        
            - obj:DataStructure := The wrapped, underlying object
            
            - shape:SHAPE := The shape property of the wrapped object
            
            - | dims:DIMS := Labels for the dimentions of the object - |
              the axii
            
            - | coords:COORDS := Labels for each element in each axis
              (i.e |) distinct row labels, column labels etc
            
            - rank:int := The tensors rank
            
            - | dtype := The datatype for the elements. For consistency
              |all `DataStructure` are coerced into homogenous types
            
        Methods:
        ---------
            Methods exposed by the tensor
        
            - | transpose(axis:Optional[AXES_PERMUTATION] = None) :=
                Return a transposed version of the object. Signature is
                the same as numpy and must return the same default.
                Should always return the same type of object. The T
                attribute is an alias for this method
            
            - | isna(axis:Optional[int] = None) := Elementwise `isnan`.
                Should default to returning the a boolean tensor of the
                same shape as the original tensor. When `axis` is
                provided should this is equivalent to an `any` operation
                over this axis. The axis should be preserved in the
                return
            
            - | any(axis:Optional[int] = None) := When `axis=None`
                perform `any` over the entire array and return a
                boolean. Otherwise perform the operation over the
                specified axis, preserving the axis
            
            - | all(axis:Optional[int] = None) := When `axis=None`
                perform `all` over the entire array and return a
                boolean. Otherwise perform the operation over the
                specified axis, preserving the axis
            
            - | iterrows() := Iterate over the first axis of the
                structure. Similar to `pandas.DataFrame.iterrows()`
            
            - | itercolumns() := Iterate over the second axis of the
                structure. Similar to `pandas.DataFrame.itercolumns`
            
            - | cast(dtype, **kwargs) := Attempt to cast tensor elements
                to `dtype`. All kwargs are forwarded to `numpy`. Returns
                a new copy of the tensor (as a DataStructure object)
                with the update data type
            
            - | __getitem__(self, obj) := Slice the object in any number
                of ways. Integer and label based, slices should be
                acceptable along with arbitrary combinations thereof.
                Acceptable slice inputs should be: int, str, slice,
                Ellipsis, list[int,str], tuple[int,str] and all
                combinations of these. Slice should accept label based
                specs or integers based ones, or mixes of the two. The
                step argument only accept integers and raise otherwise.
                All of the following should be valid:
                
                .. code-block::
                
                   obj[5,6] obj["sample_0", 7] obj["sample_7":10:1, ...]
                   obj[[6,9], "var1":10:2,...]
                
                If the object would be reduced below a 2d structure, it
                should padded into 2D as a row-vector
            
            - | unique(axis=None) := Return a the unique values of the
                data structure as a generator of length 2 tuples. When
                axis is specified as an integer, the generator iterates
                over the specified axis, yielding tuples of the label
                and the unique values of the subtensor. When axis is
                None (default) return a single element Generator which
                yields exactly one tuple of (None, UNIQUES), where
                UNIQUES is a vector of all the unique values in the
                structure
            
            - | mean(axis:Optional[int] = None, keepdims:bool=True,
                skipna:bool=True) := Compute the mean along the
                specified axis. If axis is `None` the mean will be
                computed over the entire structure and a numeric is
                returned. Otherwise a data structure of the same type as
                the original is returned. If axis is not None, the mean
                is computed over the specified axis. If `keepdims=True`
                (default) the axis is reduced and removed. If
                `keepdims=True` then the axis is maintained (and the
                result is broadcastable to the original) with a single
                coordinate named "sum". If `skipna=True` any invalid
                elements will be ignored (default) otherwise `nan` is
                returned where mean would otherwise be.
            
            - | ops := Elementwise comparison operations such as '>',
                '>=', '==', '<=', '<' and 'neq' are included in the
                interface but generally delegated to the underlying
                library 
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
    
    @abstractmethod
    def __eq__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def __ne__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def __lt__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def __le__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def __ge__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @abstractmethod
    def __gt__(self)->Union[bool, DataStructure]:
        raise NotImplementedError()
    
    @staticmethod
    def __isna__(array):
        '''
            Custom `numpy.isnan` implementation, capable of handling arrays
            of objects. Exploits the fact that in `numpy` and
            derived implementations `numpy.nan!=numpy.nan`
        '''
            
        cmp:Callable = np.vectorize(lambda elem : elem!=elem)
        return cmp(array)
    
    def __mean__(self, obj, axis: Optional[int] = None, 
                 skipna:bool=True, keepdims: bool=True)->NamedTuple:
        r'''
            Compute the mean along the specified axis. 

            Args:
            -----
            
                - obj := The data structure object
                
                - | axis:Optional[int] = None := The dimension along
                    which the mean will be computed. When `None`
                    computes the mean of the entire structure. When an
                    integer is specified, return a data structure, of
                    the same type as the original with the computed
                    means
                
                - | skipna:bool=True := If `True`, `NaN` values will be
                    ignored during computation. Else `NaN` will be
                    returned if any `NaN` values are encountered along
                    each coordinate
                    
                - | keepdims:bool=True := If `True`, the axis along
                        which the mean is computed in preserved in the
                        returned structure, making it correctly
                        broadcastable to the original. Otherwise, the
                        dimention is reduced. If this reduction would
                        reduce the structure below 2D, a 2D row-vector
                        structure is returned instead
                
            Returns:
            ---------
            
                - | mean:float := The mean of the entire structure (if
                  `axis=None`)
            
                - | Results:namedtuple: A namedtuple containing the
                    results of mean. Has three fields 'structure' the
                    actual structure, 'dims' for the dimentions after
                    computation and coords containing the coordinates of
                    the result
        '''
        from copy import copy
        from collections import namedtuple
        Results = namedtuple('Results', ['structure', 'dims', 'coords'])
        if axis is None:
            return np.nanmean(obj)
        else:
            ndims:int = len(obj.shape)
            computer = np.nanmean if skipna else np.mean
            vals = computer(obj, keepdims = keepdims,
                            axis=axis)
            if keepdims or ndims==2:
                ndims:DIMS = copy(self.dims)
                ncoords:COORDS = {
                    k:(v if i!=axis else np.asarray(["sum"])
                    ) for i, (k,v) in enumerate(self.coords.items())
                }
            else:
                ndims:DIMS = copy(self.dims).tolist()
                ndims.pop(axis)
                ndims = np.asarray(ndims)
                ncoords:COORDS = {
                    k:v for i, (k,v) in enumerate(self.coords.items()) if i!=axis 
                }
        return Results(structure = vals, dims = ndims, coords = ncoords)

    @abstractmethod
    def mean(self, axis:Optional[int] = None, keepdims:bool=True,
             skipna:bool=True):
        raise NotImplementedError()
    
    def _slice_coords(self, obj:Iterable)->COORDS:
        r'''
            Convert the argument to indexer into coordinates
            
            Given a slice/index type object, collects the sliced
            objects' labels as coords and returns them
            
            Args:
            -----
            
                - obj:Iterable := The input to the indexer function
                
            Returns:
            --------
            
                - coords:COORDS := The corresponding coordinates object
                  (a dictionary of dimension names to numpy arrays or
                  coordinates)
        '''
        from copy import copy
        from itertools import count
        odims = self._dims
        ocoords = self._coords
        ncoords = copy(self._coords)
        for i, (dim, e) in enumerate(zip(odims, obj)):
            if e is Ellipsis:
                break
            elif isinstance(e, slice):
                ncoords[dim] = ocoords[dim][e]
            else:
                del ncoords[dim]
        if len(ncoords)>=2:
            return ncoords
        else:
            for i in count():
                if not f"dim_{i}" in set(ncoords.keys()):
                        ncoords = {
                            f"dim_{i}": np.asarray([0]),
                            }|ncoords
                        break
            return ncoords

    
    def __getitem__(self, obj:Union[str, int, Iterable]
                    )->Union[NDArrayStructure, np.ndarray]:
        r'''
            Index slicing for CommonDataStructure objects. 
            
            Index or label based slicing is supported in arbitrary
            combinations of DataStructure objects can be sliced with
            (nearly) any combination of int, str, slice, list, None,
            Ellipsis. Note label slicing is supported, however the
            `step` argument must be blank or an integer, not a label.
            Example usage:
            
            .. code-block::
            
                # Pseudo-code
                obj = DataStructure()
                obj[5]
                obj[:5]
                # Mix and match labelling
                obj["sample_0", [0,6], 0:6:3,...]
                # Legal
                obj["sample_0":"sample_10":1]
                # Illegal
                obj["sample_0":"sample_10":"group"]
                
            Returns:
            --------
                - numpy.NDArray := If boolean indexing or an exact element
                is selected i.e. `obj[1,0,1]` or `obj[obj.values>5]`
                
                - DataStructure := Of the same type as the original. Note
                is all cases where the resulting structure would have been
                1-D or 0-D, a 2-D array is returned i.e. instead of
                (9,) (1, 9) is returned
        '''
        lookup = lambda dim ,e: int(np.where(
                        self.coords[dim]== e
                        )[0]) if isinstance(e, str) else e
        nobj:tuple = obj if isinstance(
            obj, tuple) else (obj, )
        isarray_indexing:bool = isinstance(nobj[0], np.ndarray)
        boolean_indexing:bool = isarray_indexing and nobj[0].dtype=='bool'
        if boolean_indexing:
            return self._obj[obj]
        if Ellipsis not in nobj and len(nobj)<self.rank:
            nobj = tuple(list(nobj)+[...])
        try:
            collected:list = []
            for dim, e in zip(self.dims ,nobj):
                if e is Ellipsis:
                    collected.append(Ellipsis)
                    break
                elif isinstance(e, (str, int)):
                    i = lookup(dim, e)
                    collected.append(i)
                elif isinstance(e, slice):
                    if not (isinstance(e.step, int) or e.step is None):
                        raise IndexError((
                            "Step parameter of a slice indexer cannot be"
                            "be label. Expecte None or an integer greater"
                            f"than 0, received {e.step} of type "
                            f"{type(e.step)} instead"
                        ))
                    collected.append(slice(
                        lookup(dim, e.start),
                        lookup(dim, e.stop),
                        e.step # Cannot be a label. Raise
                    ))
                elif isinstance(e, list):
                    subcollected:list[int] = []
                    for elem in e:
                        alltype = (int, str, np.str_, np.int_,
                                   )
                        if isinstance(elem, alltype):
                            subcollected.append(lookup(dim, elem)
                                )
                        else:
                            raise IndexError((
                                "Only lists of integers and labels are "
                                f"valid indexers. Received {elem} of type "
                                f"{type(elem)}"
                            ))
                    collected.append(subcollected)
        except TypeError:
            raise IndexError(f"Indexer {e} is invalid")
        nobj = tuple(collected)
        exact_match:bool = not any([
            isinstance(e, (slice, list, tuple)) or e is Ellipsis for e in nobj
            ])
        if not exact_match:
            ncoords = self._slice_coords(nobj)
            ndims = np.asarray([k for k in ncoords.keys()])
            return NDArrayStructure(
                self._obj[nobj],
                dims = ndims, coords=ncoords
            )
        else:
            return self._obj[nobj]

class UtilityMixin:
    
    def _cut_dims_(self, axis:Optional[int])->tuple[DIMS, COORDS]:
        from copy import copy
        if axis is None:
            return self.dims, self.coords
        elif len(self.dims)<=2:
            ndims = copy(self.dims)
            ncoords = {
                ndims[0] : np.asarray([0]),
                ndims[1] : np.asarray(self.coords[self.dims[0]])
            }
            return ndims, ncoords
        else:
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
    r'''
        Wrapper class around numpy arrays implementing the common
        interface.
        
        Like all implementations this class implements a common,
        standardized interface for acceptable tensor data structures, as
        defined by the `DataStructure` abstract base class
        
        Object Properties:
        ------------------
        
            - obj:numpy.typing.NDArray := The underlying `numpy.ndarray`
              object
            
            - shape:tuple[int,...] := The shape of the object
            
            - dims:DIMS := Dimensions of the object. A `numpy` vector of
              labels of the dimensions / axes of the object. For numpy arrays defaults are typically used (since numpy arrays have no labels). The default names are 'dim_{i}' where i the integer indexer of the axis
            
            - coords:COORDS := The coordinates of the object. Is a
                dictionary of strings, which are axes names (the same as
                those of `dims`) mapped to numpy vectors of labels.
                These are the labels of the 'steps' in each axis
            
            - rank:int := The structures' rank i.e. the number of axes
            
            - dtype:np.dtype := The data type of the structure
            
            - missing_nan_flag:Optional[bool] = None := Flag for
              existence of   missing values. Should be set by the public
              interface class
              
        Object Methods:
        ---------------
        
            - isna()
            
            - any()
            
            - all()
            
            - transpose()
            
            - iterrows()
            
            - itercolumns()
            
            - cast()
            
            - unique()
            
            - mean()
    '''
    
    def __init__(self, obj:Union[ndarray, DataStructure],
                 dims:Optional[DIMS] = None,
                 coords:Optional[COORDS] = None,
                 dtype:Optional[Any] = None)->None:
        
        self._obj = obj if len(obj.shape)>=2 else obj[None, :]
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
    
    def __eq__(self, obj):
        raw = self._obj == obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ne__(self, obj):
        raw = self._obj != obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __lt__(self, obj):
        raw = self._obj < obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __gt__(self, obj):
        raw = self._obj > obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ge__(self, obj):
        raw = self._obj >= obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __le__(self, obj):
        raw = self._obj <= obj
        if isinstance(raw, bool):
            return raw
        else:
            return NDArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )

    def isna(self):
        r'''
            Check if the structure has missing or `nan` values
            
        '''
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
        
    def unique(self, axis:Optional[int]=None):
        r'''
            Return unique values of the NDArrayStructure as Generator
            of length 2 tuples. When axis is None, the generator yields
            a single tuple of (None, vals) where vals are all the unique
            values in the array. When axis is specified, the Generator
            iterates over the specified axis, yielding tuples of label,
            unique_values
        '''
        if axis is None:
            yield (None, np.unique(self._obj))
        else:
            axii = [axis]+[e for e in range(len(self.dims)) if e!=axis]
            crds = self.coords.get(list(self.coords.keys())[axis])
            for subtensor, crd in zip(np.transpose(self._obj, axii),
                                 crds):
                yield (crd , np.unique(subtensor))
    
    def mean(self, axis:Optional[int] = None, keepdims:bool=True,
             skipna:bool=True):
        r'''
            Compute the arithmetic mean along the specified axis. 
            
            Args:
            ------
            
                - axis:Optional[int]=None : = When `axis=None` return the scalar 
                mean of the entire structure (default). Otherwise computes 
                the mean along the specified axis.
                
                - keepdims:bool=True := If `keepdims=True` the dimention is maintained 
                in the resulting structure with a single coordinate named 
                'sum' (default). Otherwise, the dimention is reduced and 
                removed from the resulting structure. 
                
                - skipna:bool=True :=  If `skipna=True` `NaN` values will be 
                ignored in the result, otherwise `NaN` is returned for 
                coordinates with at least one `NaN`
                
            Returns:
            ---------
            
                - nstruct:NDArray := Returns a new NDArrayStructure of means
                
                - mean:float := If `axis=None` the mean of the entire structure
        '''
        results:Union[NamedTuple, float] = super().__mean__(
            self._obj, axis=axis, keepdims=keepdims, skipna=skipna
            )
        if axis is None:
            return results
        else:
            return NDArrayStructure(
            results.structure, coords= results.coords, dims=results.dims
            )
        

class DataFrameStructure(DataStructure, UtilityMixin):
    
    r'''
        Wrapper class around `pandas.DataFrame` implementing the
        common interface.
        
        Like all implementations this class implements a common,
        standardized interface for acceptable tensor data
        structures, as defined by the `DataStructure` abstract base
        class.
        
        Class Attributes:
        -----------------
        
            - | accepted_inputs:set := Valid inputs the constructor.
                Structures which can be converted are Series, DataFrames
                and numpy ndarrays
        
        Object Properties:
        ------------------
        
            - | obj:pandas.DataFrame := The underlying
                `pandas.DataFrame` object
            
            - shape:tuple[int,...] := The shape of the object
            
            - | dims:DIMS := Dimensions of the object. A `numpy` vector
                of labels of the dimensions / axes of the object.The
                default names are 'dim_{i}' where i the integer indexer
                of the axis
            
            - | coords:COORDS := The coordinates of the object. Is a
                dictionary of strings, which are axes names (the
                same as those of `dims`) mapped to numpy vectors of
                labels. These are the labels of the 'steps' in each
                axis. If the object has index and columns, these will be
                used, else defaults are used (as enumerated
                integers)
            
            - | rank:int := The structures' rank i.e. the number of axes
            
            - dtype:np.dtype := The data type of the structure
            
            - | missing_nan_flag:Optional[bool] = None := Flag for
                existence of missing values. Should be set by the
                public interface class
            
        Object Methods:
        ---------------
        
            - | isna()->DataFrameStructure := Return a boolean
                structure, of the same class and shape as the
                original, whose elements are booleans indicating if
                the corresponding element is nan or not.
            
            - | any(axis:Optional[int]=None)-> Union[bool,
                DataFrameStructure] := If axis is `None` reduce via
                element wise or the entire array. Else reduce over
                the axis specified
            
            - | all(axis:Optional[int]=None)-> Union[bool,
                DataFrameStructure] := If axis is `None` reduce via
                element wise and the entire array. Else reduce over
                the axis specified
            
            - | transpose(axis:Optional[tuple[int,...]]
                )->DataFrameStructure := Return a transposed
                structure. If axes is `None` reverses the
                dimensions. If provided, `axis` should be a
                permutation of the objects' axes (as a tuple),
                defining the transposition
            
            - | iterrows()->(str, DataFrameStructure) := Returns an
                iterator over the zeroth axis of the structure.
                Yields tuples of coordinates to substructures.
                Loosely equivalent to:
                
                .. code-block::
                
                    def iterrows(X:pandas.DataFrame):
                        for i in range(X.shape[0]):
                            yield (X.index[i], X.iloc[i,...])
            
            - | itercolumns()->(str, DataFrameStructure) := Returns
                an iterator over the first axis of the structure.
                Yields tuples of coordinates to substructures.
                Loosely equivalent to:
                
                .. code-block::
                
                    def iterrows(X:pandas.DataFrame):
                        for i in range(X.shape[1]):
                            yield (X.columns[i], X.iloc[:,i,...])
            
            - | cast(dtype:numpy.dtype)->DataArrayStructure := Casts
                the structure to the specified data type. Returns a
                fresh DataStrcture object
            
            - | unique(axis:Optional[int]=None)->DataFrameStructure
                := Return a unique values in the structure. If axis is provided, unique values will over the specified axis are returned. Else unique values over the entire structure are returned
            
            - | mean(axis:Optional[int]=None)->DataFrameStructure :=
                Return the mean along the specified axis, or over
                the entire structure (if `axis=None`)
        '''
    
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
        self._dims = np.asarray([
            "dim_0", "dim_1"]) if dims is None else dims
        self._coords = dict(dim_0 = np.asarray(self.obj.index), 
                            dim_1 =np.asarray(self.obj.columns)
                            ) if coords is None else coords
        self._rank:int = 2
        self._dtype = obj.values.dtype if dtype is None else dtype
        self._missing_nan_flag:Optional[bool] = None
        
    def __getitem__(self, obj:Union[str, int, Iterable]
                    )->Union[DataFrameStructure, np.ndarray]:
        r'''
            Index slicing for CommonDataStructure objects. Index or 
            label based slicing is supported in arbitary combinations.
            DataStructure objects can be sliced with (nearly) any
            combination of int, str, slice, list, None, Ellipsis.
            Note label slicing is supported, however the `step` argument
            must be blank or an integer, not a label. Example usage:
            .. code-block::
                # Pseudo-code
                obj = DataStructure()
                obj[5]
                obj[:5]
                # Mix and match labelling
                obj["sample_0", [0,6], 0:6:3,...]
                # Legal
                obj["sample_0":"sample_10":1]
                # Illegal
                obj["sample_0":"sample_10":"group"]
                
            Returns:
                - numpy.NDArray := If boolean indexing or an exact element
                is selected i.e. `obj[1,0,1]` or `obj[obj.values>5]`
                
                - DataStructure := Of the same type as the original. Note
                is all cases where the resulting structure would have been
                1-D or 0-D, a 2-D array is returned i.e. instead of
                (9,) (1, 9) is returned
        '''
        lookup = lambda dim ,e: int(np.where(
                        self.coords[dim]== e
                        )[0]) if isinstance(e, str) else e
        nobj:tuple = obj if isinstance(
            obj, tuple) else (obj, )
        isarray_indexing:bool = isinstance(nobj[0], np.ndarray)
        boolean_indexing:bool = isarray_indexing and nobj[0].dtype=='bool'
        if boolean_indexing:
            return self.obj.values[obj]
        if Ellipsis not in nobj and len(nobj)<self.rank:
            nobj = tuple(list(nobj)+[...])
        try:
            collected:list = []
            for dim, e in zip(self.dims ,nobj):
                if e is Ellipsis:
                    collected.append(Ellipsis)
                    break
                elif isinstance(e, (str, int)):
                    i = lookup(dim, e)
                    collected.append(i)
                elif isinstance(e, slice):
                    if not (isinstance(e.step, int) or e.step is None):
                        raise IndexError((
                            "Step parameter of a slice indexer cannot be"
                            "be label. Expecte None or an integer greater"
                            f"than 0, received {e.step} of type "
                            f"{type(e.step)} instead"
                        ))
                    collected.append(slice(
                        lookup(dim, e.start),
                        lookup(dim, e.stop),
                        e.step # Cannot be a label. Raise
                    ))
                elif isinstance(e, list):
                    subcollected:list[int] = []
                    for elem in e:
                        if isinstance(elem, (int, str)):
                            subcollected.append(lookup(dim, elem)
                                )
                        else:
                            raise IndexError((
                                "Only lists of integers and labels are "
                                f"valid indexers. Received {elem} of type "
                                f"{type(elem)}"
                            ))
                    collected.append(subcollected)
        except TypeError:
            raise IndexError(f"Indexer {e} is invalid")
        nobj = tuple(collected)
        exact_match:bool = not any([
            isinstance(e, (slice, list, tuple)) or e is Ellipsis for e in nobj
            ])
        # To have an exact match, we must receive an object of length exactly
        # equal to the number of axis, no elements of which are Ellipsis, or slices
        # and if it has structures i.e. lists/tuples, they should have a length of 1
        
        if not exact_match:
            ncoords = self._slice_coords(nobj)
            ndims = np.asarray([k for k in ncoords.keys()])
            return DataFrameStructure(
                self._obj.iloc[nobj],
                dims = ndims, coords=ncoords
            )
        else:
            return self._obj.iloc[nobj]
    
    def __eq__(self, obj):
        raw = self._obj == obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ne__(self, obj):
        raw = self._obj != obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __lt__(self, obj):
        raw = self._obj < obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __gt__(self, obj):
        raw = self._obj > obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ge__(self, obj):
        raw = self._obj >= obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __le__(self, obj):
        raw = self._obj <= obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataFrameStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
            
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
                pd.DataFrame(self.obj.any(axis=1).values[None, :], #type:ignore
                             index = ["0"],
                             columns = self.coords['dim_0']
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
                pd.DataFrame(self.obj.all(axis=1).values[None, :], #type:ignore
                             columns = self.coords['dim_0'],
                             index = ["0"]
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
        
    def unique(self, axis:Optional[int] = None):
        if axis is not None and axis not in (0,1):
            raise ValueError(("axis argument must be one of None, 0 or "
                              f"1. Received {axis} instead"))
        if axis is None:
            yield (None, np.unique(self._obj.values))
        else:
            ob = self._obj.values if axis==0 else self._obj.values.T
            kz = list(self.coords.keys())[axis]
            for crd, substruct in zip(self.coords[kz], ob):
                yield (crd, np.unique(substruct))
    
    def mean(self, axis:Optional[int] = None, keepdims:bool=True,
             skipna:bool=True):
        r'''
            Compute the arithmetic mean along the specified axis
            
            Args:
            -----
            
                - axis:Optional[int]=None := The axis to compute the mean
                over. If `None` (default), computes the mean over the entire
                `DataFrame`. Values are `0` (mean over the rows), `1` (mean
                over the columns) and `None` (mean of the entire array)
                
                - skipna:bool=True := If `True` ignores `NaN` values in the
                dataframe. Else returns `NaN` for coordinates with at least one
                `NaN`
                
                - keepdims:bool=True := If `True` the axis over which the mean
                is computed, is kept in the result with a single coordinate
                named 'sum' (default), making the result correctly broadcastable
                against the original. Otherwise, the result axis is reduced. Since
                `ArrayStructure` object cannot be reduced past 2D the arguement
                if effectively ignored and always `True` for DataFrames
                
            Returns:
            ---------
            
                - mean:float := The mean of the entire dataframe (`axis=None`)
                
                - means:DataFrameStructure := A DataFrame with a single row
                of means along the specified axis
                
            Raises:
            -------
            
                - ValueError := If `axis` is not `None`, `1` or `0`
        '''
        if axis is None:
            return self._obj.mean().mean()
        elif axis==1:
            vals = self._obj.mean(axis=axis, skipna=skipna).to_frame().T
            return DataFrameStructure(
                vals, dims=self.dims, coords= {
                    self.dims[0] : self.coords[self.dims[0]],
                    self.dims[1] : np.asarray(["sum"])
                } )
        elif axis==0:
            vals = self._obj.mean(axis=axis, skipna=skipna).to_frame().T
            return DataFrameStructure(
                vals, dims=self.dims, coords= {
                    self.dims[1] : self.coords[self.dims[1]],
                    self.dims[0] : np.asarray(["sum"])
                } )
            
        else:
            raise ValueError((
                "Illegal argument for axis. Expected on of `0` (rows)"
                "`1` (columns) or `None` (along the entire DataFrame). "
                f"Received axis={axis} instead"
            ))
            

class DataArrayStructure(DataStructure, UtilityMixin):
    r'''
            Wrapper class around xarray.DataArray implementing the
            common interface.
            
            Like all implementations this class implements a common,
            standardized interface for acceptable tensor data
            structures, as defined by the `DataStructure` abstract base
            class.
            
            Class Properties:
            -----------------
            
                - | accepted_inputs:set := Set of input structured that
                    can be converted into `xarray.DataArray`. Acceptable
                    classes for the `obj` attribute
            
            Object Properties:
            ------------------
            
                - | obj:xarray.DataArray := The underlying
                    `xarray.DataArray` object
                
                - shape:tuple[int,...] := The shape of the object
                
                - | dims:DIMS := Dimensions of the object. A `numpy`
                    vector of labels of the dimensions / axes of the
                    object. If the object has dimentions these will be
                    used else defaults are generated automatically. The
                    default names are 'dim_{i}' where i the integer
                    indexer of the axis
                
                - | coords:COORDS := The coordinates of the object. Is a
                    dictionary of strings, which are axes names (the
                    same as those of `dims`) mapped to numpy vectors of
                    labels. These are the labels of the 'steps' in each
                    axis. If the object has coordinates, these will be
                    used, else defaults are used (as enumerated
                    integers)
                
                - rank:int := The structures' rank i.e. the number of
                  axes
                
                - dtype:np.dtype := The data type of the structure
                
                - | missing_nan_flag:Optional[bool] = None := Flag for
                    existence of missing values. Should be set by the
                    public interface class
                
            Object Methods:
            ---------------
            
                - | isna()->DataArrayStructure := Return a boolean
                    structure, of the same class and shape as the
                    original, whose elements are booleans indicating if
                    the corresponding element is nan or not. Unlike
                    `numpy.isnan` will work on objects but not strings
                
                - | any(axis:Optional[int]=None)-> Union[bool,
                    DataArrayStructure] := If axis is `None` reduce via
                    element wise or the entire array. Else reduce over
                    the axis specified
                
                - | all(axis:Optional[int]=None)-> Union[bool,
                    DataArrayStructure] := If axis is `None` reduce via
                    element wise and the entire array. Else reduce over
                    the axis specified
                
                - | transpose(axis:Optional[tuple[int,...]]
                    )->DataArrayStructure := Return a transposed
                    structure. If axes is `None` reverses the
                    dimentions. If provided, `axis` should be a
                    permutation of the objects' axes (as a tuple),
                    defining the transposition
                
                - | iterrows()->(str, DataArrayStructure) := Returns an
                    iterator over the zeroth axis of the structure.
                    Yields tuples of coordinates to substructures.
                    Loosely equivalent to:
                    
                    .. code-block::
                    
                        def iterrows(X:xarray.DataArray):
                            for i in range(X.shape[0]):
                                yield (X.coords[i], X[i,...])
                
                - | itercolumns()->(str, DataArrayStructure) := Returns
                    an iterator over the first axis of the structure.
                    Yields tuples of coordinates to substructures.
                    Loosely equivalent to:
                    
                    .. code-block::
                    
                        def iterrows(X:xarray.DataArray):
                            for i in range(X.shape[1]):
                                yield (X.coords[i], X[:,i,...])
                
                - | cast(dtype:numpy.dtype)->DataArrayStructure := Casts
                    the structure to the specified data type. Returns a
                    fresh DataStrcture object
                
                - | unique(axis:Optional[int]=None)->DataArrayStrcture
                    := Return a unique values in the structure. If axis is provided, unique values will over the specified axis are returned. Else unique values over the entire structure are returned
                
                - | mean(axis:Optional[int]=None)->DataArrayStructure :=
                    Return the mean along the specified axis, or over
                    the entire structure (if `axis=None`)
        '''
    
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
            icoords:COORDS = dict()
            if not isinstance(obj.index, np.ndarray):
                icoords["dim_0"] = np.asarray(obj.index)
            else:
                icoords['dim_0'] = obj.index
            if not isinstance(obj.columns, np.ndarray):
                icoords['dim_1'] = np.asarray(obj.columns)
            else:
                icoords['dim_1'] = obj.columns
        elif _t == pd.Series:
            self._obj = xr.DataArray(obj.values[None,:], coords =dict(
                dim_0 = ["0"], dim_1 = obj.index 
            ))
            icoords:COORDS = dict(
                dim_0 = np.asarray(["0"]),
                dim_1 = np.asarray(obj.index)
            )
            self._dtype = dtype if dtype is not None else \
                obj.values.dtype
            
        elif _t == np.ndarray:
            if len(obj.shape)>=2:
                icoords:COORDS = {
                    f"dim_{i}": np.asarray(range(axis)) for i, axis in \
                    enumerate(obj.shape)
                }
                self._obj = xr.DataArray(obj, coords = icoords)
            else:
                icoords = dict(
                    dim_0=np.asarray([0]), dim_1 = np.asarray(
                        range(obj.shape[0])
                    )
                )
                self._obj = xr.DataArray(obj[None, :], coords = icoords)
            self._dtype = dtype if dtype is not None else obj.dtype
        else:
            idims = dims if dims is not None else obj.dims
            icoords = coords if coords is not None else {
                k:v.values for k,v in obj.coords.items()
                }
            if icoords == dict():
                icoords = {
                    k: np.asarray([
                        e for e in range(obj.shape[i])
                        ]) for i,k in enumerate(idims)
                }
                obj=obj.assign_coords(icoords)
            
            self._obj:xr.DataArray = obj
            self._dtype = obj.dtype
        idims:DIMS = np.asarray([e for e in icoords.keys()])
        self._shape:SHAPE = self._obj.shape
        if dims is None:
            self._dims = idims
        else:
            if len(dims)!=len(self._obj.shape):
                raise ValueError((
                    "When provided, the length of dims must match the number "
                    "of dimentions on the object. Object has axii "
                    f"{self.obj.shape} but saw {dims} instead"
                ))
            else:
                self._dims = dims
        if coords is not None:
            if len(coords)!=len(self._obj.shape):
                raise ValueError((
                    "When provided, the length of coords must match the number "
                    "of dimentions on the object. Object has axii "
                    f"{self.obj.shape} but saw {len(coords)} instead"
                ))
            else:
                self._coords=coords
        else:
            self._coords = icoords
        self._rank:int = len(self._coords)
        self._missing_nan_flag:Optional[bool] = None
        
    def __eq__(self, obj):
        raw = self._obj == obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ne__(self, obj):
        raw = self._obj != obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __lt__(self, obj):
        raw = self._obj < obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __gt__(self, obj):
        raw = self._obj > obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __ge__(self, obj):
        raw = self._obj >= obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
    
    def __le__(self, obj):
        raw = self._obj <= obj
        if isinstance(raw, bool):
            return raw
        else:
            return DataArrayStructure(
                raw,
                dims= self.dims, 
                coords = self.coords
                )
        
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
        core_obj if len(core_obj.shape)>=2 else core_obj[None,:],
                                  dims=ndims, coords=ncoords
                                  )
    
    def isna(self):
        return DataArrayStructure( 
                                  super().__isna__(self._obj),
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
    def __getitem__(self, obj:Union[str, int, Iterable]
                    )->Union[DataArrayStructure, np.ndarray]:
        r'''
            Index slicing for CommonDataStructure objects. Index or 
            label based slicing is supported in arbitary combinations.
            DataStructure objects can be sliced with (nearly) any
            combination of int, str, slice, list, None, Ellipsis.
            Note label slicing is supported, however the `step` argument
            must be blank or an integer, not a label. Example usage:
            .. code-block::
                # Pseudo-code
                obj = DataStructure()
                obj[5]
                obj[:5]
                # Mix and match labelling
                obj["sample_0", [0,6], 0:6:3,...]
                # Legal
                obj["sample_0":"sample_10":1]
                # Illegal
                obj["sample_0":"sample_10":"group"]
                
            Returns:
                - numpy.NDArray := If boolean indexing or an exact element
                is selected i.e. `obj[1,0,1]` or `obj[obj.values>5]`
                
                - DataStructure := Of the same type as the original. Note
                is all cases where the resulting structure would have been
                1-D or 0-D, a 2-D array is returned i.e. instead of
                (9,) (1, 9) is returned
        '''
        lookup = lambda dim ,e: int(np.where(
                        self.coords[dim]== e
                        )[0]) if isinstance(e, str) else e
        nobj:tuple = obj if isinstance(
            obj, tuple) else (obj, )
        isarray_indexing:bool = isinstance(nobj[0], np.ndarray)
        boolean_indexing:bool = isarray_indexing and nobj[0].dtype=='bool'
        if boolean_indexing:
            return self._obj.values[obj]
        if Ellipsis not in nobj and len(nobj)<self.rank:
            nobj = tuple(list(nobj)+[...])
        try:
            collected:list = []
            for dim, e in zip(self.dims ,nobj):
                if e is Ellipsis:
                    collected.append(Ellipsis)
                    break
                elif isinstance(e, (str, int)):
                    i = lookup(dim, e)
                    collected.append(i)
                elif isinstance(e, slice):
                    if not (isinstance(e.step, int) or e.step is None):
                        raise IndexError((
                            "Step parameter of a slice indexer cannot be"
                            "be label. Expecte None or an integer greater"
                            f"than 0, received {e.step} of type "
                            f"{type(e.step)} instead"
                        ))
                    collected.append(slice(
                        lookup(dim, e.start),
                        lookup(dim, e.stop),
                        e.step # Cannot be a label. Raise
                    ))
                elif isinstance(e, list):
                    subcollected:list[int] = []
                    for elem in e:
                        if isinstance(elem, (int, str)):
                            subcollected.append(lookup(dim, elem)
                                )
                        else:
                            raise IndexError((
                                "Only lists of integers and labels are "
                                f"valid indexers. Received {elem} of type "
                                f"{type(elem)}"
                            ))
                    collected.append(subcollected)
        except TypeError:
            raise IndexError(f"Indexer {e} is invalid")
        nobj = tuple(collected)
        exact_match:bool = not any([
            isinstance(e, (slice, list, tuple)) or e is Ellipsis for e in nobj
            ])
        if not exact_match:
            ncoords = self._slice_coords(nobj)
            ndims = np.asarray([k for k in ncoords.keys()])
            return DataArrayStructure(
                self._obj.values[nobj],
                dims = ndims, coords=ncoords
            )
        else:
            return self._obj.values[nobj]
        
    def unique(self, axis:Optional[int]=None):
        r''' 
            Return unique values of the NDArrayStructure as Generator
            of length 2 tuples. 
            
            When axis is None, the generator yields a single tuple of (None, vals) where vals are all the unique values in the array. When axis is specified, the Generator iterates over the specified axis yielding tuples of (label, unique_values)
        '''
        if axis is not None and axis not in list(range(len(self.shape))):
            raise ValueError((
                "axis argument must be a positive integer no larger "
                "than the number of axii on the object, or None. Expected "
                f"axis either None or in {list(range(len(self.shape)))}."
                f" Saw {axis} instead"
            ))
        elif axis is None:
            yield (None, np.unique(self._obj.values))
        else:
            crds_key = list(self.coords.keys())[axis]
            axii = [axis] + [e for e in range(len(self.shape)) if e!=axis]
            for crd, subtensor in zip(
                 self.coords[crds_key],
                 np.transpose(
                    self._obj.values, axii)
                ):
                yield (crd, np.unique(subtensor))
                
    def mean(self, axis:Optional[int] = None, keepdims:bool=True,
             skipna:bool=True):
        raw_mean = super().__mean__(
            self._obj.values, axis=axis, keepdims=keepdims, skipna=skipna)
        if axis is None:
            return raw_mean
        else:
            return DataArrayStructure(
                raw_mean.structure, coords = raw_mean.coords,
                dims = raw_mean.dims
            )
            



class DataStructureInterface(ABC):
    r'''
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
            
            - __gettitem__(indexer) := Return the values specified by
            the `indexer`. Mix and matching label and integer based
            indexing is supported. If a slice is provided, the start and
            stop arguments can be labels, but the step argument must be
            an integer
            
            - unique(axis=None) := Return all the unique values in the
            tensor as a Generator that yields length 2 tuples. If 
            axis is None the Generator yields a single tuple of the form
            (None, values), where values is a vector of unique values.
            If axis is provided, the generator iterates over the specified
            dimention, yielding tuples of the form (label, values) where
            values is a vector of unique values for the flattened subtensor.
            
            - mean(axis=None, keepdims=True, skipna=True) := Compute the
            arithmetic mean along the specified axis (or the entire structure
            if `None` - default). If `keepdims=True` the specified dimention
            is kept in the result with a single coordinate named 'sum', making
            the result correctly broadcastable to the original. Else the 
            dimention is reduced. If `skipna=True` `NaN` values are ignored
            else, all coordinates with at least one `NaN` will return `NaN`
    
            - ops := Elementwise operations '>', '<', '>=', '<=', '==', 
            '!=' are delegated to the underlying library but a wrapped
            object is returned for 'pointwise' operations, i.e. obj==5,
            otherwise a boolean is returned
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
    
    @abstractmethod
    def __getitem__(self, obj)->DataStructureInterface:
        raise NotImplementedError()
    
    @abstractmethod
    def __eq__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def __ne__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def __lt__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def __le__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def __ge__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def __gt__(self, obj)->Union[bool, DataStructureInterface]:
        raise NotImplementedError()
    
    @abstractmethod
    def mean(self, axis: Optional[int]=None, keepdims: bool=True,
             skipna: bool= False)->None:
        raise NotImplementedError()


@dataclass(kw_only=True)
class CommonDataStructureInterface(DataStructureInterface):
    r'''
        Core interface for supported data structures. Should be the only
        interface provided
        
        Object Attributes:
        ------------------
            
            - | implementor:Type[DataStructure] := Class reference to the type
                of implementor
                
                .. caution::
                    This attribute is deprecated and scheduled for removal
            
        
        Object Properties:
        ------------------
            - | data_structure:DataStructure := The core data structure
                implementation
            
            - rank:int := The structures' rank (number of axes)
            
            - shape:tuple[int] := The shape of the structure
            
            - dims:numpy.ndarray := Labels for the axes of the structure
            
            - | coords:dict[str, numpy.ndarray] := Labels for the coordinates
                of the structure. Stored as dictionary mapping axes labels to
                numpy arrays of labels in that axes. Keys should match
                elements of the `dims` property
            
            - | values:numpy.ndarray := The underlying numpy array structure.
                Usefull for unpacking the structure for other software (like
                `pymc`)
         
        Object Methods:
        ----------------
        
            Methods exposed by the tensor. Except where methods reduce
            the entire structure to 0D, all should return another
            `CommonDataStructureInterface` permitting method chaining.
            Whenever a structure would be reduced to 1D, a 2D row-vector
            structure will be returned instead
        
            - | transpose(axis:Optional[AXES_PERMUTATION] = None) :=
                Return a tranposed version of the object. Signature is
                the same as numpy and must return the same default.
                Should always return the same type of object. The T
                attribute is an alias for this method
            
            - | isna(axis:Optional[int] = None) := Elementwise `isnan`.
                Should default to returning the a boolean tensor of the
                same shape as the original tensor. When `axis` is
                provided this is equivalent to an `any` operation over
                this axis. The axis is preseved in the return.
                
                .. note::
                
                    This does not actually use `numpy.isnan` internally.
                    It extends its functionaly by allowing nan checks on
                    `object` type arrays.
                    
                .. danger::
                
                    Will not work with strings as missing nan values
                    automatically converted into strings
            
            - | any(axis:Optional[int] = None) := When `axis=None`
                perform `any` over the entire array and return a
                boolean. Otherwise perform the operation over the
                specified axis, preserving the axis
            
            - | all(axis:Optional[int] = None) := When `axis=None`
                perform `all` over the entire array and return a
                boolean. Otherwise perform the operation over the
                specified axis, preserving the axis
            
            - | iterrows() := Iterate over the first axis of the
                structure Similar to `pandas.DataFrame.iterrows()`
            
            - | itercolumns() := Iterate over the second axis of the
                structure. Similar to `pandas.DataFrame.itercolumns`
            
            - | unique(axis:Optional[int] = None) := Return unique
                values of the structure. Returns a Generator object that
                yields  length 2 tuple of the general form
                (Optional[label:str],  array). The first element of the
                tuple is either None or an array of unique elements.
                When axis is None, (default) the Generator yields only a
                single element, whose first element is None, and whose
                other element is an array of all unique values in the
                array. When axis is provided an integer, the resulting
                Generator, loops over the specified axis, yielding
                tuples of the label in the current iteration (coordinate
                of the  specified axis) and numpy arrays of all unique
                values in the subtensor (as a vector)
            
            - | __getitem__(obj) := DataStructure indexing. All
                conventional indexing options are supported, including
                slicing with labels and mixed label/index based indexing
                and selecting. The `step` argument must be an integer
                (or None) all others can be any mix of label and index
                based indexers. For example:
            
                    .. code-block::
                    
                        obj[0,0,0] obj[:5:2, "var1",0]
                        obj['sample_0':'sample_10':2, 5,...]
                        obj['sample_5',...] # Illegal - step must be an
                        integer obj[0:15:"sample",...]
                
            - | mean(axis=None, keepdims=True, skipna=True) := Compute
                the arithmetic mean along the specified axis (or the
                entire structure if `None` - default). If
                `keepdims=True` the specified dimention is kept in the
                result with a single coordinate named 'sum', making the
                result correctly broadcastable to the original. Else the
                dimention is reduced. If `skipna=True` `NaN` values are
                ignored else, all coordinates with at least one `NaN`
                will return `NaN`
                
            - | ops := Basic operators are supported and generally
                delegated to the underlying library '==', '>=', '!=',
                '>', '<', '<='
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
        
    def __getitem__(self, obj:Any)->CommonDataStructureInterface:
        return CommonDataStructureInterface(
            _data_structure = self.data_structure[obj]
        )
        
    def unique(self, axis:Optional[int] = None, **kwargs):
        return self._data_structure.unique(axis=axis)
    
    def __eq__(self, obj)->DataStructureInterface:
        raw = self._data_structure == obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            ) 
    
    def __ne__(self, obj)->DataStructureInterface:
        raw = self._data_structure != obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            ) 
            
    def __lt__(self, obj)->DataStructureInterface:
        raw = self._data_structure < obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            ) 
            
    def __le__(self, obj)->DataStructureInterface:
        raw = self._data_structure <= obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            ) 
            
    def __gt__(self, obj)->DataStructureInterface:
        raw = self._data_structure > obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            ) 
            
    def __ge__(self, obj)->DataStructureInterface:
        raw = self._data_structure >= obj
        if isinstance(raw, bool):
            return raw
        else:
            return CommonDataStructureInterface(
                _data_structure = raw
            )
            
    def mean(self, axis: Optional[int]=None, skipna: bool=True,
             keepdims: bool=True)->Union[float, 
                                         CommonDataStructureInterface]:
            r'''
                Compute the arithmetic mean along the specified axis
                
                Args:
                ------
                
                    - | axis:Optional[int]=None := The axis along which
                        to compute the mean. If `None` (default) returns
                        the mean along the entire structure
                    
                    - | skipna:bool=True := If `True` (default) `NaN`
                        values will be ignored, else every coordinate
                        along the specified axis with at least one `NaN`
                        will return `NaN`.
                    
                    - | keepdims:bool=True := If `True` (default), the
                        axis along which the mean is computed is kept in
                        the result with a single coordinate named 'sum',
                        making the result correctly broadcastable agaist
                        the original. Else reduce the dimension in the
                        result. This argument is ignored if axis is
                        `None`
                    
                Returns:
                --------
                
                    - | mean:float := The mean of the entire structure
                        (if `axis=None`)
                    
                    - | means:CommonDataStructureInterface := A new
                        structure of means. If `keepdims=False` would
                        reduce the structure below 2D, a 2D structure is
                        returned instead (equivalent to `keepdims=True`)
            '''
            if axis is None:
                return self._data_structure.mean(axis=axis, skipna=skipna,
                                            keepdims=keepdims)
            else:
                return CommonDataStructureInterface(
                    _data_structure = self._data_structure.mean(
                        axis=axis, skipna=skipna, keepdims=keepdims) 
                )
            
    

class NANHandler(ABC):
    r'''
        Abstract Base Class for missing value handlers
    '''
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        raise NotImplementedError()

@dataclass
class ImputeMissingNAN(NANHandler):
    r'''
        Core class from missing data imputation strategy. Currently not
        implemented and will raise
    '''
    
    def __call__(self, data: DataStructureInterface
                 )->DataStructureInterface:
        raise NotImplementedError(("Data Imputation not yet "
                                   "implemented"))

@dataclass
class ExcludeMissingNAN(NANHandler):
    r'''
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
        
        not_nan = np.logical_not(indices.T().values()[:,0])
        
        clean_data= data._data_structure._obj[not_nan]
        
        self.new_coords = copy(data.coords())
        dim0 = data.dims()[0]
        self.new_coords[dim0] = data.coords()[dim0][not_nan]
        self.new_dims = data.dims()
        obj = self.constructor(
                clean_data,
                coords=self.new_coords,
                dims = self.new_dims
            )
        obj.missing_nan_flag = True
        this = CommonDataStructureInterface(
            _data_structure = obj
        )
        return this

@dataclass
class IgnoreMissingNAN(NANHandler):
    r'''
        Identity strategy for nan handling the does nothing. 
        
        Only included for completeness' sake. Returns the object 
        unmodified
        
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
    r'''
        Composite for missing values handling. Defines the external interface
        
        Object Properties:
        --------------------
        
            - nan_handler:NANHandler := The nan handling strategy to
              apply
            
        Object Methods:
        ------------------
        
            - __call__(data:DataStructureInterface))->DataStructureInterace
              :=
                Delegate missing value handling to the handler and return
                the results
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
    r'''
        Abstract base class for Data Processors
    '''
    
    @abstractmethod
    def __call__(self, data: DataStructure)->DataStructureInterface:
        raise NotImplementedError()

@dataclass(kw_only = True)
class CommonDataProcessor(DataProcessor):
    r'''
        Common use-case data pre processor.
        
        Will handle the following data preprocessing tasks:
        
            - | converting the data structure to a common internal
                interface

            - handling of missing values
            
            - casting to data type (Optional)
            
            - validate data types (Incomplete)
        
        Can be subclassed for extended functionality or overriden.
        
        
        Object Attributes:
        --------------------
        
            - | nan_handler:NANHandlerContext := The missing values
                handler. Optional. Defaults to ExcludeMissingNAN.
                Initially a ref to the context class, will be replaced
                by a instance of that class.
            
            - | cast:Optional[np.dtype]=None := Attempt to forcefully
                cast all inputs to the specified type. Optional.
                Defaults to `np.float32`. Setting this to `None` will
                disable typecasting
            
            - | type_spec := Schema to validate. Not implemented and
                will be ignored
            
            - | casting_kwargs:dict={} := Keyword arguements to be
                forwarded to the underlying typecaster. See numpy for
                details. Defaults to an empty dict.
                
            .. danger::
            
                Typecasting is not fully implemented due to the
                limitations of numpy arrays (they are homogenuous
                structures, whereas pandas DataFrames are not). Use this
                option only to cast the entire structure to a certain
                dtype
            
        Object Methods:
        ----------------
        
            - | __call__(data:InputData)->CommonDataStructureInterface
                := Preprocess the data according the set options and
                return the result.
            
    '''
    
    nan_handler_context:NANHandlingContext = NANHandlingContext(
        _nan_strategy = ExcludeMissingNAN
    )
    cast:Optional[np.dtype] = None
    type_spec:Any = None
    casting_kwargs:dict = field(default_factory = dict)
    
    def _convert_structure(self, data: InputData
                           )->DataStructureInterface:
        r'''
            Convert the input structure to a common interface by
            wrapping it in the appropriate implementation class
            
            Args:
            -----
            
                - data:InputData := The data to process
                
            Returns:
            --------
            
                - common:CommonDataStructureInterface := The object,
                  converted
                to a common interface
                
            Raises:
            -------
            
                - RuntimeError := When unable to identify the data type.
                  This
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
        return data.isna().any() #type:ignore
    
    def __call__(self, data: InputData)->DataStructureInterface:
        r'''
            Preprocess the object according to set options and return
            the results
            
            Args:
            -----
            
                data:InputData := The data to process
                
            Returns:
            ---------
            
                - | processed:CommonDataStructureInterface := The
                    processed data structure
        '''
        from warnings import warn
        _data = self._convert_structure(data)
        _data.data_structure._missing_nan_flag = self._detect_missing_nan(_data)
        if _data.data_structure._missing_nan_flag:
            warn((
                "Input data contains missing or invalid values. These "
                "will be handled by the currently selected strategy "
                f"{self.nan_handler_context._nan_strategy}. Specify the"
                "`nan_handling` argument in `Data` if this is not the "
                "desired behavior"
            ))
        _data = self._handle_nan(_data)
        _data = self._cast_data(_data)
        return _data
    


@dataclass(kw_only=True)
class DataProcessingDirector:
    r'''
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
    r'''
        Container for model data with optional preprocessing
        functionality.
        
        Class Attributes:
        ------------------
        
            - | nan_handlers:set[str]=['exlude', 'impute', 'ignore'] := Valid
                strategies for missing value handling
            
            - | input_types:set[str]=['ndarray', 'DataFrame', 'DataArray'] :=
                Supported input data structures
            
        Object Attributes:
        -------------------
            
            - | nan_handling:str='exclude' := The missing data handling
                strategy. Has to be one of Data.nan_handlers. Optional.
                Defaults to 'exclude' and discards all axis=0 coordinates with
                missing values (i.e rows).
            
            - | cast:Any := A data type to force-cast the data to. Optional.
                Defaults to `numpy.float32`. Set to `None` to disable casting
            
            - | type_spec:dict={} := Dictionary specification data validation
                across the second dimention. Keys should be coordinates
                (labels) along the second axis(=1) and values should be valid
                numpy dtypes. Currently ignored
            
            - | casting_kwargs:dict={} := Optional keyword arguments to be
                forwarded to the type caster. Optional. Defaults to an empty
                dict. See the `numpy` documentation for further details.
                Ignored if `cast=None`
            
            - | processor:Type[DataProcessor]=CommonDataProcessor := The
                processor to be used for data processing. Optional and
                defaults to the generic data processor. Can be overriden to
                customized with a user specified processor that subclasses
                `DataProcessor` or `CommonDataProcessor`
            
            - | process_director:Optional[DataProcessDirector] := The director
                for data processing. Optional
            
            - | nan_handler:Optional[NANHandler]=None := The class that
                handles missing values. None only when unset
                
        Object Methods:
        ----------------
        
            - | __call__(data:InputData) := Process the data and return the
                result
    
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
        r'''
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
        