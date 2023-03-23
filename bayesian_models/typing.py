from typing import Any, Union, Optional
from numpy.typing import NDArray
from pandas import DataFrame
from xarray import DataArray


ndarray = NDArray
InputData = Union[DataArray, DataFrame, NDArray]
SHAPE = tuple[int, ...]
DIMS = tuple[str, ...]
COORDS = dict[str,ndarray ]
AXIS_PERMUTATION = Optional[Union[list[int], tuple[int, ...]]]

