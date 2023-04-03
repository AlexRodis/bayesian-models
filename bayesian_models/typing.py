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
#   This module contains type aliases

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

