import unittest
from bayesian_models.data import NDArrayStructure, DataFrameStructure
from bayesian_models.data import DataArrayStructure, NANHandlingContext
from bayesian_models.data import CommonDataStructureInterface, Data
from bayesian_models.data import CommonDataProcessor, ExcludeMissingNAN
from bayesian_models.data import ImputeMissingNAN, IgnoreMissingNAN
from bayesian_models.data import DataProcessingDirector
from typing import Any, Optional, Union, Literal
import pandas
import xarray
import numpy
import numpy as np
import pandas as pd
import xarray as xr


dict_arr_compare = lambda one, other : all([(v1 == v2).all() \
            for (k1,v1), (k2,v2) in zip(one.items(),
                                        other.items()
                                        )
         ])


class TestDataModule(unittest.TestCase):
    
    
    def setUp(self) -> None:
        
        A = numpy.random.rand(90,9,3)
        B = pandas.DataFrame(A[:,:,0], columns = [
            f"var{i}" for i in range(A.shape[1])], 
                             index = range(A.shape[0]))
        C = xarray.DataArray(A, 
                             coords={f"dim{i}":[
                                 j for j in range(A.shape[i])] for i \
                                     in range(len(A.shape))
                                    })
        self.A = A
        self.B = B
        self.C = C
    
    
    def test_pd_series_input(self):
        ser_row = self.B.iloc[0,:]
        ser_col = self.B.iloc[:,0]
        serA = DataFrameStructure(ser_row)
        serB = DataFrameStructure(ser_col)
        c = isinstance(serA.obj, pandas.DataFrame)
        self.assertTrue(
            all([
                isinstance(serA.obj, pandas.DataFrame),
                isinstance(serB.obj, pandas.DataFrame),
                serA.shape == (1, self.B.shape[1]),
                serB.shape == (1, self.B.shape[0])
                ])
        )
        
    def test_np_all(self):
        rank3 = NDArrayStructure(self.A)
        rank2 = NDArrayStructure(self.A[:,:,0])
        rank1 = NDArrayStructure(self.A[:,0,0])
        rank3_axis2 = numpy.ones_like(self.A[:,:,0], dtype=bool)
        rank3_axis1 = numpy.ones_like(self.A[:,0,:], dtype = bool)
        rank3_axis0 = numpy.ones_like(self.A[0,:,:], dtype = bool)
        cond1 = rank3.all()
        cond4= rank3.all(axis=2).shape == rank3_axis2.shape
        cond2 =  rank3.all(axis=1).shape == rank3_axis1.shape
        cond3 = rank3.all(axis=0).shape == rank3_axis0.shape
        cond5 = rank2.all()
        cond6 = rank1.all()
        cond7 = rank2.all(axis = 1).shape == numpy.ones(
            shape=(1, rank2.obj.shape[0])).shape
        self.assertTrue(
            all([cond1, cond2, cond3, cond4, cond5, cond6, cond7]))
        
    def test_pd_all(self):
        rank2 = DataFrameStructure(self.B)
        cond1 = rank2.all()
        cond2  = rank2.all(axis=1).shape == (1, self.B.shape[0])
        cond3 = rank2.all(axis = 0).shape == (1,self.B.shape[1],)
        self.assertTrue(
            cond1 and cond2 and cond3
        )
    
    def test_xr_all(self):
        rank3 = DataArrayStructure(self.C)
        rank2 = DataArrayStructure(self.C[:,:,0])
        rank1 = DataArrayStructure(self.C[:,0,0])
        rank3_axis2 = numpy.ones_like(self.C[:,:,0], dtype=bool)
        rank3_axis1 = numpy.ones_like(self.C[:,0,:], dtype = bool)
        rank3_axis0 = numpy.ones_like(self.C[0,:,:], dtype = bool)
        cond1 = rank3.all()
        cond4= rank3.all(axis=2).shape == rank3_axis2.shape
        cond2 =  rank3.all(axis=1).shape == rank3_axis1.shape
        cond3 = rank3.all(axis=0).shape == rank3_axis0.shape
        cond5 = rank2.all()
        cond6 = rank1.all()
        cond7 = rank2.all(axis = 1).shape == numpy.ones(
            shape=(1,rank2.obj.shape[0])).shape
        self.assertTrue(cond1 and cond2 and cond3 and cond4 and cond5 \
            and cond6 and cond7)
        
    def test_np_any(self):
        rank3 = NDArrayStructure(self.A)
        rank2 = NDArrayStructure(self.A[:,:,0])
        rank1 = NDArrayStructure(self.A[:,0,0])
        rank3_axis2 = numpy.ones_like(self.A[:,:,0], dtype=bool)
        rank3_axis1 = numpy.ones_like(self.A[:,0,:], dtype = bool)
        rank3_axis0 = numpy.ones_like(self.A[0,:,:], dtype = bool)
        cond1 = rank3.any()
        cond4= rank3.any(axis=2).shape == rank3_axis2.shape
        cond2 =  rank3.any(axis=1).shape == rank3_axis1.shape
        cond3 = rank3.any(axis=0).shape == rank3_axis0.shape
        cond5 = rank2.any()
        cond6 = rank1.any()
        cond7 = rank2.any(axis = 1).shape == numpy.ones(
            shape=(1, rank2.obj.shape[0])).shape
        self.assertTrue(all([cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7]))
        
    def test_pd_any(self):
        rank2 = DataFrameStructure(self.B)
        cond1 = rank2.any()
        cond2  = rank2.any(axis=1).shape == (1, self.B.shape[0])
        cond3 = rank2.any(axis = 0).shape == (1, self.B.shape[1])
        self.assertTrue(
            cond1 and cond2 and cond3)
        
    def test_pd_any_error(self):
        self.assertRaises(
            ValueError,
            DataFrameStructure(self.B).any, axis = 5
        )
    
    def test_xr_any(self):
        rank3 = DataArrayStructure(self.C)
        rank2 = DataArrayStructure(self.C[:,:,0])
        rank1 = DataArrayStructure(self.C[:,0,0])
        rank3_axis2 = numpy.ones_like(self.C[:,:,0], dtype=bool)
        rank3_axis1 = numpy.ones_like(self.C[:,0,:], dtype = bool)
        rank3_axis0 = numpy.ones_like(self.C[0,:,:], dtype = bool)
        cond1 = rank3.any()
        cond4= rank3.any(axis=2).shape == rank3_axis2.shape
        cond2 =  rank3.any(axis=1).shape == rank3_axis1.shape
        cond3 = rank3.any(axis=0).shape == rank3_axis0.shape
        cond5 = rank2.any()
        cond6 = rank1.any()
        cond7 = rank2.any(axis = 1).shape == numpy.ones(
            shape=(1, rank2.obj.shape[0])).shape
        self.assertTrue(cond1 and cond2 and cond3 and cond4 and cond5 \
            and cond6 and cond7)
        
    def test_np_isna(self):
        notnan = self.A
        nan = self.A.copy()
        nan = numpy.append(self.A,self.A[[-1],:,:], axis=0)
        nan[-1,-1,-1] = numpy.nan
        obj_clean = NDArrayStructure(notnan)
        obj_dirty = NDArrayStructure(nan)
        cond_shape_full:bool = obj_clean.isna().shape == obj_clean.shape
        cond_clean_correct:bool = not obj_clean.isna().any()
        cond_dirty_correct:bool = obj_dirty.isna().any()
        self.assertTrue(cond_shape_full and cond_clean_correct and\
            cond_dirty_correct)
        
    def test_pd_isna(self):
        notnan = self.B
        nan = self.B.copy()
        nan.loc[nan.shape[0]+1,:] = numpy.asarray([numpy.nan]+[1.2]*(
            nan.shape[1]-1))
        obj_clean = DataFrameStructure(notnan)
        obj_dirty = DataFrameStructure(nan)
        cond_shape_full:bool = obj_clean.isna().shape == obj_clean.shape
        cond_clean_correct:bool = not obj_clean.isna().any()
        cond_dirty_correct:bool = obj_dirty.isna().any()
        self.assertTrue(cond_shape_full and cond_clean_correct and\
            cond_dirty_correct)
        
    def test_xr_isna(self):
        notnan = self.C
        nan = self.C.copy()
        nan[0,0,0] = numpy.nan
        obj_clean = NDArrayStructure(notnan)
        obj_dirty = NDArrayStructure(nan)
        cond_shape_full:bool = obj_clean.isna().shape == obj_clean.shape
        cond_clean_correct:bool = not obj_clean.isna().any()
        cond_dirty_correct:bool = obj_dirty.isna().any()
        self.assertTrue(cond_shape_full and cond_clean_correct and\
            cond_dirty_correct)
        
        
    def test_np_transpose(self):
        obj = NDArrayStructure(self.A).transpose()
        rev_coords = dict(
            dim_2 = numpy.asarray(list(range(self.A.shape[-1]))),
            dim_1 = numpy.asarray(list(range(self.A.shape[1]))),
            dim_0 = numpy.asarray(list(range(self.A.shape[0])))
        )
        obj.T((1,0,2))
        self.assertTrue(
            obj._dims == ["dim_2", "dim_1", "dim_0"] and \
            dict_arr_compare(rev_coords, obj._coords)
        )
        
    def test_pd_tranpose(self):
        # No need to retest. Core shuffles are inherited from the numpy
        # adaptor
        DataFrameStructure(self.B).T
        
    def test_xr_transpose(self):
        DataArrayStructure(self.C).transpose((1,0,2))
    
    @unittest.expectedFailure
    def test_multidim_iterrows_warn(self):
        obj1 = NDArrayStructure(self.A)
        self.assertWarns(UserWarning, 
                         obj1.iterrows)
            
    
    def test_np_iterrows(self):
        obj = NDArrayStructure(self.A)
        ind=[]
        rows=[]
        for i, row in  obj.iterrows():
            ind.append(i)
            rows.append(row)
        shape_cond = all(row.shape == obj.shape[1:]for row in rows)
        dim_cond = all(all(row.dims == numpy.asarray(["dim_1", "dim_2"])
                           ) for row in rows)
        cut_coords = {
            k:v for i,(k,v) in enumerate(obj.coords.items()) if i!=0
        }
        coords_cond = all(
            dict_arr_compare(row.coords, cut_coords) for row in rows
        )
        self.assertTrue(shape_cond and dim_cond and coords_cond)
    
    def test_np_itercolumns(self):
        obj = NDArrayStructure(self.A)
        tshape = list(obj.shape)
        tshape.pop(1)
        tshape = tuple(tshape)
        ind=[]
        cols=[]
        for i, col in  obj.itercolumns():
            ind.append(i)
            cols.append(col)
        shape_cond = all(col.shape == tshape for col in cols)
        dim_cond = all(all(col.dims == numpy.asarray(["dim_0", "dim_2"])
                                                 ) for col in cols)
        cut_coords = {
            k:v for i,(k,v) in enumerate(obj.coords.items()) if i!=1
        }
        coords_cond = all(
            dict_arr_compare(col.coords, cut_coords) for col in cols
        )
        self.assertTrue(shape_cond and dim_cond and coords_cond)
        
    def test_pd_iterrows(self):
        from copy import copy
        obj = DataFrameStructure(self.B)
        tshape = tuple([1]+list(obj.shape)[1:])
        ind=[]
        cols=[]
        for i, col in  obj.iterrows():
            ind.append(i)
            cols.append(col)
        shape_cond = all(col.shape == tshape for col in cols)
        dim_cond = all(
            all(col.dims == numpy.asarray(["dim_0", "dim_1"], dtype="U5")
                ) for col in cols)
        cut_coords = copy(obj.coords)
        cut_coords["dim_0"] = pandas.RangeIndex(0,1)
        coords_cond = all(
            dict_arr_compare(col.coords, cut_coords) for col in cols
        )
        self.assertTrue(shape_cond and dim_cond and coords_cond)
    
    @unittest.skip("Needs to be implemented")
    def test_pd_itercolumns(self):
        obj = DataFrameStructure(self.B)
        for i, col in obj.itercolumns():
            print(col)
        self.assertTrue(True)
        
        
    def test_np_casting(self):
        obj = NDArrayStructure(
            self.A
        )
        nobj=obj.cast(numpy.float32)
        self.assertTrue(
            nobj.values.dtype == numpy.float32
        )
        
    def test_pd_casting(self):
        obj = DataFrameStructure(
            self.B
        )
        nobj = obj.cast(numpy.float32)
        self.assertTrue(
            nobj.values.dtype == numpy.float32
        )
    
    def test_xr_casting(self):
        obj = DataArrayStructure(
            self.C
        )
        nobj = obj.cast(numpy.float32)
        self.assertTrue(
            nobj.values.dtype == numpy.float32
        )
        
    def test_np_interface(self):
        '''
            Test errors only. The behavior of underlaying implementations
            are tested elsewhere
        '''
        bridge = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(self.A)
        )
        bridge.transpose()
        bridge.T()
        bridge.transpose(axes=[1,0,2])
        bridge.all(axis=1)
        bridge.all()
        bridge.any()
        bridge.any(axis=1)
        bridge.isna()
        bridge.values()

    def test_pd_interface(self):
        '''
            Test errors only. The behavior of underlaying implementations
            are tested elsewhere
        '''
        bridge = CommonDataStructureInterface(
            _data_structure = DataFrameStructure(self.B)
        )
        bridge.transpose()
        bridge.T()
        bridge.all(axis=1)
        bridge.all()
        bridge.any()
        bridge.any(axis=1)
        bridge.isna()
        bridge.values()
        
    
    def test_xr_interface(self):
        '''
            Test errors only. The behavior of underlaying implementations
            are tested elsewhere
        '''
        bridge = CommonDataStructureInterface(
            _data_structure = DataArrayStructure(self.C)
        )
        bridge.transpose()
        bridge.T()
        bridge.transpose(axes=[1,0,2])
        bridge.all(axis=1)
        bridge.all()
        bridge.any()
        bridge.any(axis=1)
        bridge.isna()
        bridge.values()
    
    def test_processor_nan_handling_exclude_np(self):
        obj = self.A.copy()

        obj[0,0,0] = numpy.nan
        obj[-2,0,0] = numpy.nan
        
        processor = CommonDataProcessor(
            nan_handler_context = NANHandlingContext(
                _nan_strategy = ExcludeMissingNAN)
        )
        processed_dirty = processor(obj)
        processed_clean = processor(self.A)
        bridge = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(
                self.A
            )
        )
        coords_cond:bool = all(numpy.asarray([
            i for i in range(obj.shape[0]) if i not in (0,88)
            ]) == processed_dirty.coords()[processed_dirty.dims()[0]])
        not_nan_cond:bool = not processed_dirty.isna().any()
        clean_coords_cond  = dict_arr_compare(processed_clean.coords(),
                                              bridge.coords())
        clean_dims_cond = all(
            processed_clean.dims() == bridge.dims()
            )
        val_unchanged_cond = (
            processed_clean.values()==bridge.values()).all()
        return self.assertTrue(
            all((coords_cond, not_nan_cond,  clean_dims_cond,
                val_unchanged_cond and clean_coords_cond)
                )
            )
    
    def test_processor_nan_handling_exclude_pd(self):
        obj = self.B.copy(deep=True)
        obj.iloc[0,0] = numpy.nan
        obj.iloc[-2,0] = numpy.nan     
        processor = CommonDataProcessor(
            nan_handler_context = NANHandlingContext(
                _nan_strategy = ExcludeMissingNAN)
        )
        processed_dirty = processor(obj)
        processed_clean = processor(self.B)
        bridge = CommonDataStructureInterface(
            _data_structure = DataFrameStructure(
                self.B
            )
        )
        coords_cond:bool = all(numpy.asarray([
            i for i in range(obj.shape[0]) if i not in (0,88)
            ]) == processed_dirty.coords()[processed_dirty.dims()[0]])
        not_nan_cond:bool = not processed_dirty.isna().any()
        clean_coords_cond  = dict_arr_compare(processed_clean.coords(),
                                              bridge.coords())
        clean_dims_cond = all(processed_clean.dims() == bridge.dims())
        val_unchanged_cond = (
            processed_clean.values()==bridge.values()).all()
        return self.assertTrue(
            all((coords_cond, not_nan_cond,  clean_dims_cond,
                val_unchanged_cond and clean_coords_cond)
                )
            )
        
    def test_processor_nan_handling_exclude_xr(self):
        obj = self.C.copy()
        obj[0,0,0] = numpy.nan
        obj[-2,0,0] = numpy.nan
        processor = CommonDataProcessor(
            nan_handler_context= NANHandlingContext(
                _nan_strategy = ExcludeMissingNAN)
        )
        processed_dirty = processor(obj)
        processed_clean = processor(self.C)
        bridge = CommonDataStructureInterface(
            _data_structure = DataArrayStructure(
                self.C
            )
        )
        coords_cond:bool = all(numpy.asarray([
            i for i in range(obj.shape[0]) if i not in (0,88)
            ]) == processed_dirty.coords()[processed_dirty.dims()[0]])
        not_nan_cond:bool = not processed_dirty.isna().any()
        clean_coords_cond  = dict_arr_compare(processed_clean.coords(),
                                              bridge.coords())
        clean_dims_cond = all(processed_clean.dims() == bridge.dims())
        val_unchanged_cond = (
            processed_clean.values()==bridge.values()).all()
        return self.assertTrue(
            all((coords_cond, not_nan_cond,  clean_dims_cond,
                val_unchanged_cond, clean_coords_cond)
                )
            )
    
    def test_nan_handling_ignore(self):
        obj = self.A.copy()
        obj[0,0,0] = numpy.nan
        obj[-2,0,0] = numpy.nan
        bridge = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(obj)
            )
        processor = CommonDataProcessor(
            nan_handler_context= NANHandlingContext(
                _nan_strategy = IgnoreMissingNAN)
        )
        processed = processor(obj)
        coords_cond = dict_arr_compare(
            processed.coords(), bridge.coords()
        )
        
        return self.assertTrue(coords_cond and
            processed.isna().any()
                               )
    
    def test_nan_handling_impute(self):
         self.assertRaises(
             NotImplementedError, CommonDataProcessor(
                 nan_handler_context= NANHandlingContext(
                _nan_strategy = ImputeMissingNAN)
             ).__call__, self.A
         )
    
    
    
    def test_common_processor(self):
        from copy import copy
        p_core = CommonDataProcessor(
                nan_handler_context= NANHandlingContext(
                    _nan_strategy = ExcludeMissingNAN),
                                 cast = numpy.float32)
        evals:dict[str, bool] = {}
        np_clean = self.A
        np_dirty = self.A.copy()
        pd_clean = self.B
        pd_dirty = self.B.copy(deep=True)
        xr_clean = self.C
        xr_dirty = self.C.copy(deep=True)
        np_dirty[0,0,0], np_dirty[-2,0,0] = numpy.nan, numpy.nan
        xr_dirty[0,0,0], xr_dirty[-2,0,0] = numpy.nan, numpy.nan
        pd_dirty.iloc[0,0], pd_dirty.iloc[-2,0] = numpy.nan, numpy.nan
        
        np_clean_processed = p_core(np_clean)
        np_dirty_processed = p_core(np_dirty)
        pd_clean_processed = p_core(pd_clean)
        pd_dirty_processed = p_core(pd_dirty)
        xr_clean_processed = p_core(xr_clean)
        xr_dirty_processed = p_core(xr_dirty)
        
        evals = dict(
        cond_np_clean_nan = not np_clean_processed.isna().any(),
        cond_np_dirty_nan = not np_dirty_processed.isna().any(),
        cond_pd_clean_nan = not pd_clean_processed.isna().any(),
        cond_pd_dirty_nan = not pd_dirty_processed.isna().any(),
        cond_xr_clean_nan = not xr_clean_processed.isna().any(),
        cond_xr_dirty_nan = not xr_dirty_processed.isna().any(),
        )|evals

        evals = dict(
        cond_np_clean_dtype = np_clean_processed.dtype() == numpy.float32,
        cond_np_dirty_dtype = np_dirty_processed.dtype() == numpy.float32,
        cond_pd_clean_dtype = pd_clean_processed.dtype() == numpy.float32,
        cond_pd_dirty_dtype = pd_dirty_processed.dtype() == numpy.float32,
        cond_xr_clean_dtype = xr_clean_processed.dtype() == numpy.float32,
        cond_xr_dirty_dtype = xr_dirty_processed.dtype() == numpy.float32,
        )|evals
        
        
        np_clean_coords = dict(
            dim_0 = numpy.asarray(range(np_clean.shape[0])),
            dim_1 = numpy.asarray(range(np_clean.shape[1])),
            dim_2 = numpy.asarray(range(np_clean.shape[2])),
            
        )
        np_dirty_coords = copy(np_clean_coords)
        modded = list(range(np_clean.shape[0]))
        modded.pop(0)
        modded.pop(-2)
        np_dirty_coords["dim_0"] = numpy.asarray(modded)
        
        xr_clean_coords = dict(
            dim_0 = numpy.asarray(range(np_clean.shape[0])),
            dim_1 = numpy.asarray(range(np_clean.shape[1])),
            dim_2 = numpy.asarray(range(np_clean.shape[2])),
            
        )
        xr_dirty_coords = copy(xr_clean_coords)
        modded = list(range(np_clean.shape[0]))
        modded.pop(0)
        modded.pop(-2)
        xr_dirty_coords["dim_0"] = numpy.asarray(modded)
        
        pd_clean_coords = dict(
            dim_0 = numpy.asarray(range(pd_clean.shape[0])),
            dim_1 = numpy.asarray([f"var{i}" for i in range(pd_clean.shape[1])], dtype='object'),
        )
        
        dict_arr_compare(np_clean_coords, np_clean_processed.coords())
        
        pd_dirty_coords = copy(pd_clean_coords)
        modded = numpy.asarray([i for i in range(pd_clean.shape[0]) if i not in (0, 88)])
        pd_dirty_coords["dim_0"] = modded
        
        evals = dict(
        np_dims_cond_clean = (np_clean_processed.dims() ==numpy.asarray(
            ["dim_0", "dim_1", "dim_2"]) ).all(),
        np_dims_cond_dirty = (np_dirty_processed.dims() ==numpy.asarray(
            ["dim_0", "dim_1", "dim_2"])).all(),
        np_coords_cond_clean = dict_arr_compare(np_clean_coords,
                                                np_clean_processed.coords()),
        np_coords_cond_dirty = dict_arr_compare(np_dirty_coords,
                                                np_dirty_processed.coords()),
        
        pd_clean_dims_cond = (pd_clean_processed.dims()==numpy.asarray([
            "dim_0", "dim_1"])).all(),
        pd_dirty_dims_cond = (pd_dirty_processed.dims()==numpy.asarray([
            "dim_0", "dim_1"])).all(),
        pd_clean_coords_cond = dict_arr_compare(pd_clean_processed.coords(),
                                                pd_clean_coords),
        pd_dirty_coords_cond = dict_arr_compare(pd_dirty_processed.coords(),
                                                pd_dirty_coords),
        xr_dims_cond_clean = (xr_clean_processed.dims() ==numpy.asarray(
            ["dim0", "dim1", "dim2"])).all(),
        xr_dims_cond_dirty = (xr_dirty_processed.dims() ==numpy.asarray(
            ["dim0", "dim1", "dim2"])).all(),
        xr_coords_cond_clean = dict_arr_compare(xr_clean_coords,
                                                xr_clean_processed.coords()),
        xr_coords_cond_dirty = dict_arr_compare(xr_dirty_coords,
                                                xr_dirty_processed.coords()),
        
        )|evals
        cols = [f"dummy_{i}" for i in range(9)]
        idx = [f"idx_{i}" for i in range(90)]
        new_pd_obj = pandas.DataFrame(data=self.B.values,
                                      columns = cols, index=idx)
        new_xr_obj = xarray.DataArray(
            self.C.values, coords = {
                "dummydim_0": pandas.RangeIndex(0,90),
                "dummydim1":[f"dummy_{i}" for i in range(9)],
                "dummydim2":[f"ddummy_{i}" for i in range(3)],
            }
        )
        new_pd_processed = p_core(new_pd_obj)
        new_xr_processed = p_core(new_xr_obj)
        _x = new_xr_processed.coords()
        _y = dict(
                    dummydim_0=numpy.asarray([range(90)]),
                    dummydim1=numpy.asarray([f"dummy_{i}" for i in range(9)]),
                    dummydim2=numpy.asarray([f"ddummy_{i}" for i in range(3)]),
                    
                )
        evals = dict(
            custom_pd_dims = (new_pd_processed.dims()==numpy.asarray([
                "dim_0","dim_1"])).all(),
            custom_pd_coords = dict_arr_compare(new_pd_processed.coords(),
                                                dict(
                                                    dim_0 = numpy.asarray(idx),
                                                    dim_1 = numpy.asarray(cols)
                                                )),
            custom_xr_dims = (
                numpy.asarray(["dummydim_0", "dummydim1", "dummydim2"]
                              ) == new_xr_processed.dims()).all(),

            custom_xr_coords = dict_arr_compare(
                new_xr_processed.coords(), dict(
                    dummydim_0=numpy.asarray([range(90)]),
                    dummydim1=numpy.asarray([f"dummy_{i}" for i in range(9)]),
                    dummydim2=numpy.asarray([f"ddummy_{i}" for i in range(3)]),
                    
                )),
            )|evals
        

        self.assertTrue(all([v for _,v in evals.items()]))
 
    def test_data_director(self):
        '''
            Check for exceptions only. Functionality tested elsewhere
        '''
        d = DataProcessingDirector(
            processor = CommonDataProcessor,
            nan_handler_context = NANHandlingContext(
                _nan_strategy = ExcludeMissingNAN)
        )
        d(self.A)
        
        
    def test_data_facade(self):
        conditionals={}
        default = Data()
        no_cast = Data(cast=None)
        ignore = Data(nan_handling='ignore')
        impute = Data(nan_handling='impute')
        exclude_explicit = Data(nan_handling="exclude")
        exclude_explicit(self.A)
        ignore(self.A)
        np_processed = default(self.A)
        pd_processed = default(self.B)
        xr_processed = default(self.C)
        conditionals = dict(
        cond1 = np_processed.dtype() ==numpy.float32,
        cond2 = pd_processed.dtype() == numpy.float32,
        cond3 = xr_processed.dtype() == numpy.float32,
        cond4 = no_cast(self.A).dtype() == numpy.float64,
        ignore = ignore.nan_handler._nan_strategy == IgnoreMissingNAN,
        exclude_explicit = exclude_explicit.nan_handler._nan_strategy == ExcludeMissingNAN,
        exclude_implicit = default.nan_handler._nan_strategy == ExcludeMissingNAN,
        )|conditionals
        
        self.assertTrue(
            all([v for _,v in conditionals.items()])
        )
        self.assertRaises(
            NotImplementedError, impute.__call__, self.A
        )
        self.assertRaises(
            ValueError, Data, nan_handling="hello"
        )
        
    def test_np_slicing(self):
        # More test cases needed
        from random import choice
        raw_arr = numpy.random.rand(100,9)
        ridx:int = choice(list(range(raw_arr.shape[0])))
        sidx:int = choice(list(range(raw_arr.shape[1])))
        arr=NDArrayStructure(raw_arr) # type:ignore
        narr = NDArrayStructure(raw_arr)
        narr._coords[f'dim_0'] = numpy.asarray([
            f'sample_{i}' for i in range(raw_arr.shape[0])])
        predicates = dict(
            single_idx = (
                raw_arr[[ridx]] == arr[ridx].values).all(),
            single_idx_dims = (arr.dims==arr[ridx].dims).all(),
            single_idx_coords = dict_arr_compare(dict(
                dim_0 = numpy.asarray([0]),
                dim_1 = arr.coords['dim_1']), arr[ridx].coords
                ),
            slice_idx = raw_arr[ridx, sidx] == arr[ridx, sidx],
            ellipse = (
                raw_arr[ridx, ...] == arr[ridx, ...].values[0,:]).all(),
            ellipse_dims = (arr.dims==arr[ridx,...].dims).all(),
            ellipse_coords = dict_arr_compare(
                dict(dim_0 = numpy.asarray([0]),
                    dim_1 = arr.coords['dim_1']),
                arr[ridx, ...].coords 
                ),
            label_ellipsis = (
                           narr["sample_5",...].values==arr[5,...].values
                           ).all(),
            multilabel = (
                            narr[["sample_5", "sample_6"],...].values==arr[[5,6],...].values
                           ).all(),
            unilabel = (
                           narr["sample_7"].values==arr[7].values
                           ).all(),
            unislice = (
                           narr["sample_0":"sample_4"].values==arr[0:4,...].values
                           ).all(),
            label_and_intslice =(
                           narr["sample_0":"sample_4",...].values==arr[0:4,...].values
                           ).all(),
            neg_unislice = (
                           narr["sample_0":"sample_4"].values!=arr[[6,7],...].values
                           ),
            boolean = (narr[raw_arr>.5]==raw_arr[raw_arr>.5]).all(),
        )
        self.assertTrue(all([
            v for _,v in predicates.items()
        ]))
    
    def test_pd_slicing(self):
        from random import choice
        raw_arr = pandas.DataFrame(
            data=numpy.random.rand(100,9)
            )
        ridx:int = choice(list(range(raw_arr.shape[0])))
        sidx:int = choice(list(range(raw_arr.shape[1])))
        arr = DataFrameStructure(raw_arr) # type:ignore
        narr = DataFrameStructure(raw_arr)
        narr._coords[f'dim_0'] = numpy.asarray([
            f'sample_{i}' for i in range(raw_arr.shape[0])])
        predicates = dict(
            single_idx = (
                raw_arr.iloc[[ridx]] == arr[ridx].values).all(axis=None),
            single_idx_dims = (arr.dims==arr[ridx].dims).all(),
            single_idx_coords = arr[
                ridx].coords["dim_0"]==numpy.asarray([0]) and (
                arr[ridx].coords["dim_1"]==numpy.asarray(raw_arr.columns)
                ).all(),
            slice_idx = raw_arr.iloc[ridx, sidx] == arr[ridx, sidx],
            ellipse = (
                raw_arr.iloc[ridx, ...] == arr[ridx, ...].values[0,:]).all(),
            ellipse_dims = (arr[[1]].dims==arr[ridx,...].dims).all(),
            ellipse_coords = dict_arr_compare(
                dict(
                    dim_0 = numpy.asarray([0]),
                    dim_1 = arr.coords['dim_1']
                    ),
                arr[ridx, ...].coords 
                ),
            label_ellipsis = (
                           narr["sample_5",...].values==arr[5,...].values
                           ).all(),
            multilabel = (
                            narr[["sample_5", "sample_6"],...].values==arr[[5,6],...].values
                           ).all(),
            unilabel = (
                           narr["sample_7"].values==arr[7].values
                           ).all(),
            unislice = (
                           narr["sample_0":"sample_4"].values==arr[0:4,...].values
                           ).all(),
            label_and_intslice =(
                           narr["sample_0":"sample_4",...].values==arr[0:4,...].values
                           ).all(),
            neg_unislice = (
                           narr["sample_0":"sample_4"].values!=arr[[6,7],...].values
                           ),
            boolean = (narr[raw_arr.values>.5]==raw_arr.values[
                raw_arr.values>.5]).all(),
        )
        self.assertTrue(all([
            v for _,v in predicates.items()
        ]))
    
    def test_xr_slicing(self):
        # More test cases needed
        from random import choice
        raw_arr = numpy.random.rand(100,9)
        ridx:int = choice(list(range(raw_arr.shape[0])))
        sidx:int = choice(list(range(raw_arr.shape[1])))
        arr = DataArrayStructure(raw_arr) # type:ignore
        narr = DataArrayStructure(raw_arr)
        narr._coords[f'dim_0'] = numpy.asarray([
            f'sample_{i}' for i in range(raw_arr.shape[0])])
        _x = arr[ridx]
        _y = raw_arr[[ridx]]
        predicates = dict(
            single_idx = (
                raw_arr[[ridx]] == arr[ridx].values).all(),
            single_idx_dims = (arr.dims==arr[ridx].dims).all(),
            single_idx_coords = dict_arr_compare(dict(
                dim_0 = numpy.asarray([0]),
                dim_1 = arr.coords['dim_1']), arr[ridx].coords
                ),
            slice_idx = raw_arr[ridx, sidx] == arr[ridx, sidx],
            ellipse = (
                raw_arr[ridx, ...] == arr[ridx, ...].values[0,:]).all(),
            ellipse_dims = (arr.dims==arr[ridx,...].dims).all(),
            ellipse_coords = dict_arr_compare(
                dict(dim_0 = numpy.asarray([0]),
                    dim_1 = arr.coords['dim_1']),
                arr[ridx, ...].coords 
                ),
            label_ellipsis = (
                           narr["sample_5",...].values==arr[5,...].values
                           ).all(),
            multilabel = (
                            narr[["sample_5", "sample_6"],...].values==arr[[5,6],...].values
                           ).all(),
            unilabel = (
                           narr["sample_7"].values==arr[7].values
                           ).all(),
            unislice = (
                           narr["sample_0":"sample_4"].values==arr[0:4,...].values
                           ).all(),
            label_and_intslice =(
                           narr["sample_0":"sample_4",...].values==arr[0:4,...].values
                           ).all(),
            neg_unislice = (
                           narr["sample_0":"sample_4"].values!=arr[[6,7],...].values
                           ),
            boolean = (narr[raw_arr>.5]==raw_arr[raw_arr>.5]).all(),
        )
        self.assertTrue(all([
            v for _,v in predicates.items()
        ]))
        
    def test_slice_interface(self):
        '''
            Test forwarding by testing lack of errors only.
            Underlying implementation tested elsewhere
        '''
        raw_arr = np.random.rand(100,9,3)
        arr:np.ndarray[float] = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(
                raw_arr
            )
        )
        df = CommonDataStructureInterface(
            _data_structure = DataFrameStructure(
                pd.DataFrame(raw_arr[:,:,0])
            )
        )
        xarr =CommonDataStructureInterface(
                _data_structure = DataArrayStructure(
                    xr.DataArray(raw_arr)
            )
        )
        x = xarr[0:10:5]
        d = df[0:2]
        a = arr[0:10]
    
    def test_unique(self):                
        A = np.random.randint(0, high=5, size=(30,9,3)).astype(object)
        B = A.copy()
        A[0,0,0] =np.nan
        A[1,1,1] = np.nan
        
        arr:np.typing.ndarray[Any] = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(
                A
            )
        )
        arr_clean:np.typing.ndarray[Any] = CommonDataStructureInterface(
            _data_structure = NDArrayStructure(
                B
            )
        )
        df = CommonDataStructureInterface(
            _data_structure = DataFrameStructure(
                pd.DataFrame(A[:,:,0])
            )
        )
        df_clean = CommonDataStructureInterface(
            _data_structure = DataFrameStructure(
                pd.DataFrame(B[:,:,0])
            )
        )
        xarr =CommonDataStructureInterface(
                _data_structure = DataArrayStructure(
                    xr.DataArray(A)
            )
        )
        xarr_clean =CommonDataStructureInterface(
                _data_structure = DataArrayStructure(
                    xr.DataArray(B)
            )
        )
        predicates = dict(
            np_axNone_left = next(arr_clean.unique())[0] is None,
            np_axNone_right =  (next(arr_clean.unique())[1] == np.unique(B)).all(),
            np_ax0_left = list(
                e[0] for e in arr_clean.unique(axis=0)
                )==list(range(arr_clean.shape()[0])),
            np_ax0_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[i,...]) for i in range(B.shape[0])),
                    (e[1] for e in arr_clean.unique(axis=0)),
                    )
                ))),
            np_ax1_left = list(
                e[0] for e in arr_clean.unique(axis=1)
                )==list(range(arr_clean.shape()[1])),
            np_ax1_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[:,i,:]) for i in range(B.shape[1])),
                    (e[1] for e in arr_clean.unique(axis=1)),
                    )
                ))), 
            np_ax2_left = list(
                e[0] for e in arr_clean.unique(axis=2)
                )==list(range(arr_clean.shape()[2])),
            np_ax2_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[:,:,i]) for i in range(B.shape[2])),
                    (e[1] for e in arr_clean.unique(axis=2)),
                    )
                ))), 
            pd_axNone_left = next(df_clean.unique())[0] is None,
            pd_axNone_right =  (next(df_clean.unique())[1] == np.unique(B)).all(),
            pd_ax0_left = list(
                e[0] for e in df_clean.unique(axis=0)
                )==list(range(df_clean.shape()[0])),
            pd_ax0_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[i,:,0]) for i in range(B.shape[0])),
                    (e[1] for e in df_clean.unique(axis=0)),
                    )
                ))),
            pd_ax1_left = list(
                e[0] for e in df_clean.unique(axis=1)
                )==list(range(df_clean.shape()[1])),
            pd_ax1_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[:,i,0]) for i in range(B.shape[1])),
                    (e[1] for e in df_clean.unique(axis=1)),
                    )
                ))),
            xarr_axNone_left = next(xarr_clean.unique())[0] is None,
            xarr_axNone_right =  (next(xarr_clean.unique())[1] == np.unique(B)).all(),
            xarr_ax0_left = list(
                e[0] for e in xarr_clean.unique(axis=0)
                )==list(range(xarr_clean.shape()[0])),
            xarr_ax0_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[i,...]) for i in range(B.shape[0])),
                    (e[1] for e in xarr_clean.unique(axis=0)),
                    )
                ))),
            xarr_ax1_left = list(
                e[0] for e in xarr_clean.unique(axis=1)
                )==list(range(xarr_clean.shape()[1])),
            xarr_ax1_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[:,i,:]) for i in range(B.shape[1])),
                    (e[1] for e in xarr_clean.unique(axis=1)),
                    )
                ))), 
            xarr_ax2_left = list(
                e[0] for e in xarr_clean.unique(axis=2)
                )==list(range(xarr_clean.shape()[2])),
            xarr_ax2_right = all(list(map(
                lambda e: np.array_equal(e[0], e[1]),
                zip(
                    (np.unique(B[:,:,i]) for i in range(B.shape[2])),
                    (e[1] for e in xarr_clean.unique(axis=2)),
                    )
                ))), 
        )
        self.assertTrue(all([
            v for _,v in predicates.items()
        ]))
        
    def test_np_ops(self):
        A = np.random.rand(50,9,3)
        B = A.copy().astype(str)
        obj = NDArrayStructure(A)
        obj_str = NDArrayStructure(B)
        interface = CommonDataStructureInterface(
            _data_structure = obj
        )
        str_interface = CommonDataStructureInterface(
            _data_structure = obj_str
        )
        eq_ref = A==A[0,0]
        lt_ref = A < 0
        le_ref = A <= 0
        gt_ref = A >= 0
        ge_ref = A > 0
        ne_ref = A != A[0,0]
        eq_str_ref = B == B[0,0]
        neq_str_ref = B != B[0,0]
        
        predicates_nums = dict(
            eq = ((obj == A[0,0]).values == eq_ref).all(),
            lt = ((obj < 0).values == lt_ref).all(),
            le = ((obj <= 0).values == le_ref).all(), 
            gt = ((obj >= 0).values == gt_ref).all(),
            ge = ((obj > 0).values == ge_ref).all(),
            ne = ((obj != A[0,0]).values == ne_ref).all()
        )
        predicates_str = dict(
            eq_str = ( (obj_str == B[0,0]).values ==  eq_str_ref).all(),
            neq_str = ((obj_str != B[0,0]).values == neq_str_ref).all(),
        )
        predicates_interface = dict(
            eq_int = ((interface == A[0,0]).values() == eq_ref).all(),
            lt_int = ((interface < 0).values() == lt_ref).all(),
            le_int = ((interface <= 0).values() == le_ref).all(), 
            gt_int = ((interface >= 0).values() == gt_ref).all(),
            ge_int = ((interface > 0).values() == ge_ref).all(),
            ne_int = ((interface != A[0,0]).values() == ne_ref).all(),
            eq_str_int = ( (str_interface == B[0,0]).values() ==  eq_str_ref).all(),
            neq_str_int = ((str_interface != B[0,0]).values() == neq_str_ref).all(),
            
        )
        
        predicates=predicates_nums|predicates_str|predicates_interface
        
        self.assertTrue(
            all([
                v for _,v in predicates.items()
            ])
        )
    
    def test_pd_ops(self):
        A = pd.DataFrame(np.random.rand(50,9),
                         columns = [f"var_{i}" for i in range(9)],
                         index = [f"sample_{i}" for i in range(50)])
        B = A.copy(deep=True).astype(str)
        obj = DataFrameStructure(A)
        interface = CommonDataStructureInterface(
            _data_structure = obj
        )
        obj_str = DataFrameStructure(B)
        str_interface = CommonDataStructureInterface(
            _data_structure = obj_str
        )
        eq_ref = A==A.iloc[0,0]
        lt_ref = A < 0
        le_ref = A <= 0
        gt_ref = A >= 0
        ge_ref = A > 0
        ne_ref = A != A.iloc[0,0]
        eq_str_ref = B==B.iloc[0,0]
        neq_str_ref = B != B.iloc[0,0]
        
        predicates_nums = dict(
            eq = ((obj == A.iloc[0,0]).values == eq_ref).all(axis=None),
            lt = ((obj < 0).values == lt_ref).all(axis=None),
            le = ((obj <= 0).values == le_ref).all(axis=None), 
            gt = ((obj >= 0).values == gt_ref).all(axis=None),
            ge = ((obj > 0).values == ge_ref).all(axis=None),
            ne = ((obj != A.iloc[0,0]).values == ne_ref).all(axis=None)
        )
        predicates_str = dict(
            eq_str = (
                (obj_str == B.iloc[0,0]) == eq_str_ref
                ).all(axis=None),
            neq_str = (
                (obj_str != B.iloc[0,0]) == neq_str_ref
                ).all(axis=None),
        )
        predicates_interface = dict(
            eq_int = ((interface == A.iloc[0,0]).values() == eq_ref).all(axis=None),
            lt_int = ((interface < 0).values() == lt_ref).all(axis=None),
            le_int = ((interface <= 0).values() == le_ref).all(axis=None), 
            gt_int = ((interface >= 0).values() == gt_ref).all(axis=None),
            ge_int = ((interface > 0).values() == ge_ref).all(axis=None),
            ne_int = ((interface != A.iloc[0,0]).values() == ne_ref).all(axis=None),
            eq_str_int = ( (str_interface == B.iloc[0,0]).values() ==  eq_str_ref).all(axis=None),
            neq_str_int = ((str_interface != B.iloc[0,0]).values() == neq_str_ref).all(axis=None),    
        )
        predicates=predicates_nums|predicates_str|predicates_interface
        self.assertTrue(
            all([
                v for _,v in predicates.items()
            ])
        )
        
    def test_xr_ops(self):
        A = xr.DataArray(np.random.rand(50,9,3),
                         coords = dict(
                             sample = np.asarray([
                                 f"sample_{e}" for e in range(50)
                                 ]),
                             data_dim1 = np.asarray([
                                 f"var_1{e}" for e in range(9)
                             ]),
                             dat_dim2 = np.asarray([
                                 f"var_2{e}" for e in range(3)
                             ]),
                         )
                        )
        B = A.copy(deep=True).astype(str)
        obj = DataArrayStructure(A)
        interface = CommonDataStructureInterface(
            _data_structure = obj
        )
        obj_str = DataArrayStructure(B)
        str_interface = CommonDataStructureInterface(
            _data_structure = obj_str
        )
        eq_ref = A==A[0,0]
        lt_ref = A < 0
        le_ref = A <= 0
        gt_ref = A >= 0
        ge_ref = A > 0
        ne_ref = A != A[0,0]
        eq_str_ref = B == B[0,0]
        neq_str_ref = B != B[0,0]
        
        predicates_nums = dict(
            eq = ((obj == A[0,0]).values == eq_ref).all(),
            lt = ((obj < 0).values == lt_ref).all(),
            le = ((obj <= 0).values == le_ref).all(), 
            gt = ((obj >= 0).values == gt_ref).all(),
            ge = ((obj > 0).values == ge_ref).all(),
            ne = ((obj != A[0,0]).values == ne_ref).all(),
        )
        predicates_str = dict(
            eq_str = ( (obj_str == B[0,0]).values ==  eq_str_ref).all(),
            neq_str = ((obj_str != B[0,0]).values == neq_str_ref).all(),
        )
        predicates_interface = dict(
            eq_int = ((interface == A[0,0]).values() == eq_ref).all(),
            lt_int = ((interface < 0).values() == lt_ref).all(),
            le_int = ((interface <= 0).values() == le_ref).all(), 
            gt_int = ((interface >= 0).values() == gt_ref).all(),
            ge_int = ((interface > 0).values() == ge_ref).all(),
            ne_int = ((interface != A[0,0]).values() == ne_ref).all(),
            eq_str_int = ( (str_interface == B[0,0]).values() ==  eq_str_ref).all(),
            neq_str_int = ((str_interface != B[0,0]).values() == neq_str_ref).all(),
            
        )
        predicates=predicates_nums|predicates_str|predicates_interface
        self.assertTrue(
            all([
                v for _,v in predicates.items()
            ])
        )
      
        