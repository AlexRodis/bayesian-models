import unittest
from bayesian_models.data import NDArrayStructure, DataFrameStructure, \
    DataArrayStructure, CommonDataStructureInterface, \
        CommonDataProcessor, ExcludeMissingNAN, ImputeMissingNAN, \
        IgnoreMissingNAN, DataProcessingDirector, NANHandlingContext
        
import pandas
import xarray
import numpy

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
            shape=(rank2.obj.shape[0],1)).shape
        self.assertTrue(
            all([cond1, cond2, cond3, cond4, cond5, cond6, cond7]))
        
    def test_pd_all(self):
        rank2 = DataFrameStructure(self.B)
        cond1 = rank2.all()
        cond2  = rank2.all(axis=1).shape == (self.B.shape[0],1)
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
            shape=rank2.obj.shape[0]).shape
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
            shape=(rank2.obj.shape[0],1)).shape
        self.assertTrue(all([cond1, cond2, cond3, cond4, cond5, cond6,
                             cond7]))
        
    def test_pd_any(self):
        rank2 = DataFrameStructure(self.B)
        cond1 = rank2.any()
        cond2  = rank2.any(axis=1).shape == (self.B.shape[0], 1)
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
            shape=(rank2.obj.shape[0],1)).shape
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
            nan_handler = ExcludeMissingNAN()
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
        clean_dims_cond = all(processed_clean.dims() == bridge.dims())
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
            nan_handler = ExcludeMissingNAN()
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
            nan_handler = ExcludeMissingNAN()
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
            nan_handler = IgnoreMissingNAN()
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
                 nan_handler = ImputeMissingNAN()
             ).__call__, self.A
         )
          