import unittest
from bayesian_models.data import NDArrayAdaptor, DataFrameAdaptor, \
    DataArrayAdaptor
import pandas
import xarray
import numpy


class TestDataAdaptor(unittest.TestCase):
    
    
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
        
        
    def test_np_all(self):
        rank3 = NDArrayAdaptor(self.A)
        rank2 = NDArrayAdaptor(self.A[:,:,0])
        rank1 = NDArrayAdaptor(self.A[:,0,0])
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
            shape=rank2.obj.shape[0]).shape
        self.assertTrue(cond1 and cond2 and cond3 and cond4 and cond5 \
            and cond6 and cond7)
        
    def test_pd_all(self):
        rank2 = DataFrameAdaptor(self.B)
        cond1 = rank2.all()
        cond2  = rank2.all(axis=1).shape == (self.B.shape[0],)
        cond3 = rank2.all(axis = 0).shape == (self.B.shape[1],)
        self.assertTrue(
            cond1 and cond2 and cond3
        )
    
    def test_xr_all(self):
        rank3 = DataArrayAdaptor(self.C)
        rank2 = DataArrayAdaptor(self.C[:,:,0])
        rank1 = DataArrayAdaptor(self.C[:,0,0])
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
        rank3 = NDArrayAdaptor(self.A)
        rank2 = NDArrayAdaptor(self.A[:,:,0])
        rank1 = NDArrayAdaptor(self.A[:,0,0])
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
            shape=rank2.obj.shape[0]).shape
        self.assertTrue(cond1 and cond2 and cond3 and cond4 and cond5 \
            and cond6 and cond7)
        
    def test_pd_any(self):
        rank2 = DataFrameAdaptor(self.B)
        cond1 = rank2.any()
        cond2  = rank2.any(axis=1).shape == (self.B.shape[0],)
        cond3 = rank2.any(axis = 0).shape == (self.B.shape[1],)
        self.assertTrue(
            cond1 and cond2 and cond3)
    
    def test_xr_any(self):
        rank3 = DataArrayAdaptor(self.C)
        rank2 = DataArrayAdaptor(self.C[:,:,0])
        rank1 = DataArrayAdaptor(self.C[:,0,0])
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
            shape=rank2.obj.shape[0]).shape
        self.assertTrue(cond1 and cond2 and cond3 and cond4 and cond5 \
            and cond6 and cond7)
        
    def test_np_isna(self):
        notnan = self.A
        nan = self.A.copy()
        nan = numpy.append(self.A,self.A[[-1],:,:], axis=0)
        nan[-1,-1,-1] = numpy.nan
        obj_clean = NDArrayAdaptor(notnan)
        obj_dirty = NDArrayAdaptor(nan)
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
        obj_clean = DataFrameAdaptor(notnan)
        obj_dirty = DataFrameAdaptor(nan)
        cond_shape_full:bool = obj_clean.isna().shape == obj_clean.shape
        cond_clean_correct:bool = not obj_clean.isna().any()
        cond_dirty_correct:bool = obj_dirty.isna().any()
        self.assertTrue(cond_shape_full and cond_clean_correct and\
            cond_dirty_correct)
        
    def test_xr_isna(self):
        notnan = self.C
        nan = self.C.copy()
        nan[0,0,0] = numpy.nan
        obj_clean = NDArrayAdaptor(notnan)
        obj_dirty = NDArrayAdaptor(nan)
        cond_shape_full:bool = obj_clean.isna().shape == obj_clean.shape
        cond_clean_correct:bool = not obj_clean.isna().any()
        cond_dirty_correct:bool = obj_dirty.isna().any()
        self.assertTrue(cond_shape_full and cond_clean_correct and\
            cond_dirty_correct)