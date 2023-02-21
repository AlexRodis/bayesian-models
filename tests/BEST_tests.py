import unittest
from bayesian_models.models import BEST
import numpy as np
import xarray as xr
import pandas as pd
from os import remove

class TestBESTModel(unittest.TestCase):

    def setUp(self):
        drug = (101,100,102,104,102,97,105,105,98,101,100,123,
                105,103,100,95,102,106, 109,102,82,102,100,102,102,101,
                102,102,103,103,97,97,103,101,97,104, 96,103,124,101,
                101,100,101,101,104,100,101)
        placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,
                   105,88,101,100, 104,100,100,100,101,102,103,97,101,
                   101,100,101,99,101,100,100,101,100,99,101,100,102,99,
                   100,99)
        y1 = np.array(drug)
        y2 = np.array(placebo)
        self.df = pd.DataFrame(
            dict(value=np.r_[y1, y2],
                 group=np.r_[["drug"] * len(drug), ["placebo"] * len(
            placebo)])
            )
        # self.df = y.drop("index", axis=1)
        self.multi_df = self.df.copy(deep= True)
        self.multi_df.columns = [["iq","treatment"], self.df.columns]  

    def tearDown(self):
        targets = ["temp_model.netcdf"]
        for target in targets:
            try:
                remove(target)
            except FileNotFoundError:
                pass

    def test_no_errors_run(self):
        obj = BEST()(self.df, "group")
        obj.fit(draws=100, chains=2, tune=100)
        obj.predict()
        self.assertTrue(True)

    def test_save_netcdf(self):
        obj = BEST(save_path='ignored_path.netcdf')(self.df, "group")
        obj.fit(draws=100, chains=2, tune=100)
        obj.save("temp_model.netcdf")
        obj_other=BEST()(self.df, "group")
        obj_other.load("temp_model.netcdf")
        self.assertTrue(obj.idata==obj_other.idata )

    @unittest.expectedFailure
    def test_save_pickle(self):
        
        '''
            See issue #5
        '''
        import pickle
        obj = BEST()(self.df, "group")
        obj.fit(draws=100, chains=2, tune=100)
        obj.save("temp_model.pickle", method="pickle")
        with open("temp_model.pickle", "rb") as file:
            obj_other = pickle.load(file)
        self.assertTrue(obj.idata==obj_other.idata)


    def test_illegal_nan_option(self):
        self.assertRaises(ValueError, BEST, 
                          nan_handling="this_is_wrong", 
                          )

    def test_nan_valid(self):
        obj = BEST(nan_handling='impute')
        obj2 = BEST(nan_handling='exclude')

    def test_missing_nan_warns(self):
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "drug"]
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "placebo"]
        obj = BEST()(missing_nan, "group")
        with self.assertWarns(UserWarning):
            obj.fit(tune=50, draws=10, chains=2)

    def test_nan_exclude(self):
        from bayesian_models.utilities import flatten
        gather = lambda object: list(
            flatten(list(v for k,v in object._groups.items())))
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "drug"]
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "placebo"]
        obj_clean = BEST()
        obj_dirty = BEST()
        obj_dirty._preprocessing_(missing_nan, "group")
        obj_clean._preprocessing_(self.df, "group")
        dirty_indices= gather(obj_dirty)
        clean_indices = gather(obj_clean)
        self.assertTrue(
            dirty_indices==clean_indices and \
                not obj_clean.nan_present_flag and \
                obj_dirty.nan_present_flag
        )

    # See issue #7
    @unittest.expectedFailure
    def test_nan_impute(self):
        self.assertRaises(NotImplementedError ,
                          BEST(nan_handling='impute').__call__,
                          self.df, "group")

    @unittest.expectedFailure
    def test_nan_present(self):
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "drug"]
        missing_nan.loc[missing_nan.shape[-1]+1]=[None, "placebo"]
        obj_exclude = BEST()(missing_nan, "group")
        self.assertTrue(obj_exclude.nan_present_flag)

    # See issue #8
    def test_tidy_data(self):
        mindex = self.df.copy(deep=True)
        mindex.columns = [["iq", "test_group"], mindex.columns]
        mindex.index = [['dummy1']*5+["dummy2"]*5, mindex.index]
        obj_0 = BEST()(self.df, "group")
        obj_1 = BEST()(mindex, "group")
        print(obj_1.features)

    def test_group_combinations(self):
        pass

    def test_ddof_warning(self):
        pass

    def test_multivariate_warning_shape_ignored(self):
        pass

    def test_levels(self):
        pass

    def test_features(self):
        pass

    def test_initialized_fit(self):
        pass

    def test_fit(self):
        pass

    def test_ground_truth(self):
        pass

    def test_decition_rule(self):
        pass

