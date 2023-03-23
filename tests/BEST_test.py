import unittest
from bayesian_models.models import BEST
import numpy as np
import pandas as pd
from os import remove
import warnings


PREDICATES = dict[str, bool]
warnings.filterwarnings("ignore")

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
        

    def tearDown(self):
        targets = ["temp_model.netcdf","temp_model.pickle"]
        for target in targets:
            try:
                remove(target)
            except FileNotFoundError:
                pass

    def test_no_errors_run(self):
        obj = BEST()(self.df, "group")
        flag:bool = self.df.isna().any(axis=None)
        obj.fit(draws=100, chains=2, tune=100,
                progressbar=False)
        obj.predict()
        self.assertTrue(True)

    def test_save_netcdf(self):
        obj = BEST(save_path='ignored_path.netcdf')(self.df, "group")
        obj.fit(draws=30, chains=2, tune=30,
                progressbar=False)
        obj.save("temp_model.netcdf")
        obj_other=BEST()(self.df, "group")
        obj_other.load("temp_model.netcdf")
        self.assertTrue(obj.idata==obj_other.idata )

    # See issue #5
    @unittest.expectedFailure
    def test_save_pickle(self):
        import pickle
        obj = BEST()(self.df, "group")
        obj.fit(draws=100, chains=2, tune=100,
                progressbar=False)
        obj.save("temp_model.pickle", method="pickle")
        with open("temp_model.pickle", "rb") as file:
            obj_other = pickle.load(file)
        self.assertTrue(obj.idata==obj_other.idata)


    def test_illegal_nan_option(self):
        
        self.assertRaises(ValueError, BEST, 
                          nan_handling="this_is_wrong", 
                          )

    def test_nan_valid(self):
        '''
            Test errors only. The behavior is tested in the
            data module
        '''
        obj = BEST(nan_handling='impute')
        obj2 = BEST(nan_handling='exclude')

    def test_nan_idx3_error(self):
        '''
            Test for weird heisenbug where setting a nan value at
            index 3 raises an InderError. Was due to wrong value lookups
            Testing lack of errors only
        '''
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[-1]+1]=[np.nan, "placebo"]
        BEST()(missing_nan, "group")
        



    def test_missing_nan_warns(self):
        from bayesian_models.data import Data
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[0]+1]=[np.nan, "drug"]
        missing_nan.loc[missing_nan.shape[0]+1]=[np.nan, "placebo"]
        with self.assertWarns(UserWarning):
            obj = BEST()(missing_nan, "group")



    def test_nan_exclude(self):
        from bayesian_models.utilities import flatten
        gather = lambda object: list(
            flatten(list(v for k,v in object._groups.items())))
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[0]+1]=[np.nan, "drug"]
        missing_nan.loc[missing_nan.shape[0]+1]=[np.nan, "placebo"]
        missing_nan.loc[missing_nan.shape[0]+1] = [100.0, 'drug']
        obj_dirty = BEST()(missing_nan, "group")
        obj_clean = BEST()(self.df, "group")
        dirty_indices:list[int] = {e for e in flatten([
           v for _, v in obj_dirty._groups.items()])}
        clean_indices:list[int] = { e for e in flatten([
           v for _, v in obj_clean._groups.items()])}
        
        predicates:PREDICATES = dict(
            dirty_indices = set(range(self.df.shape[0]+1))==dirty_indices,
            clean_indices = set(range(self.df.shape[0]))==clean_indices
        )
        self.assertTrue(all([
            v for _, v in  predicates.items()
        ]))

    def test_nan_impute(self):
        missing_df = self.df.copy(deep=True)
        missing_df.loc[missing_df.shape[0]+1]=[np.nan,'placebo']
        self.assertRaises(NotImplementedError ,
                          BEST(nan_handling='impute').__call__,
                          missing_df, "group")
    

    def test_nan_present(self):
        missing_nan = self.df.copy(deep=True)
        missing_nan.loc[missing_nan.shape[-1]+1]=[np.nan, "drug"]
        missing_nan.loc[missing_nan.shape[-1]+1]=[np.nan, "placebo"]
        obj_exclude = BEST()(missing_nan, "group")
        self.assertTrue(obj_exclude.nan_present_flag)

    # See issue #10
    @unittest.expectedFailure
    def test_tidy_data(self):
        # Index currently ignored by the model
        multi_df = self.df.copy(deep= True)
        multi_df.columns = [["iq","treatment"], self.df.columns]
        multi_df.index = [['dummy0']*5+['dummy1']*5, multi_df.index]
        obj_multi = BEST()(multi_df, "treatment.group")
        obj_plain = BEST()(self.df, "group")
        tidy_columns = ['iq.value', 'treatment.group']
        plain_columns = self.df.columns
        self.assertTrue(
            obj_plain._coords['dimentions'] == self.df.loc['']
        )
        
    def test_levels(self):
        multipair_df = self.df.copy(deep=True)
        multipair_df.loc[multipair_df.shape[0]+1] = [100, 
                                                      'dummy_level0']
        multipair_df.loc[multipair_df.shape[0]+1] = [150, 
                                                      'dummy_level0']
        multipair_df.loc[multipair_df.shape[0]+1] = [100, 
                                                      'dummy_level1']
        multipair_df.loc[multipair_df.shape[0]+1] = [130, 
                                                      'dummy_level1']
        obj_multiple_pairs = BEST()(multipair_df, "group")
        obj_single_pair = BEST()(self.df, "group")
        single_pair_cond = set(obj_single_pair._groups.keys()) == \
                set(['placebo',"drug"])
        multiple_pairs_cond = set(obj_multiple_pairs._groups.keys()) ==\
            set(['placebo', "drug",  "dummy_level0", "dummy_level1"])
        self.assertTrue(single_pair_cond and multiple_pairs_cond)


    def test_group_combinations(self):
        from itertools import combinations
        obj_single_pair = BEST()(self.df, "group")
        multipair_df = self.df.copy(deep=True)
        multipair_df.loc[multipair_df.shape[0]+1] = [100, 
                                                      'dummy_level0']
        multipair_df.loc[multipair_df.shape[0]+1] = [150, 
                                                      'dummy_level0']
        multipair_df.loc[multipair_df.shape[0]+1] = [100, 
                                                      'dummy_level1']
        multipair_df.loc[multipair_df.shape[0]+1] = [130, 
                                                      'dummy_level1']
        multipair_obj = BEST()(multipair_df, "group")
        simple_condition =  set(combinations(["drug",'placebo',],2)) ==\
              set(obj_single_pair._permutations)
        sorted_vals =["drug","placebo", "dummy_level0", 
                          "dummy_level1"]
        sorted_vals.sort()
        multipair_condition = set(
            combinations(sorted_vals,2)) == set(
            multipair_obj._permutations)
        self.assertTrue(simple_condition and multipair_condition)



    def test_features(self):
        obj_simple = BEST()(self.df, "group")
        complex_df = self.df.copy(deep=True)
        complex_df["iq"] = complex_df.value*.9
        obj_complex = BEST()(complex_df, "group")
        self.assertTrue(
            set(obj_simple.features) == set(("value",)) and set(obj_complex.features) == \
            set(["value", "iq"])
        )

    def test_initialized_fit(self):
        obj = BEST()
        self.assertRaises(
            RuntimeError, obj.fit,
        )

    def test_fit(self):
        BEST()(self.df, "group").fit(50, tune=50, progressbar=False)

    def test_ground_truth(self):
        ε = 1e-1
        ref_val_mu = 1.0
        ref_val_sigma = .93
        ref_val_sig = "Not Significant"
        obj = BEST(std_difference=True)(self.df, "group")
        obj.fit(tune=1000, draws=2000, chains=2,
                progressbar=False)
        results = obj.predict(var_names=["Δμ", "Δσ"],
                          ropes=[(0,3), (0,3)],
                          hdis=[.94,.94])
        Δμ = results["Δμ"].loc[:,'mean']
        Δσ =  results["Δμ"].loc[:,'mean']
        sig = results["Δσ"].loc[:,"Significance"]
        self.assertTrue( Δμ.iloc[0]-ref_val_mu <= ε)

    def test_desition_rule(self):
        obj = BEST()(self.df, "group")
        obj.fit(tune=1000, draws=2000, chains=2,
                progressbar=False)
        not_sig_results = obj.predict(var_names=["Δμ"],
                          ropes=[(0.0,3.0)],
                          hdis=[.94,])["Δμ"]
        sig_results = obj.predict(var_names=["Δμ"],
                          ropes=[(2.0,5.0)],
                          hdis=[.94,])["Δμ"]
        ind_results = obj.predict(var_names=["Δμ"],
                          ropes=[(1.0,2.0)],
                          hdis=[.94,])["Δμ"]
        
        self.assertTrue(
            not_sig_results.Significance.values[0] == "Not Significant"\
            and \
            sig_results.Significance.values[0] == "Significant" and \
            ind_results.Significance.values[0] == "Indeterminate"
        )

    def test_multivariate(self):
        complex_df = self.df.copy(deep=True)
        complex_df["iq"] = complex_df.value*.9
        obj = BEST(multivariate_likelihood = True)(complex_df, "group")
        obj.fit(tune=50, draws=50 ,chains=2 ,progressbar=False)
        obj.predict()


    def test_univariate_sepperate(self):
        complex_df = self.df.copy(deep=True)
        complex_df["iq"] = complex_df.value*.9
        obj = BEST(common_shape=False)(complex_df, "group")
        obj.fit(tune=50, draws=50 ,chains=2 ,progressbar=False)
        obj.predict()
        
    def test_predict_raises_missing_sigma(self):
        obj = BEST()(self.df, 'group')
        obj.fit(draws=10, chains=2, tune=10,
                progressbar=False)
        self.assertRaises(
            RuntimeError, obj.predict,
            var_names=['Δσ'],
            ropes = [(0,1)],
            hdis = [.94]
        )
        
    def test_predict_raises_missing_effect(self):
        obj = BEST(std_difference=True)(self.df, 'group')
        obj.fit(draws=10, chains=2, tune=10,
                progressbar=False)
        self.assertRaises(
            RuntimeError, obj.predict,
            var_names=['Effect_Size'],
            ropes = [(0,1)],
            hdis = [.94]
        )
        
    def test_predict_warns_uneven_lengths(self):

        obj = BEST(std_difference=True)(self.df, 'group')
        obj.fit(draws=10, chains=2, tune=10,
                progressbar=False)
        self.assertWarns(
            UserWarning, obj.predict,
            var_names=['Δσ', 'Δμ'],
            ropes = [(0,1)],
            hdis = [.94]
        )
        
    def test_zero_variance(self):
        single_member_group_df = self.df.copy(deep=True)
        same_vals_group_df = self.df.copy(deep=True)
        same_vals_group_df.loc[
            same_vals_group_df.shape[0]+1] = [100, 'dummy_level0']
        same_vals_group_df.loc[
            same_vals_group_df.shape[0]+1] = [100, 'dummy_level0']
        single_member_group_df.loc[
            single_member_group_df.shape[0]+1
            ] = [100, 'dummy_level1']
        with self.assertRaises(ValueError):
            single_val_obj = BEST()(single_member_group_df, 'group')
        with self.assertRaises(ValueError):
            same_vals_group_obj = BEST()(same_vals_group_df, 'group')
        
    def test_consistency_checks_common_ddof(self):
        self.assertWarns(UserWarning, BEST, common_shape=False)
        
    def test_consistency_mv_warns(self):
        self.assertWarns(UserWarning, BEST,
                         multivariate_likelihood=True)
    
    def test_consistency_checks_multivariate(self):
        self.assertWarns(
            UserWarning, BEST, multivariate_likelihood=True,
            common_shape=False
        )
        
    def test_55(self):
        import sklearn
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True, as_frame = True)
        names = load_iris().target_names
        df = pd.concat([X, y], axis=1)
        df_names = df.copy(deep=True)
        df_names.iloc[:, -1] = df_names.iloc[:,-1].replace({
            i:names[i] for i in range(len(names))
        })
        obj = BEST()(df_names, 'target')
        obj.fit(tune=100, draws=100, chains=2)