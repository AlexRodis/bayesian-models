import unittest
import numpy
import pandas
from bayesian_models.utilities import tidy_multiindex


class TestTidyMultiIndex(unittest.TestCase):


    def setUp(self):
        base_cols:list[str] =["col1", "col2", "col3"]
        base_idx:list = list(range(10))
        multicols_single = ["dummy_level"]*3
        multicols_multi = ["dummy_level1"]*1+["dummy_level2"]*2
        multiindex_single = ["dummy"]*10
        multiindex_multi = ["dummy1"]*5+["dummy2"]*5
        x = numpy.random.rand(10,3)
        self.df0 = pandas.DataFrame(x, columns=base_cols, 
                                index=base_idx)
        self.df1 = pandas.DataFrame(x,
                                    columns=[multicols_single,
                                             base_cols], 
                                    index=base_idx)
        self.df2 = pandas.DataFrame(x, columns=[
                                multicols_multi, base_cols], 
                                index=base_idx)
        self.df3 = pandas.DataFrame(x, columns= base_cols, 
                             index=[multiindex_single, base_idx])
        self.df4 = pandas.DataFrame(x, columns=base_cols, 
                               index= [multiindex_multi, base_idx])
        self.df5 = pandas.DataFrame(x, columns = [multicols_single,
                                                  base_cols],
                               index= [multiindex_multi,base_idx])
        self.df6 = pandas.DataFrame(x, columns=[multicols_multi,
                                                base_cols],
                               index=[multiindex_single, base_idx])
        self.df7 = pandas.DataFrame(x, columns=[multicols_multi, 
                                                base_cols],
                        index=[multiindex_multi, base_idx])
    

    def test_no_multiindex_index(self):
        ind =self.df0.index
        ndf = tidy_multiindex(self.df0)
        self.assertTrue((ind==ndf.index).all())
    
    def test_no_multiindex_cols(self):
        cols = self.df0.columns
        ndf = tidy_multiindex(self.df0)
        self.assertTrue((cols==ndf.columns).all())
    
    def test_multiindex_cols(self):
        ind = self.df1.index
        cols = self.df1.columns
        ndf = tidy_multiindex(self.df1)
        clause1 = all(ind == ndf.index)
        ncols=[f"dummy_level.col{i}" for i in range(1,4)]
        self.assertTrue(clause1 and all(ncols==ndf.columns))

    def test_multiindex_rows(self):
        ind = [f"dummy.{i}" for i in range(10)]
        cols = self.df3.columns
        ndf = tidy_multiindex(self.df3)
        clause = all(cols == ndf.columns)
        self.assertTrue(clause and all(ind==ndf.index))
    
    @unittest.expectedFailure
    def test_multiindex_rows_wrong_sepperator(self):
        ind = [f"dummy*{i}" for i in range(10)]
        cols = self.df3.columns
        ndf = tidy_multiindex(self.df3)
        clause = all(cols == ndf.columns)
        self.assertTrue(clause and all(ind==ndf.index))

    def test_multimulti_index(self):
        cols = self.df4.columns
        ind = [f"dummy1.{i}" for i in range(5)
               ]+[f"dummy2.{i}" for i in range(5,10)]
        ndf = tidy_multiindex(self.df4)
        clause = all(cols == ndf.columns)
        self.assertTrue(
            clause and all(ind==ndf.index)
        )

    def test_multindex_both_index_multilevel(self):
        ind = [f"dummy1.{i}" for i in range(5) ]+[
            f"dummy2.{i}" for i in range(5,10)]
        cols = [f"dummy_level.col{i}" for i in range(1,4)]
        ndf = tidy_multiindex(self.df5)
        self.assertTrue(
            all(ndf.columns == cols) and all(ndf.index == ind)
        )

    def test_multiindex_both_columns_multilevel(self):
        ind = [f"dummy.{i}" for i in range(10)]
        cols = ["dummy_level1.col1"]+[
            f"dummy_level2.col{i}" for i in range(2,4)]
        ndf = tidy_multiindex(self.df6)
        self.assertTrue(
            all(ind == ndf.index) and all(cols==ndf.columns)
        )


    def test_both_multimultiindices(self):
        ndf = tidy_multiindex(self.df7)
        ind = [f"dummy1.{i}" for i in range(5) ]+[
            f"dummy2.{i}" for i in range(5,10)]
        cols = ["dummy_level1.col1"]+[
            f"dummy_level2.col{i}" for i in range(2,4)]
        
        self.assertTrue(
            all(ind==ndf.index) and all(cols==ndf.columns)
        )


    def test_sepperator(self):
        ndf = tidy_multiindex(self.df7, sep = "+")
        ind = [f"dummy1+{i}" for i in range(5) ]+[
            f"dummy2+{i}" for i in range(5,10)]
        cols = ["dummy_level1+col1"]+[
            f"dummy_level2+col{i}" for i in range(2,4)]
        
        self.assertTrue(
            all(ind==ndf.index) and all(cols==ndf.columns)
        )