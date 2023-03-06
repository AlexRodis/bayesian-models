import unittest
import pymc
import pytensor
import numpy as np
from functools import partial 
from bayesian_models.core import CoreModel, ModelDirector, \
    FreeVariables, Likelihood


class TestCoreModule(unittest.TestCase):
    
    def setUp(self):
        X = np.random.rand(50,3)
        Y = 3.54*X+ np.asarray([1.0]*3)
        self.X = X
        self.Y = Y
        with pymc.Model() as model:
            inputs = pymc.Data('inputs', X, mutable=True)
            outputs = pymc.Data('outputs', Y, mutable=True)
            W = pymc.Normal('W',mu =0.0, sigma=1.0 , shape = 3)
            b = pymc.Normal('b', mu=0.0, sigma=1.0)
            f = pymc.Deterministic('f', inputs*W+b)
            ε = pymc.Normal('ε', 0,1) 
            y_obs = pymc.Normal('y_obs', observed=outputs, 
                                mu = f, sigma=ε)
            
            
    def test_dummy(self):
        core_obj = CoreModel(
            variables = dict(
                inputs = partial(pymc.Data, 'inputs', self.X, mutable=True),
                outputs = partial(pymc.Data, 'outputs', self.Y, mutable=True),
                W = partial(pymc.Normal, 'W',mu =0.0, sigma=1.0 , shape = 3),
                b = partial(pymc.Normal, 'b', mu=0.0, sigma=1.0),
            )
        )
        free_vars_obj = FreeVariables(
            variables = dict(
                ε = partial(pymc.Normal, mu = 0, sigma = 1, shape=3),
            )
        )
        likelihood_obj = Likelihood(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'f', sigma = 'ε',
            )
        )
        builder = ModelDirector(
            core_component = core_obj,
            free_vars_component = free_vars_obj,
            likelihood_component = likelihood_obj
        )
        m = builder()
        return
        
        
        
        
        
            
        