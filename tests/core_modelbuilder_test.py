import unittest
import pymc
import pytensor
import numpy as np
from functools import partial 
from bayesian_models.core import CoreModel, ModelDirector
from bayesian_models.core import FreeVariables, Likelihood, Distribution
from bayesian_models.core import LinearRegressionCoreComponent
from bayesian_models.data import Data



class TestCoreModule(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Can be accessed via the object as self.X etc
        X = np.random.rand(50,3)
        Y = 3.54*X + np.asarray([1.0]*3)
        Y_cat = np.concatenate([Y[:,[0]], Y[:,[0]]>2], axis=1)
        cls.X = X
        cls.Y = Y
        cls.Y_cat = Y_cat
        
    def test_dummy(self):
        data_preprocessor = Data()
        core_obj = LinearRegressionCoreComponent(
            variables = dict(
                inputs = partial(pymc.Data, 'inputs', 
                                 data_preprocessor(self.X).values(), mutable=True),
                outputs = partial(pymc.Data, 'outputs',
                                  data_preprocessor(self.Y).values(), mutable=True),
                W = partial(pymc.Normal, 'W',mu =0.0, sigma=1.0 , shape = 3),
                b = partial(pymc.Normal, 'b', mu=0.0, sigma=1.0),
            )
        )
        free_vars_obj = FreeVariables(
            [Distribution(
                name = 'ε', dist = pymc.Normal, dist_args = (0,),
                dist_kwargs = dict(
                    sigma = 1
                )
            )]
        )
        likelihood_obj = [Likelihood(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'f', sigma = 'ε',
            )
        )]
        builder = ModelDirector(
            core_component = core_obj,
            free_vars_component = free_vars_obj,
            likelihood_components = likelihood_obj
        )
        m = builder()
        with m:
            pymc.sample(chains=2, draws=600, tune=1600)
        return
    
    
    def test_best(self):
        def func(e:float)->str:
            if e ==0.0:
                return "g1"
            else:
                return "g2"
        
        d_transformer = Data(cast = None)
        Y = d_transformer(self.Y_cat).values()
        Y_new = np.asarray([func(e) for e in Y[:,[1]]])[:,None]
        Y_new = np.concatenate([Y[:,[0]], Y_new], axis = 1)
        levels = np.unique(Y_new[:,-1]).astype(str)
        i0 = np.where(Y_new[:,[-1]]=="g1")[0]
        i1 = np.where(Y_new[:,[-1]]=="g2")[0]
        core = CoreModel(
            variables = {
                'inputs_g1' : partial(
                    pymc.Data, 'inputs_g1', Y_new[i0, 0].astype(np.float32)
                    ),
                'inputs_g2' : partial(
                    pymc.Data,'inputs_g2', Y_new[i1, 0].astype(np.float32)
                    ),
                'nu' : partial(pymc.Exponential, 'nu',1/29.0),
                'mu_g1' : partial(pymc.Normal, 'mu_g1',mu = np.mean(
                    Y_new[i0, 0].astype(np.float32)),
                                  sigma = np.std(
                                      Y_new[i0, 0].astype(np.float32) 
                                      )
                                  ),
                'mu_g2' : partial(pymc.Normal, 'mu_g2',
                    mu = np.mean( Y_new[i1, 0].astype(np.float32)
                    ),
                    sigma = np.std(
                        Y_new[i1, 0].astype(np.float32)
                        )
                    ),
                'sigma_g1' : partial(pymc.Uniform, 'sigma_g1',0.1, 10),
                'sigma_g2' : partial(pymc.Uniform, 'sigma_g2',0.1, 10),
            },
        ) 
        like = [
            Likelihood(
                name = 'y_obs_g1',
                observed = 'inputs_g1',
                distribution = pymc.StudentT,
                var_mapping = {
                    'mu' : "mu_g1",
                    'sigma' : "sigma_g1",
                    'nu' : 'nu'
                }
            ),
            Likelihood(
                name = 'y_obs_g2',
                observed = 'inputs_g2',
                distribution = pymc.StudentT,
                var_mapping = {
                    "mu" : "mu_g2",
                    "sigma" : "sigma_g2",
                    "nu" : "nu"
                }
            )
            ]
        print("Hi")
        builder = ModelDirector(
            core_component = core,
            free_vars_component = None,
            likelihood_components = like,
        )
        m = builder()
        
    def test_nnet(self):
        pass