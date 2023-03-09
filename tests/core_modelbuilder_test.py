# Testing suite for core model builder module
import unittest
import pymc
import pytensor
import numpy as np
from functools import partial 
from bayesian_models.core import CoreModelComponent, ModelDirector
from bayesian_models.core import FreeVariablesComponent
from bayesian_models.core import LinearRegressionCoreComponent
from bayesian_models.core import Distribution, CoreModelBuilder
from bayesian_models.core import LikelihoodComponent
from bayesian_models.core import ModelAdaptorComponent
from bayesian_models.core import ResponseFunctionComponent
from bayesian_models.core import NeuralNetCoreComponent
from bayesian_models.data import Data
from bayesian_models.models import Layer



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
            distributions = dict(
                inputs =Distribution(
                    dist = pymc.Data, name = 'inputs', 
                    dist_args = (data_preprocessor(self.X).values(),),
                    dist_kwargs = dict(mutable=True)
                    ),
                outputs = Distribution(
                    dist = pymc.Data, name ='outputs',
                    dist_args = (data_preprocessor(self.Y).values(),), 
                    dist_kwargs = dict(mutable=True)
                    ),
                W = Distribution(
                    dist = pymc.Normal, name = 'W', dist_args = tuple(),
                    dist_kwargs = dict(mu =0.0, sigma=1.0 , shape = 3)
                    ),
                b =Distribution(
                    dist = pymc.Normal, name = 'b',
                    dist_kwargs = dict(mu=0.0, sigma=1.0),
                    dist_args = tuple(),
                ),
            )
        )
        free_vars_obj = FreeVariablesComponent(
            dict(
                epsilon = Distribution(
                    name = 'ε', dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(
                        sigma = 1
                    )
                )
            )
        )
        likelihood_obj = [LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'f', sigma = 'epsilon',
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
        core = CoreModelComponent(
            distributions = {
                'inputs_g1' : Distribution(
                    dist = pymc.Data, name = 'inputs_g1', 
                    dist_args = (Y_new[i0, 0].astype(np.float32), ),
                    dist_kwargs = dict(),
                    ),
                'inputs_g2' : Distribution(
                    dist = pymc.Data, name = 'inputs_g2', 
                    dist_args = (Y_new[i1, 0].astype(np.float32),),
                    dist_kwargs = dict(),
                    ),
                'nu' : Distribution(dist = pymc.Exponential, name = 'nu',
                                    dist_args = (1/29.0,),
                                    dist_kwargs={}
                    ),
                'mu_g1' : Distribution(
                    dist = pymc.Normal, name = 'mu_g1',
                    dist_kwargs = dict(mu = np.mean(
                    Y_new[i0, 0].astype(np.float32)),
                                  sigma = np.std(
                                      Y_new[i0, 0].astype(np.float32) 
                                      )), dist_args = tuple()
                    ),
                'mu_g2' : Distribution( dist = pymc.Normal, name = 'mu_g2',
                        dist_kwargs = dict(
                            mu = np.mean( Y_new[i1, 0].astype(np.float32)),
                            sigma = np.std(
                                Y_new[i1, 0].astype(np.float32)
                            ),
                        ), dist_args = tuple(),
                    ),
                'sigma_g1' : Distribution(
                    dist = pymc.Uniform, name = 'sigma_g1',
                    dist_args = (0.1, 10), dist_kwargs = dict()
                    ),
                'sigma_g2' : Distribution(
                    dist = pymc.Uniform, name = 'sigma_g2',
                    dist_args = (0.1, 10), dist_kwargs = dict(),
                    ),
            },
        ) 
        like = [
            LikelihoodComponent(
                name = 'y_obs_g1',
                observed = 'inputs_g1',
                distribution = pymc.StudentT,
                var_mapping = {
                    'mu' : "mu_g1",
                    'sigma' : "sigma_g1",
                    'nu' : 'nu'
                }
            ),
            LikelihoodComponent(
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
        ls = [
            Layer(3,) for _ in range(4)
        ]
        core_component = NeuralNetCoreComponent(
            distributions = dict(
                inputs = Distribution(name="inputs", dist = pymc.Data,
                                      dist_args = (self.X,), 
                                      dist_kwargs = dict(mutable=True)
                ),
                outputs = Distribution(name="outputs", dist = pymc.Data,
                                      dist_args = (self.Y,), 
                                      dist_kwargs = dict(mutable=False)
                ),
                W0 = Distribution(
                    name = "W0", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = (3,3))
                ),
                W1 = Distribution(
                    name = "W1", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = (3,3))
                                  ),
                W2 = Distribution(
                    name = "W2", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = (3,2))
                                  ),
                b0 = Distribution(
                    name = "b0", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = 3)
                ),
                b1 = Distribution(
                    name = "b1", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = 3)
                ),
                b2 = Distribution(
                    name = "b2", dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape = 2)
                ),
            ),
            n_layers = 3
        )
        splitter = lambda i, f: f.T[f.shape[i], ...]
        adaptor = ModelAdaptorComponent(
            var_mapping = dict(
                mu = partial(splitter, 0),
                sigma = partial(splitter, 1)
            )
        )
        
        response = ResponseFunctionComponent(
            response_function = dict(
                f_trans = pymc.math.tanh
                )
        )
        
        like = LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'mu',
                sigma = 'sigma',
            )
        )
        
        builder = ModelDirector(
            core_component = core_component,
            adaptor_component = adaptor,
            likelihood_components = [like]
            
        )
        m = builder()
    
    def test_splines(self):
        # with pm.Model(rng_seeder=RANDOM_SEED) as m4_7:
        #     a = pm.Normal("a", 100, 5)
        #     w = pm.Normal("w", mu=0, sd=3, shape=B.shape[1])
        #     mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
        #     sigma = pm.Exponential("sigma", 1)
        #     D = pm.Normal("D", mu, sigma, observed=d2.doy)
        pass
    
    def test_missing_components(self):
        dis = dict(
                inputs =Distribution(
                    dist = pymc.Data, name = 'inputs', 
                    dist_args = (self.X,),
                    dist_kwargs = dict(mutable=True)
                    ),
                outputs = Distribution(
                    dist = pymc.Data, name ='outputs',
                    dist_args = (self.Y,), 
                    dist_kwargs = dict(mutable=True)
                    ),
                W = Distribution(
                    dist = pymc.Normal, name = 'W', dist_args = tuple(),
                    dist_kwargs = dict(mu =0.0, sigma=1.0 , shape = 3)
                    ),
                b =Distribution(
                    dist = pymc.Normal, name = 'b',
                    dist_kwargs = dict(mu=0.0, sigma=1.0),
                    dist_args = tuple(),
            )
        )
        core_obj = CoreModelComponent(
            distributions = dis
        )
        var = dict(
                epsilon = Distribution(
                    name = 'ε', dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(
                        sigma = 1
                    )
                )
            )

        likelihood_obj = [LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'f', sigma = 'epsilon',
            )
        )]
        self.assertRaises(ValueError,
                          CoreModelBuilder, 
                          likelihoods = LikelihoodComponent,
                          )
        self.assertRaises(ValueError,
                          CoreModelBuilder, 
                          core_model = core_obj)
        
        
    def test_distribution(self):
        # Unclear if we should test this
        pass
    
    def test_core(self):
        core = CoreModelComponent(
            distributions = dict(
                W = Distribution(
                    name = 'W', dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape=3)
                ),
                b = Distribution(
                    name = 'b', dist = pymc.Normal, dist_args = (0,),
                    dist_kwargs = dict(sigma=1, shape=3)
                ),
            )
        )
        with pymc.Model() as testmodel:
            core()
        self.assertTrue(set(e.name for e in testmodel.free_RVs)=={
            "W","b"
        })
        
    def test_free_vars(self):
        freevars = FreeVariablesComponent(
            dict(
                alpha = Distribution(
                    name="alpha", dist = pymc.StudentT,
                dist_kwargs = {"mu":0,"sigma":1,"nu":1
                               }
                ),
                beta = Distribution(
                    name = "beta", dist = pymc.Beta,
                    dist_kwargs = dict(alpha=1, beta=.5)
                ),
            )
        )
        with pymc.Model() as testmodel:
            freevars()
        
        self.assertTrue(
           set(e.name for e in testmodel.free_RVs)=={"alpha","beta"}
       )
        
    def test_adaptor(self):
        X = np.random.rand(100,9,3)
        tensor = pytensor.shared(X, name="X")
        core = CoreModelComponent(
            distributions = dict(
                f = Distribution(
                    name = "f", dist = pymc.Data, dist_args = (
                         tensor,)
                    )
            )
        )
        adaptor = ModelAdaptorComponent(
            var_mapping = dict(
                t1 = lambda t: t.T[0,...],
                t2 = lambda t: t.T[1,...],
                t3 = lambda t: t.T[2,...],
            ), record=True
        )
        with pymc.Model() as testmodel:
            core()
            adaptor(testmodel.named_vars["X"])
            
        self.assertTrue(
            set(e.name for e in testmodel.deterministics)=={"t1",
                                                            "t2", "t3"}
        )
        
    def test_likelihood(self):
        pass
            
