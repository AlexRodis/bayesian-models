# Testing suite for core model builder module
import unittest
import pymc
import pytensor
import numpy as np
from functools import partial 
from typing import Generator, Callable
from bayesian_models.core import CoreModelComponent, ModelDirector
# Test suite needs work
from bayesian_models.core import FreeVariablesComponent
from bayesian_models.core import LinearRegressionCoreComponent
from bayesian_models.core import Distribution, CoreModelBuilder
from bayesian_models.core import LikelihoodComponent
from bayesian_models.core import ModelAdaptorComponent
from bayesian_models.core import ResponseFunctionComponent
from bayesian_models.core import ResponseFunctions
from bayesian_models.core import NeuralNetCoreComponent
from bayesian_models.data import Data
from bayesian_models.models import Layer
from bayesian_models.utilities import powerset, dict_powerset

def distribution(dist:pymc.Distribution,name:str,
                 *args, **kwargs)->Distribution:
    return Distribution(dist = dist, name = name,
                        dist_args = args, dist_kwargs = kwargs)

# class TestCoreModule(unittest.TestCase):
    
    
#     @classmethod
#     def setUpClass(cls):
#         # Can be accessed via the object as self.X etc
#         X = np.random.rand(50,3)
#         Y = 3.54*X + np.asarray([1.0]*3)
#         Y_cat = np.concatenate([Y[:,[0]], Y[:,[0]]>2], axis=1)
#         cls.X = X
#         cls.Y = Y
#         cls.Y_cat = Y_cat
        
#     def test_dummy(self):
#         data_preprocessor = Data()
#         core_obj = LinearRegressionCoreComponent(
#             distributions = dict(
#                 inputs =Distribution(
#                     dist = pymc.Data, name = 'inputs', 
#                     dist_args = (data_preprocessor(self.X).values(),),
#                     dist_kwargs = dict(mutable=True)
#                     ),
#                 outputs = Distribution(
#                     dist = pymc.Data, name ='outputs',
#                     dist_args = (data_preprocessor(self.Y).values(),), 
#                     dist_kwargs = dict(mutable=True)
#                     ),
#                 W = Distribution(
#                     dist = pymc.Normal, name = 'W', dist_args = tuple(),
#                     dist_kwargs = dict(mu =0.0, sigma=1.0 , shape = 3)
#                     ),
#                 b =Distribution(
#                     dist = pymc.Normal, name = 'b',
#                     dist_kwargs = dict(mu=0.0, sigma=1.0),
#                     dist_args = tuple(),
#                 ),
#             )
#         )
#         free_vars_obj = FreeVariablesComponent(
#             dict(
#                 epsilon = Distribution(
#                     name = 'ε', dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(
#                         sigma = 1
#                     )
#                 )
#             )
#         )
#         likelihood_obj = [LikelihoodComponent(
#             distribution = pymc.Normal,
#             var_mapping = dict(
#                 mu = 'f', sigma = 'epsilon',
#             )
#         )]
#         builder = ModelDirector(
#             core_component = core_obj,
#             free_vars_component = free_vars_obj,
#             likelihood_components = likelihood_obj
#         )
#         m = builder()
#         with m:
#             pymc.sample(chains=2, draws=600, tune=1600)
#         return
    
    
#     def test_best(self):
#         def func(e:float)->str:
#             if e ==0.0:
#                 return "g1"
#             else:
#                 return "g2"
        
#         d_transformer = Data(cast = None)
#         Y = d_transformer(self.Y_cat).values()
#         Y_new = np.asarray([func(e) for e in Y[:,[1]]])[:,None]
#         Y_new = np.concatenate([Y[:,[0]], Y_new], axis = 1)
#         levels = np.unique(Y_new[:,-1]).astype(str)
#         i0 = np.where(Y_new[:,[-1]]=="g1")[0]
#         i1 = np.where(Y_new[:,[-1]]=="g2")[0]
#         core = CoreModelComponent(
#             distributions = {
#                 'inputs_g1' : Distribution(
#                     dist = pymc.Data, name = 'inputs_g1', 
#                     dist_args = (Y_new[i0, 0].astype(np.float32), ),
#                     dist_kwargs = dict(),
#                     ),
#                 'inputs_g2' : Distribution(
#                     dist = pymc.Data, name = 'inputs_g2', 
#                     dist_args = (Y_new[i1, 0].astype(np.float32),),
#                     dist_kwargs = dict(),
#                     ),
#                 'nu' : Distribution(dist = pymc.Exponential, name = 'nu',
#                                     dist_args = (1/29.0,),
#                                     dist_kwargs={}
#                     ),
#                 'mu_g1' : Distribution(
#                     dist = pymc.Normal, name = 'mu_g1',
#                     dist_kwargs = dict(mu = np.mean(
#                     Y_new[i0, 0].astype(np.float32)),
#                                   sigma = np.std(
#                                       Y_new[i0, 0].astype(np.float32) 
#                                       )), dist_args = tuple()
#                     ),
#                 'mu_g2' : Distribution( dist = pymc.Normal, name = 'mu_g2',
#                         dist_kwargs = dict(
#                             mu = np.mean( Y_new[i1, 0].astype(np.float32)),
#                             sigma = np.std(
#                                 Y_new[i1, 0].astype(np.float32)
#                             ),
#                         ), dist_args = tuple(),
#                     ),
#                 'sigma_g1' : Distribution(
#                     dist = pymc.Uniform, name = 'sigma_g1',
#                     dist_args = (0.1, 10), dist_kwargs = dict()
#                     ),
#                 'sigma_g2' : Distribution(
#                     dist = pymc.Uniform, name = 'sigma_g2',
#                     dist_args = (0.1, 10), dist_kwargs = dict(),
#                     ),
#             },
#         ) 
#         like = [
#             LikelihoodComponent(
#                 name = 'y_obs_g1',
#                 observed = 'inputs_g1',
#                 distribution = pymc.StudentT,
#                 var_mapping = {
#                     'mu' : "mu_g1",
#                     'sigma' : "sigma_g1",
#                     'nu' : 'nu'
#                 }
#             ),
#             LikelihoodComponent(
#                 name = 'y_obs_g2',
#                 observed = 'inputs_g2',
#                 distribution = pymc.StudentT,
#                 var_mapping = {
#                     "mu" : "mu_g2",
#                     "sigma" : "sigma_g2",
#                     "nu" : "nu"
#                 }
#             )
#             ]
#         print("Hi")
#         builder = ModelDirector(
#             core_component = core,
#             free_vars_component = None,
#             likelihood_components = like,
#         )
#         m = builder()
        
#     def test_nnet(self):
#         ls = [
#             Layer(3,) for _ in range(4)
#         ]
#         core_component = NeuralNetCoreComponent(
#             distributions = dict(
#                 inputs = Distribution(name="inputs", dist = pymc.Data,
#                                       dist_args = (self.X,), 
#                                       dist_kwargs = dict(mutable=True)
#                 ),
#                 outputs = Distribution(name="outputs", dist = pymc.Data,
#                                       dist_args = (self.Y,), 
#                                       dist_kwargs = dict(mutable=False)
#                 ),
#                 W0 = Distribution(
#                     name = "W0", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = (3,3))
#                 ),
#                 W1 = Distribution(
#                     name = "W1", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = (3,3))
#                                   ),
#                 W2 = Distribution(
#                     name = "W2", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = (3,2))
#                                   ),
#                 b0 = Distribution(
#                     name = "b0", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = 3)
#                 ),
#                 b1 = Distribution(
#                     name = "b1", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = 3)
#                 ),
#                 b2 = Distribution(
#                     name = "b2", dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape = 2)
#                 ),
#             ),
#             n_layers = 3
#         )
#         splitter = lambda i, f: f.T[f.shape[i], ...]
#         adaptor = ModelAdaptorComponent(
#             var_mapping = dict(
#                 mu = partial(splitter, 0),
#                 sigma = partial(splitter, 1)
#             )
#         )
        
#         response = ResponseFunctionComponent(
#             response_function = dict(
#                 f_trans = pymc.math.tanh
#                 )
#         )
        
#         like = LikelihoodComponent(
#             distribution = pymc.Normal,
#             var_mapping = dict(
#                 mu = 'mu',
#                 sigma = 'sigma',
#             )
#         )
        
#         builder = ModelDirector(
#             core_component = core_component,
#             adaptor_component = adaptor,
#             likelihood_components = [like]
            
#         )
#         m = builder()
    
#     def test_missing_components(self):
#         dis = dict(
#                 inputs =Distribution(
#                     dist = pymc.Data, name = 'inputs', 
#                     dist_args = (self.X,),
#                     dist_kwargs = dict(mutable=True)
#                     ),
#                 outputs = Distribution(
#                     dist = pymc.Data, name ='outputs',
#                     dist_args = (self.Y,), 
#                     dist_kwargs = dict(mutable=True)
#                     ),
#                 W = Distribution(
#                     dist = pymc.Normal, name = 'W', dist_args = tuple(),
#                     dist_kwargs = dict(mu =0.0, sigma=1.0 , shape = 3)
#                     ),
#                 b =Distribution(
#                     dist = pymc.Normal, name = 'b',
#                     dist_kwargs = dict(mu=0.0, sigma=1.0),
#                     dist_args = tuple(),
#             )
#         )
#         core_obj = CoreModelComponent(
#             distributions = dis
#         )
#         var = dict(
#                 epsilon = Distribution(
#                     name = 'ε', dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(
#                         sigma = 1
#                     )
#                 )
#             )

#         likelihood_obj = [LikelihoodComponent(
#             distribution = pymc.Normal,
#             var_mapping = dict(
#                 mu = 'f', sigma = 'epsilon',
#             )
#         )]
#         self.assertRaises(ValueError,
#                           CoreModelBuilder, 
#                           likelihoods = LikelihoodComponent,
#                           )
#         self.assertRaises(ValueError,
#                           CoreModelBuilder, 
#                           core_model = core_obj)
        
        
#     def test_distribution(self):
#         # Unclear if we should test this
#         pass
    
#     def test_core(self):
#         core = CoreModelComponent(
#             distributions = dict(
#                 W = Distribution(
#                     name = 'W', dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape=3)
#                 ),
#                 b = Distribution(
#                     name = 'b', dist = pymc.Normal, dist_args = (0,),
#                     dist_kwargs = dict(sigma=1, shape=3)
#                 ),
#             )
#         )
#         with pymc.Model() as testmodel:
#             core()
#         self.assertTrue(set(e.name for e in testmodel.free_RVs)=={
#             "W","b"
#         })
        
#     def test_free_vars(self):
#         freevars = FreeVariablesComponent(
#             dict(
#                 alpha = Distribution(
#                     name="alpha", dist = pymc.StudentT,
#                 dist_kwargs = {"mu":0,"sigma":1,"nu":1
#                                }
#                 ),
#                 beta = Distribution(
#                     name = "beta", dist = pymc.Beta,
#                     dist_kwargs = dict(alpha=1, beta=.5)
#                 ),
#             )
#         )
#         with pymc.Model() as testmodel:
#             freevars()
        
#         self.assertTrue(
#            set(e.name for e in testmodel.free_RVs)=={"alpha","beta"}
#        )
        
#     def test_adaptor(self):
#         X = np.random.rand(100,9,3)
#         tensor = pytensor.shared(X, name="X")
#         core = CoreModelComponent(
#             distributions = dict(
#                 f = Distribution(
#                     name = "f", dist = pymc.Data, dist_args = (
#                          tensor,)
#                     )
#             )
#         )
#         adaptor = ModelAdaptorComponent(
#             var_mapping = dict(
#                 t1 = lambda t: t.T[0,...],
#                 t2 = lambda t: t.T[1,...],
#                 t3 = lambda t: t.T[2,...],
#             ), record=True
#         )
#         with pymc.Model() as testmodel:
#             core()
#             adaptor(testmodel.named_vars["X"])
            
#         self.assertTrue(
#             set(e.name for e in testmodel.deterministics)=={"t1",
#                                                             "t2", "t3"}
#         )
        
#     def test_likelihood(self):
#         core = LinearRegressionCoreComponent(
#             distributions = dict(
#                 inputs = Distribution(
#                     name = "inputs", dist = pymc.Data,
#                     dist_args = (self.X,), dist_kwargs = dict(
#                         mutable=True
#                     )
#                 ),
#                 outputs = Distribution(
#                     name = "outputs", dist = pymc.Data,
#                     dist_args = (self.X*2+1,)
#                 ),
#                 W = Distribution(
#                     name = "W", dist = pymc.Normal, dist_args = (1,),
#                     dist_kwargs = {'sigma' : .1, 'shape': 9}
#                     ),
#                 b = Distribution(
#                     name = "b", dist = pymc.Normal, 
#                     dist_kwargs = dict(mu = 0, sigma=1, shape=9)
#                 )
#             )
#         )
        
#         free_vars = FreeVariablesComponent(
#             dict(
#                 ε = Distribution(
#                     name = 'ε', dist = pymc.Normal, 
#                     dist_args = (0,1) 
#                 ),
#                 ν = Distribution(
#                     name = 'ν', dist = pymc.Exponential,
#                     dist_args = (1/29.0,),
#                     dist_transform = lambda e: e+1
#                 )
#             )
#         )
        
#         like = LikelihoodComponent(
#             distribution =  pymc.StudentT,
#             var_mapping = dict(
#                 mu = 'f', sigma = 'ε' , nu = 'ν'
#             )
#         )
        
#         builder = CoreModelBuilder(
#             core_model = core,
#             free_vars = free_vars,
#             likelihoods = [like],
#         )
#         m = builder()
#         self.assertTrue(
#             'y_obs' not in set(e.name for e in m.unobserved_RVs)
#         )
        
#     def test_missing_observations_error(self):
#         core = LinearRegressionCoreComponent(
#             distributions = dict(
#                 inputs = Distribution(
#                     name = "inputs", dist = pymc.Data,
#                     dist_args = (self.X,), dist_kwargs = dict(
#                         mutable=True
#                     )
#                 ),
#                 W = Distribution(
#                     name = "W", dist = pymc.Normal, dist_args = (1,),
#                     dist_kwargs = {'sigma' : .1, 'shape': 9}
#                     ),
#                 b = Distribution(
#                     name = "b", dist = pymc.Normal, 
#                     dist_kwargs = dict(mu = 0, sigma=1, shape=9)
#                 )
#             )
#         )
        
#         free_vars = FreeVariablesComponent(
#             dict(
#                 ε = Distribution(
#                     name = 'ε', dist = pymc.Normal, 
#                     dist_args = (0,1) 
#                 ),
#                 ν = Distribution(
#                     name = 'ν', dist = pymc.Exponential,
#                     dist_args = (1/29.0,),
#                     dist_transform = lambda e: e+1
#                 )
#             )
#         )
        
#         like = LikelihoodComponent(
#             distribution =  pymc.StudentT,
#             var_mapping = dict(
#                 mu = 'f', sigma = 'ε' , nu = 'ν'
#             )
#         )
        
#         builder = CoreModelBuilder(
#             core_model = core,
#             free_vars = free_vars,
#             likelihoods = [like],
#         )
#         self.assertRaises(
#             RuntimeError, builder.__call__
#         )
            
#     def test_input_names(self):
#         core = LinearRegressionCoreComponent(
#             distributions = dict(
#                 X = Distribution(
#                     name = "X", dist = pymc.Data,
#                     dist_args = (self.X,), dist_kwargs = dict(
#                         mutable=True
#                     )
#                 ),
#                 Y = Distribution(
#                     name = "Y", dist = pymc.Data,
#                     dist_args = (self.X*2+1,)
#                 ),
#                 W = Distribution(
#                     name = "W", dist = pymc.Normal, dist_args = (1,),
#                     dist_kwargs = {'sigma' : .1, 'shape': 9}
#                     ),
#                 b = Distribution(
#                     name = "b", dist = pymc.Normal, 
#                     dist_kwargs = dict(mu = 0, sigma=1, shape=9)
#                 )
#             )
#         )
        
#         free_vars = FreeVariablesComponent(
#             dict(
#                 ε = Distribution(
#                     name = 'ε', dist = pymc.Normal, 
#                     dist_args = (0,1) 
#                 ),
#                 ν = Distribution(
#                     name = 'ν', dist = pymc.Exponential,
#                     dist_args = (1/29.0,),
#                     dist_transform = lambda e: e+1
#                 )
#             )
#         )
        
#         like = LikelihoodComponent(
#             distribution =  pymc.StudentT,
#             var_mapping = dict(
#                 mu = 'f', sigma = 'ε' , nu = 'ν'
#             )
#         )
        
#         builder = CoreModelBuilder(
#             core_model = core,
#             free_vars = free_vars,
#             likelihoods = [like],
#         )
#         self.assertRaises(
#             RuntimeError, builder.__call__
#         )

class TestFramework(unittest.TestCase):
    
    @classmethod    
    def setUpClass(cls):
        # Can be accessed via the object as self.X etc
        X = np.random.rand(50,3)
        Y = 3.54*X + np.asarray([1.0]*3)
        Y_cat = np.concatenate([Y[:,[0]], Y[:,[0]]>2], axis=1)
        cls.X = X
        cls.Y = Y
        cls.Y_cat = Y_cat

class TestCoreComponent(TestFramework):
    
    def test_vars_added(self):
        from itertools import chain, combinations

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(
                combinations(s, r) for r in range(len(s)+1)
                )
        dists = dict(
            inputs = Distribution(
                name = "inputs", dist = pymc.Data, 
                dist_args = (self.X,)
                ),
            W = Distribution(
                name = "W", dist = pymc.Normal, dist_args = (1,),
                dist_kwargs = dict(sigma=.1)
            ),
            w = Distribution(
                name = "w", dist = pymc.Normal, dist_args = (1,.1),
            ),
            b = Distribution(
                name = "b", dist = pymc.Beta,
                dist_kwargs = dict(alpha=1, beta=5)
            ),
            a = Distribution(
                name = "a", dist = pymc.Gamma,
                dist_kwargs = dict(alpha=1, beta=5)
            ),
            c = Distribution(
                name = "c", dist = pymc.StudentT,
                dist_kwargs = dict(sigma=.1, mu = 5, nu=1.5)
            ),
            Mv = Distribution(
                name = "Mv", dist = pymc.MvStudentT,
                dist_kwargs = dict(
                    nu=1, mu = np.asarray([1,2,3,4,5]), 
                    scale = np.diag(np.asarray([5,7,1,5,4])) 
                    )
            )
        )
        # Subtests need to go here
        param_pset = map(dict, 
                         powerset(dists.items())
                         )
        for param in param_pset:
            with pymc.Model() as testmodel:
                if param != dict():
                    CoreModelComponent(
                    distributions = param
                    )()
                self.assertTrue(
                   set(testmodel.named_vars)^set(param.keys() ) == set()
                )
    
    def test_illegal_inputs(self):
        illegals = [
            [45,7], [pymc.Normal], 5, "bobo", partial(pymc.Normal),
            pymc.Normal,
        ]
        for illegal in illegals:
            
            self.assertRaises(
                ValueError, CoreModelComponent,
                distributions = illegal
            )


    def test_blank_inputs(self):
        self.assertRaises(
            ValueError, CoreModelComponent
        )
        self.assertRaises(
            ValueError, CoreModelComponent,
                distributions = dict()
            
        )

class TestLikelihoodComponent(TestFramework):
    
    def test_no_var_mapping(self):
        like = LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(mu='m', sigma= 's')
        )
        
        modelvars = dict()
        with pymc.Model() as testmodel:
            inpts = pymc.Data('inpts', self.X)
            m = pymc.Normal("m",0,1)
            s = pymc.Normal("s", 1,.1)
            modelvars['m'], modelvars['s'] = m,s
        
        var_mapping = {
            k:modelvars[v] for k,v in like.var_mapping.items()
        }
        with testmodel:
            self.assertRaises(
                ValueError, like.__call__,
                inpts
            )
    
    def test_no_observed(self):
        like = LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(mu='m', sigma= 's')
        )
        
        modelvars = dict()
        with pymc.Model() as testmodel:
            inpts = pymc.Data('inpts', self.X)
            m = pymc.Normal("m",0,1)
            s = pymc.Normal("s", 1,.1)
            modelvars['m'], modelvars['s'] = m,s
        
        var_mapping = {
            k:modelvars[v] for k,v in like.var_mapping.items()
        }
        with testmodel:
            self.assertRaises(
                RuntimeError, like.__call__,
                None, **var_mapping 
            )

    def test_like(self):
        like = LikelihoodComponent(
            distribution = pymc.MvStudentT,
            var_mapping = dict(
                mu = 'f', scale = 'cov', nu = 'ν', 
            )
        )
        modelvars = dict()
        with pymc.Model() as testmodel:
            inpts = pymc.Data('inpts', self.X)
            modelvars['inputs'] = inpts
            w = pymc.Normal('w', 0,1, shape=3)
            modelvars['w'] = w
            cov = pymc.Data('cov', np.diag(
                np.asarray([1,4,6])
                                        ), mutable=False)
            modelvars['cov'] = cov
            ν = pymc.Exponential('ν', 1/29.0)+1
            modelvars['ν'] = ν
            f = pymc.Deterministic('f',inpts*w+10)
            modelvars['f']=f
        kwargs = {
            k:modelvars[v]  for k,v in like.var_mapping.items()
            }
        with testmodel:
            like(modelvars['inputs'],
                 **kwargs)

class TestResponseFunctions(TestFramework):
    
    @classmethod
    def setUpClass(cls):
        def custom_func(X):
            return X
        
        def func_void(X):
            pass
        
        function_catalogue = dict(
            exp = pymc.math.exp,
            softmax = pymc.math.softmax,
            sigmoid = pymc.math.invlogit,
            tanh = pymc.math.tanh,
            custom = custom_func
        )
        cls.functionlib = function_catalogue
        cls.custom_func = custom_func
        cls.func_void = func_void
    
    def test_empty_call(self):
        self.assertRaises(
            ValueError, ResponseFunctions
        )
    
    def test_basic_functionality(self):
        obj = ResponseFunctions(
            functions = { 
                         f"f_{k}":v for k,v in self.functionlib.items()}
            ,
            application_targets = {
                f"f_{k}":"f" for k,v in self.functionlib.items()
                },
            records = {
                f"f_{k}":True for k,v in self.functionlib.items()
                },
        )
        for function, call in self.functionlib.items():
            functuple = obj.get_function(f"f_{function}")
            self.assertTrue(
                all([
                    functuple.name == f"f_{function}",
                    functuple.func == call,
                    functuple.record,
                    functuple.target == "f"
                    
                ])
            )
    
    def test_power_inputs(self):
        # Generate all possible 'subsets' of a function catalogue
        from itertools import product
        dset:Generator[dict[str, Callable], None, None] = (
            e for i,e in enumerate(
                dict_powerset(self.functionlib)
                ) if i!=0
            )
        # Generate all possible combinations of possible targets
        # and deterministics
        params = product(*[
         [True, False],
         ["f", "f_star", "gp", "gpf"],
         ])
        for inptdict in dset:
            for param in params:
                obj = ResponseFunctions(
                    functions = {
                        f"f_{k}":v for k,v in inptdict.items()
                    },
                    application_targets = {
                        f"f_{k}":param[1] for k in inptdict.keys()
                    },
                    records = {f"f_{k}":param[0] for k in inptdict.keys()}
                )
                for func, call in inptdict.items():
                    functuple = obj.get_function(f"f_{func}")
                    self.assertTrue(
                        all([
                            functuple.name == f"f_{func}",
                            functuple.func == call,
                            functuple.record==param[0],
                            functuple.target == param[1]
                            
                        ])
                )
            
    def test_incoherent_inputs(self):
        '''
            Testing cases of incoherent inputs. `functions` is assumed
            to be the most basic argument. Any value specified in either
            `records` or `application_targets` cannot be reasobly infered
            and should raise
        '''
        for i in range(len(self.functionlib)):
            newlib = {k:v for j,(k,v) in enumerate(
                self.functionlib.items()
                ) if i!=j} # Exclude eactly one entry for all possible entries
            invalid_keys = self.functionlib.keys()
            valid_keys = newlib.keys()
            self.assertRaises(ValueError, ResponseFunctions,
                functions = newlib,
                records = {k:True for k in invalid_keys},
                application_targets = {k:"f" for k in valid_keys}              
            )
            self.assertRaises(ValueError, ResponseFunctions,
                functions = newlib,
                records = {k:True for k in valid_keys},
                application_targets = {k:"f" for k in invalid_keys}              
            )
            self.assertRaises(ValueError, ResponseFunctions,
                functions = newlib,
                records = {k:True for k in invalid_keys},
                application_targets = {k:"f" for k in invalid_keys}              
            )
    
    def test_infered_parameters(self):
        '''
            Test common use case where targets or records can be omitted
            and sensible defaults can be used
        '''
        for i in range(len(self.functionlib)):
            newlib = {k:v for j,(k,v) in enumerate(
                self.functionlib.items()
                ) if i!=j} # Exclude eactly one entry for all possible entries
            invalid_keys = self.functionlib.keys()
            valid_keys = newlib.keys()
            obj_padded_recs = ResponseFunctions(
                functions = self.functionlib,
                records = {k:True for k in valid_keys},
                application_targets = {k:"f" for k in invalid_keys}              
            )
            obj_padded_targets = ResponseFunctions(
                functions = self.functionlib,
                records = {k:True for k in invalid_keys},
                application_targets = {k:"f" for k in valid_keys}              
            )
            obj_both_padded = ResponseFunctions(
                functions = self.functionlib,
                records = {k:True for k in valid_keys},
                application_targets = {k:"f" for k in valid_keys}              
            )
            obj_complete = ResponseFunctions(
                functions = newlib,
                records = {k:True for k in valid_keys},
                application_targets = {k:"f" for k in valid_keys}              
            )
            
            condictions = dict(
                recpad_records = obj_padded_recs.records == {
                    k:True for k in self.functionlib.keys()},
                recpad_missing = obj_padded_recs._missing_records == set(
                    self.functionlib.keys()) - set(newlib.keys()),
                recpad_notmissing = obj_padded_recs._missing_targets == set(),
                tarpad_targets = obj_padded_targets.application_targets == {
                    k:"f" for k in self.functionlib.keys()},
                tarpad_missing = obj_padded_targets._missing_targets == set(
                    self.functionlib.keys()) - set(newlib.keys()),
                tarpad_notmissing = obj_padded_targets._missing_records == set(),
                complete_recs_notmissing = obj_complete._missing_targets==set(),
                complete_tars_notmissing = obj_complete._missing_records==set(),
                complete_recs = obj_complete.records == {
                    k:True for k in newlib.keys()},
                complete_targets = obj_complete.application_targets == {
                    k:"f" for k in newlib.keys()},
                both_missing_rec = obj_both_padded.records == {
                    k:True for k in self.functionlib.keys()},
                both_missing_targets = obj_both_padded.application_targets == {
                    k:"f" for k in self.functionlib.keys()
                    },
                both_missing_missing_recs = obj_both_padded._missing_records == set(
                    self.functionlib.keys())-set(newlib.keys()),
                both_missing_missing_tars = obj_both_padded._missing_targets == set(
                    self.functionlib.keys())-set(newlib.keys()),
            )
            self.assertTrue(all(list(
                e for _,e in condictions.items()
            )))
            
    def test_iterator(self):
        obj = ResponseFunctions(
            functions = self.functionlib
        )
        for (functuple, func) in zip(obj, self.functionlib.items(), 
                                     strict=True):
            
            predicates = dict(
                name = functuple.name == func[0],
                function = functuple.func == func[1],
                target = functuple.target == "f",
                record = functuple.record
                
            )
            self.assertTrue(all([
                k for k in predicates.keys()         
            ]))
            
        
class TestResponseComponent(TestResponseFunctions):
        
    def test_insertions(self):
        # Skip null set
        funclib = (f for i,f in enumerate(
            dict_powerset(self.functionlib)) if i!=0)
        for lib in funclib:
            var_names = (f"f_{k}" for k in lib.keys())
            component = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = {
                        f"f_{k}":v for k, v in lib.items()
                        }
                )
            )
            modelvars = dict()
            X = np.random.rand(100,3)
            with pymc.Model() as model:
                W = pymc.Normal('W', mu = [1]*3, sigma=[.1]*3)
                f = pymc.Deterministic('f', pymc.math.dot(X, W)+15)
                modelvars['W'], modelvars['f'] = W,f
                component(modelvars)
            s1 = set(
                    e.name for e in model.deterministics if e.name != 'f')
            s2 = set(var_names)
            c = s1==s2
            self.assertTrue(c
            )

    def test_var_catalogue(self):
        from itertools import product
        # Skip null set
        funclib = (f for i,f in enumerate(
            dict_powerset(self.functionlib)) if i!=0)
        params = product([True,False], funclib)
        for (bln, lib) in params:
            var_names = (f"f_{k}" for k in lib.keys())
            component = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = {
                        f"f_{k}":v for k, v in lib.items()
                        },
                    records = {
                        f"f_{k}":bln for i, (k, v) in enumerate(
                            lib.items())
                        },
                )
            )
            X = np.random.rand(100,3)
            with pymc.Model() as model:
                W = pymc.Normal('W', mu = [1]*3, sigma=[.1]*3)
                f = pymc.Deterministic('f', pymc.math.dot(X, W)+15)
                component(dict(
                    W = W,
                    f = f
                ))
            self.assertTrue(
                set(component.variables.keys()) == set(
                    f"f_{k}" for k in lib.keys())
            )
            
    def test_invalid_modelspecs(self):
        component = ResponseFunctionComponent(
            ResponseFunctions(
                functions = {
                    f"f_{k}":v for k, v in self.functionlib.items()
                    },
                records = {
                    f"f_{k}":True for i, (k, v) in enumerate(
                        self.functionlib.items())
                    },
            )
        )
        X = np.random.rand(100,3)
        with pymc.Model() as model:
            W = pymc.Normal('W', mu = [1]*3, sigma=[.1]*3)
            f = pymc.Deterministic('f', pymc.math.dot(X, W)+15)
            self.assertRaises(
                ValueError, component.__call__, None
            )
            illegals = [
                ["f","f_star"], True, 15,ResponseFunctions(
                functions = {
                    f"f_{k}":v for k, v in self.functionlib.items()
                    },
                records = {
                    f"f_{k}":True for i, (k, v) in enumerate(
                        self.functionlib.items())
                    },
            )
                ]
            for illegal in illegals:
                self.assertRaises(
                    ValueError, component.__call__, illegal
                )
            self.assertRaises(
                RuntimeError, component.__call__, dict(
                    W = W
                )
            )

            
class TestLink(TestFramework):
    pass

class TestFreeVars(TestFramework):
    pass

class TestBuilder(TestFramework):
    pass

class TestCompositeModel(TestFramework):
    pass
    