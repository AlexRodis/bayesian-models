#   Copyright 2023 Alex Rodis
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Testing suite for core model builder module

import unittest
import pymc
import pytensor
import numpy as np
from functools import partial 
from typing import Generator, Callable, Optional
from bayesian_models.core import CoreModelComponent, ModelDirector
# Test suite needs work
from bayesian_models.core import FreeVariablesComponent
from bayesian_models.core import LinearRegressionCoreComponent
from bayesian_models.core import Distribution, CoreModelBuilder
from bayesian_models.core import LikelihoodComponent, distribution
from bayesian_models.core import ModelAdaptorComponent
from bayesian_models.core import ResponseFunctionComponent
from bayesian_models.core import ResponseFunctions
from bayesian_models.data import Data
from bayesian_models.utilities import dict_powerset


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


class TestDistribution(unittest.TestCase):
    
    def test_basic_run(self):
        call_ref = lambda d: d+2
        def tuple_arr_comp(one:tuple[np.ndarray], 
                           other:tuple[np.ndarray])->bool:
            conds = []
            conds.append(
                type(one)==type(other)
            )
            conds.append(
                len(one)==len(other)
            )
            conds.append( all([all(e1==e2) for e1,e2 in zip(one,other)])
            )
            return all(conds) 
            
        obj1 = Distribution(
            name = 'd1', dist = pymc.Categorical,
            dist_args = (np.asarray([0.1,.5,.4]),)
        )
        
        obj2 = Distribution(
            name = 'd2', dist = pymc.Beta,
            dist_kwargs = dict(alpha=1.0, beta=1.0)
        )
        obj3 = Distribution(
            name = 'd3', dist = pymc.MvStudentT,
            dist_kwargs = dict(mu =np.asarray([0.1]*3),
                               sigma = np.diag(np.asarray(
                                   [3.0]*3)),
                               nu = 5)
        )
        obj4 = Distribution(
            name = 'd4', dist = pymc.Normal,
            dist_args = (0,), dist_kwargs = {'sigma':5}
        )
        
        obj5 = Distribution(
            name = 'd5', dist = pymc.Exponential,
            dist_args = (1/29,), dist_transform = call_ref
         )
        
        predicates = dict(
            discreet = all([
                obj1.name == 'd1',
                obj1.dist == pymc.Categorical,
                tuple_arr_comp(obj1.dist_args ,
                               (np.asarray([0.1,.5,.4]),)),
                obj1.dist_kwargs == dict(),
                obj1.dist_transform is None
                ]),
            
            cont_kw_only = all([
                obj2.name == 'd2',
                obj2.dist == pymc.Beta,
                obj2.dist_args == tuple(),
                obj2.dist_kwargs == dict(alpha=1.0, beta=1.0),
                obj2.dist_transform is None
                ]),
            
            mutivariate = all([
                obj3.name == 'd3',
                obj3.dist == pymc.MvStudentT,
                obj3.dist_args == tuple(),
                all([
                    obj3.dist_kwargs['nu'] == 5,
                    all(obj3.dist_kwargs['mu'] == np.asarray([0.1]*3)),
                ]),
                obj3.dist_transform is None
                ]),
            kw_arg_mix = all([
                obj4.name == 'd4',
                obj4.dist == pymc.Normal,
                obj4.dist_args == (0,),
                obj4.dist_kwargs == dict(sigma=5),
                obj4.dist_transform is None
                ]),
            transformed = all([
                obj5.name == 'd5',
                obj5.dist == pymc.Exponential,
                obj5.dist_args == (1/29,),
                obj5.dist_kwargs == dict(),
                obj5.dist_transform == call_ref
                ]),
        )
        
        self.assertTrue(
            all([
                v for _,v in predicates.items()
            ])
        )
        
    def test_factory(self):
        call_ref = lambda d: d+2
        def tuple_arr_comp(one:tuple[np.ndarray], 
                           other:tuple[np.ndarray])->bool:
            conds = []
            conds.append(
                type(one)==type(other)
            )
            conds.append(
                len(one)==len(other)
            )
            conds.append( all([all(e1==e2) for e1,e2 in zip(one,other)])
            )
            return all(conds) 
            
        obj1 = distribution(
            pymc.Categorical, 'd1',
            np.asarray([0.1,.5,.4])
        )
        
        obj2 = distribution(
            pymc.Beta, 'd2',
            alpha=1.0, beta=1.0
        )
        obj3 = distribution(
            pymc.MvStudentT, 'd3',
            mu =np.asarray([0.1]*3),
                               sigma = np.diag(np.asarray(
                                   [3.0]*3)),
                               nu = 5
        )
        obj4 = distribution(
            pymc.Normal,'d4', 
            0, sigma=5
        )
        
        obj5 = distribution(
            pymc.Exponential,'d5',
            1/29, transform = call_ref
         )
        
        predicates = dict(
            discreet = all([
                obj1.name == 'd1',
                obj1.dist == pymc.Categorical,
                tuple_arr_comp(obj1.dist_args ,
                               (np.asarray([0.1,.5,.4]),)),
                obj1.dist_kwargs == dict(),
                obj1.dist_transform is None
                ]),
            
            cont_kw_only = all([
                obj2.name == 'd2',
                obj2.dist == pymc.Beta,
                obj2.dist_args == tuple(),
                obj2.dist_kwargs == dict(alpha=1.0, beta=1.0),
                obj2.dist_transform is None
                ]),
            
            mutivariate = all([
                obj3.name == 'd3',
                obj3.dist == pymc.MvStudentT,
                obj3.dist_args == tuple(),
                all([
                    obj3.dist_kwargs['nu'] == 5,
                    all(obj3.dist_kwargs['mu'] == np.asarray([0.1]*3)),
                ]),
                obj3.dist_transform is None
                ]),
            kw_arg_mix = all([
                obj4.name == 'd4',
                obj4.dist == pymc.Normal,
                obj4.dist_args == (0,),
                obj4.dist_kwargs == dict(sigma=5),
                obj4.dist_transform is None
                ]),
            transformed = all([
                obj5.name == 'd5',
                obj5.dist == pymc.Exponential,
                obj5.dist_args == (1/29,),
                obj5.dist_kwargs == dict(),
                obj5.dist_transform == call_ref
                ]),
        )
        
        self.assertTrue(
            all([
                v for _,v in predicates.items()
            ])
        )
        
    def test_illegal_name(self):
        illegal_names = [None, '', (5,6), ['no',89], {'4':4},
                         np.asarray([1.0]*5)]
        for illegal in illegal_names:
            self.assertRaises(
                ValueError, Distribution,
                    dist=pymc.Normal, name=illegal, dist_args=(0,1)
                    
            )
            self.assertRaises(
                ValueError, distribution,
                    pymc.Normal, illegal, (0,1)
                
            )
    
    def test_illegal_values(self):
        pass

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

class TestFreeVars(TestFramework):
    
    def test_insertions(self):
        ds = dict(
            sigma = distribution(pymc.Normal, "sigma",0,1,
                                 ),
            beta = distribution(
                pymc.Beta, "beta", alpha=1.0, beta=5.0
            )
        )
        free = FreeVariablesComponent(dists=ds)
        with pymc.Model() as model:
            free()
        self.assertTrue(
            set(ds.keys()).issubset(set(e.name for e in model.free_RVs)
        ))
        
    def test_variables(self):
        ds = dict(
            sigma = distribution(pymc.Normal, "sigma",0,1,
                                 ),
            beta = distribution(
                pymc.Beta, "beta", alpha=1.0, beta=5.0
            )
        )
        free = FreeVariablesComponent(dists=ds)
        with pymc.Model() as model:
            free()
        self.assertTrue(
            set(free.variables.keys())=={"sigma","beta"}
        )
        
    def test_illegal_inputs(self):
        ds = {
            'sigma' : distribution(pymc.Normal, "sigma",0,1,
                                 ),
            'beta' : distribution(
                pymc.Beta, "beta", alpha=1.0, beta=5.0
            )
        }
        illegals = [
            partial(pymc.Normal,'bob' ,0,1),
            pymc.Normal,
            dict(),
            {
            1 : distribution(pymc.Normal, "sigma",0,1,
                                 ),
            0 : distribution(
                pymc.Beta, "beta", alpha=1.0, beta=5.0
            )
            },
            {
            'sigma' : partial(pymc.Normal,'bob' ,0,1),
            'beta' : partial(pymc.Normal,'alice' ,0,1)
            }
        ]
        for i,illegal in enumerate(illegals):
            if i in (0,1,3,4): 
                self.assertRaises(
                    TypeError, FreeVariablesComponent, illegal
                )
            else:
                self.assertRaises(
                    ValueError, FreeVariablesComponent, illegal
                )

class TestBuilder(TestFramework):
    
    def test_basic_components(self):
        X = np.random.rand(50,3)
        Y = 5*X+15
        # Test missing outputs
        core = LinearRegressionCoreComponent(
            distributions = dict(
                inputs = distribution(
                    pymc.Data, 'inputs', X, mutable=True
                    ),
                outputs = distribution(
                    pymc.Data, 'outputs', Y, mutable=False
                ),
                W = distribution(
                    pymc.Normal, "W", 2,1, shape = 3
                ),
                b = distribution(
                    pymc.Normal, 'b', 2,1, shape = 3
                )
                
        ))
        fvars = FreeVariablesComponent(
            dict(
                σ = distribution(pymc.Normal, 'σ', 0,1, shape=3),
                ν = Distribution(
                    dist = pymc.Exponential, name = 'ν',
                    dist_args = (1/29.0,),
                    dist_transform = lambda d: d+1
                    ),
            )
        )
        like = LikelihoodComponent(
            distribution = pymc.StudentT,
            var_mapping = dict(
                mu = 'f',
                sigma = 'σ',
                nu ='ν', 
            )
        )
        b = ModelDirector(
            core, [like],
            free_vars_component = fvars
            
        )
        m = b()
        model = b.model
        self.assertTrue(all([
            { 'σ', 'ν', 'W','b' 
                }.issubset(set(e.name for e in  model.free_RVs)),
            {'f'}.issubset(set(e.name for e in model.deterministics)) 
        ]))
        
    def test_invalid_likelihoods(self):
        X = np.random.rand(50,3)
        Y = 5*X+15
        core = LinearRegressionCoreComponent(
            distributions = dict(
                inputs = distribution(
                    pymc.Data, 'inputs', X, mutable=True
                    ),
                outputs = distribution(
                    pymc.Data, 'outputs', Y, mutable=False
                ),
                W = distribution(
                    pymc.Normal, "W", 2,1, shape = 3
                ),
                b = distribution(
                    pymc.Normal, 'b', 2,1, shape = 3
                )
                
        ))
        fvars = FreeVariablesComponent(
            dict(
                σ = distribution(pymc.Normal, 'σ', 0,1, shape=3),
            )
        )
        like = LikelihoodComponent(
            distribution = pymc.StudentT,
            var_mapping = dict(
                mu = 'f',
                sigma = 'σ',
                nu ='ν', 
            )
        )
        b = ModelDirector(
            core, [like],
            free_vars_component = fvars
            
        )
        self.assertRaises(
            RuntimeError, b.__call__
        )

    def test_missing_inputs(self):
        X = np.random.rand(50,3)
        Y = 5*X+15
        core = LinearRegressionCoreComponent(
            distributions = dict(
                inputs = distribution(
                    pymc.Data, 'inputs', X, mutable=True
                    ),
                W = distribution(
                    pymc.Normal, "W", 2,1, shape = 3
                ),
                b = distribution(
                    pymc.Normal, 'b', 2,1, shape = 3
                )
                
        ))
        fvars = FreeVariablesComponent(
            dict(
                σ = distribution(pymc.Normal, 'σ', 0,1, shape=3),
            )
        )
        like = LikelihoodComponent(
            distribution = pymc.StudentT,
            var_mapping = dict(
                mu = 'f',
                sigma = 'σ',
                nu ='ν', 
            )
        )
        b = ModelDirector(
            core, [like],
            free_vars_component = fvars
            
        )
        self.assertRaises(
            RuntimeError, b.__call__
        )

    def test_var_catalogue(self):
        X = np.random.rand(50,3)
        Y = 5*X+15
        core = LinearRegressionCoreComponent(
            distributions = dict(
                inputs = distribution(
                    pymc.Data, 'inputs', X, mutable=True
                    ),
                outputs = distribution(
                  pymc.Data, 'outputs', Y, mutable = False  
                ),
                W = distribution(
                    pymc.Normal, "W", 2,1, shape = 3
                ),
                b = distribution(
                    pymc.Normal, 'b', 2,1, shape = 3
                )
                
        ))
        fvars = FreeVariablesComponent(
            dict(
                σ = distribution(pymc.Normal, 'σ', 0,1, shape=3),
                ν = distribution(pymc.Exponential, 'ν', 
                                 1/29.0, transform = lambda e:e+1
                                 )
            )
        )
        like = LikelihoodComponent(
            distribution = pymc.StudentT,
            var_mapping = dict(
                mu = 'f',
                sigma = 'σ',
                nu ='ν', 
            )
        )
        b = ModelDirector(
            core, [like],
            free_vars_component = fvars
        )()
        self.assertTrue(all([
            set( v.name for v in b.model.free_RVs)=={'σ', 'ν', 'W','b'},
            set(v.name for v in b.model.deterministics)=={'f'},
            ])
        )

class TestCompositeModels(TestFramework):
    
    def test_linear_regression(self):
        X = np.random.rand(50,3)
        Y = 5*X+15
        core = LinearRegressionCoreComponent(
            distributions = dict(
                X = distribution(
                    pymc.Data, 'X', X, mutable=True
                    ),
                Y = distribution(
                  pymc.Data, 'Y', Y, mutable = False  
                    ),
                β0 = distribution(
                    pymc.Normal, 'β0', 2,1, shape = 3
                    ),
                β1 = distribution(
                    pymc.Normal, 'β1', 2,1, shape = 3
                    )
            ),
            var_names = dict(
                data='X', slope = 'β0', intercept = 'β1',
                equation = 'μ'
            )
            )
        fvars = FreeVariablesComponent(
            dict(
                ε = distribution(pymc.Normal, 'ε', 0,1, shape=3)
            )
        )
        like = LikelihoodComponent(
            distribution = pymc.Normal,
            var_mapping = dict(
                mu = 'μ',
                sigma = 'ε',
            ),
            observed = 'Y'
        )
        b = ModelDirector(
            core, [like],
            free_vars_component = fvars
        )()
        self.assertTrue(all([
            {'β0', 'β1', 'ε'}<=set( v.name for v in b.model.free_RVs),
            {'μ'}<=set(v.name for v in b.model.deterministics),
            ])
        )
        
        
    