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
#   This module contains unit tests for the Gaussian Process model

import unittest
from bayesian_models.models import GaussianProcess
from bayesian_models.typing import PREDICATES
from bayesian_models.core import GPLayer, GaussianSubprocess
from bayesian_models.core import Distribution, distribution
from bayesian_models.core import FreeVariablesComponent
from bayesian_models.core import LikelihoodComponent
from bayesian_models.core import ModelAdaptorComponent
from bayesian_models.core import ResponseFunctionComponent
from bayesian_models.core import ResponseFunctions, Kernel
from bayesian_models.core import CoregionKernel
from bayesian_models.utilities import merge_dicts
import sklearn
import numpy as np
import pandas as pd
import warnings
import pymc as pm, pymc

class TestGaussianProcesses(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls)->None:
        r'''
            Set up datasets for testing
        '''
        from sklearn.datasets import load_iris, load_diabetes
        X, y = load_iris(return_X_y=True, 
                                          as_frame=True)
        names = load_iris().target_names
        Y = y.replace({i:name for i, name in enumerate(names)})
        df = pd.concat([X,Y], axis=1)
        df.columns = df.columns.tolist()[:-1]+["species"]
        cls.iris_df = df
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
        iq_df = pd.DataFrame(
            dict(value=np.r_[y1, y2],
                 group=np.r_[["drug"] * len(drug), ["placebo"] * len(
            placebo)])
            )
        iq_df.columns = ["iq", "treatment_group"]
        cls.iq_df = iq_df
        
        X, y = load_diabetes(
            return_X_y = True, 
            as_frame = True,
            scaled = False,
        )
        diab_df = pd.concat([X, y], axis=1)
        cls.diab_df = diab_df
    
    def test_kernel_context(self):
        
        with Kernel() as kernel:
            name = 'k'
            base_kernels = {
                'k_se' : pm.gp.cov.ExpQuad,
                "k_ln":pm.gp.cov.Linear,
            }
            kernel_parameters = {
                'k_se': {
                    'ls':distribution(
                        pm.HalfCauchy, 'λ', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln':{
                    'c':distribution(
                        pm.ConstantData, 'c', 5
                    )
                }
            }
            ext_vars = {
                'k_se': {
                    'η_se':distribution(
                        pm.HalfCauchy, 'η_se', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln': {
                    'η_ln':distribution(
                        pm.HalfCauchy, 'η_ln', 2, 
                        transform = lambda e:e+1
                    ),
                },
            }
            kernel_transformers = {
                'k_se' : lambda ker, pars: pars['η_se']*ker,
                'k_ln' : lambda ker, pars: pars['η_ln']*ker,
            }
            kernel_combinator = lambda kers: kers['k_se'] + \
                kers["k_ln"]
    
    def test_kernel(self):
        kernel = Kernel(
            name='k',
            base_kernels = {
                'k_se' : pm.gp.cov.ExpQuad,
                "k_ln":pm.gp.cov.Linear,
            },# type:ignore
            kernel_parameters = {
                'k_se': {
                    'ls':distribution(
                        pm.HalfCauchy, 'λ', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln':{
                    'c':distribution(
                        pm.ConstantData, 'c', 5
                    )
                }
                
            }, # type:ignore
            ext_vars = {
                'k_se': {
                    'η_se':distribution(
                        pm.HalfCauchy, 'η_se', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln': {
                    'η_ln':distribution(
                        pm.HalfCauchy, 'η_ln', 2, 
                        transform = lambda e:e+1
                    ),
                },
            },#type: ignore
            kernel_transformers = {
                'k_se' : lambda ker, pars: pars['η_se']*ker,
                'k_ln' : lambda ker, pars: pars['η_ln']*ker,
            },# type: ignore
            kernel_combinator = lambda kers: kers['k_se'] + \
                kers["k_ln"]
        )
        
    def test_predict(self):
        X = self.diab_df.values[:,:3]
        Y = self.diab_df.values[:,[-1]]
        # with GaussianProcess() as gp:
        #     with Kernel() as kernel:
        #         name="k"
        #         base_kernels = dict(
        #             k_se = pm.gp.cov.ExpQuad
        #         )
        #         kernel_parameters = dict(
        #             k_se = dict(
        #                 ls = distribution(
        #                     pm.HalfCauchy, 'λ', 2,
        #                     transform = lambda e: e+.1
        #                 )
        #             )
        #         )
        #     layers = GPLayer(
        #         [
        #             GaussianSubprocess(
        #                 kernel = kernel,
        #                 index = (0,i)
        #             ) for i in range(1)
        #         ],
        #         layer_idx=0
        #     )
        #     free_variables_component = FreeVariablesComponent(
        #         dict(
        #             σ = distribution(
        #                 pm.HalfNormal, 'σ', 1, transform=lambda e:e+.1
        #             )
        #         )
        #     )
        #     likelihood_component = LikelihoodComponent(
        #         var_mapping = dict(
        #             mu = 'f',
        #             sigma = 'σ'
        #         )
        #     )
        # gp.fit(X,Y)
        # trace = gp.predict(X)
        with pymc.Model() as model:
            k = pm.gp.cov.ExpQuad(3, ls=1)
            gp = pm.gp.HSGP([10]*3, c=1.2, cov_func=k)
            f = gp.prior('f', X)
            f_star = gp.conditional('f_star', X)
            s = pm.HalfNormal('s', 2)+.1
            y = pm.Normal('y', mu=f,sigma=s, observed=Y)
            
        with model:
            idata = pm.sample()
        with model:
            trace = pm.sample_posterior_predictive(
                idata, var_names=['f_star'])

    def test_coreg(self):
        X = self.diab_df.values[:, :-1]
        M = X.shape[-1] # Input dimensions
        p:int = 3 # Number of outputs
        q:int = 3 # Number of processes
        Y = self.diab_df.values[:,[-1]]
        with Kernel() as k_b:
            name = "kb"
            base_kernels = {
                'k_se': pm.gp.cov.ExpQuad
            }
            kernel_parameters = {
                "k_se": {
                    'ls' : distribution(
                        pymc.HalfCauchy, 'λ', 2,
                        shape = (M,),
                        transform = lambda e:e+.1
                    )
                }
            }
            ext_vars = {
                'k_se' : {
                    'η_se' : distribution(
                        pymc.HalfCauchy, 'η_se', 2,
                        transform = lambda e:e+.1
                    )
                }
            }
            kernel_transformers = {
                'k_se' : lambda k, pars: pars['η_se']*k 
            }
        with CoregionKernel() as cor:
            name = "k"
            kernels = {
                'k_se' : k_b
            }
            coreg_params = {
                'k_se' : {
                    'W': distribution(
                        pymc.Normal, 'W',0,1, shape=(p,q)
                    ),
                    'kappa' : distribution(
                        pymc.Normal, 'kappa', 0,1, shape=(p,)
                    )
                }
            }
            

    def test_init(self):
        kernel = Kernel(
            name = 'k',
            base_kernels = {
                'k_se' : pm.gp.cov.ExpQuad,
                "k_ln":pm.gp.cov.Linear,
            },# type:ignore
            kernel_parameters = {
                'k_se': {
                    'ls':distribution(
                        pm.HalfCauchy, 'λ', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln':{
                    'c':distribution(
                        pm.ConstantData, 'c', 5
                    )
                }
                
            }, # type:ignore
            ext_vars = {
                'k_se': {
                    'η_se':distribution(
                        pm.HalfCauchy, 'η_se', 2, 
                        transform = lambda e:e+1
                    ),
                },
                'k_ln': {
                    'η_ln':distribution(
                        pm.HalfCauchy, 'η_ln', 2, 
                        transform = lambda e:e+1
                    ),
                },
            },#type: ignore
            kernel_transformers = {
                'k_se' : lambda ker, pars: pars['η_se']*ker,
                'k_ln' : lambda ker, pars: pars['η_ln']*ker,
            },# type: ignore
            kernel_combinator = lambda kers: kers['k_se'] + \
                kers["k_ln"]
        )
        layers:list[GPLayer] = []
        l1 = GPLayer(
            [
                GaussianSubprocess(
                    kernel = kernel,
                    mean = pm.gp.mean.Zero,
                    index = (0,i),
                ) for i in range(3)
            ],
            layer_idx=0
        )
        layers.append(l1)
        adaptor_component = ModelAdaptorComponent(
            var_mapping = dict(
                f_mu = lambda t: t.T[0,:],
                f_sigma = lambda t: t.T[1,:],
            )
        )
        likelihoods_component = [LikelihoodComponent(
                observed = 'train_outputs',
                distribution = pm.Normal,
                var_mapping = dict(
                    mu = 'f_mu',
                    sigma = 'σ'
                )
            )]
        responses_component = ResponseFunctionComponent(
            ResponseFunctions(
                functions = dict(
                    σ = lambda t: pm.math.exp(t)
                    ),
                application_targets = dict(
                    σ = 'f_sigma',
                    ),
                records = dict(
                    σ = False,
                    ),
            )
        )
        
        obj = GaussianProcess(
            layers = layers,
            likelihoods_component = likelihoods_component,
            adaptor_component=adaptor_component,
            responses_component=responses_component,
        )
        obj([self.diab_df.values[:,:-1]],
            [self.diab_df.values[:,[-1]]])
        obj.fit(10, tune=10, chains=2)
        obj.predict(self.diab_df.values[:,:-1], method="full")
            
    def test_context(self):
        import pymc as pm
            
        with GaussianProcess() as obj:
            with Kernel() as kernel:
                name='k'
                base_kernels = {
                    'k_se' : pm.gp.cov.ExpQuad
                }
                kernel_parameters = {
                    'k_se' : {
                        'ls' : distribution(
                            pm.MutableData, 'λ', 1
                        )
                    }
                }
            layers:list[GPLayer] = []
            l1 = GPLayer(
                [
                    GaussianSubprocess(
                        kernel = kernel,
                        mean = pm.gp.mean.Zero,
                        index = (0,i),
                    ) for i in range(3)
                ],
                layer_idx=0
            )
            layers.append(l1)
            adaptor_component = ModelAdaptorComponent(
                var_mapping = dict(
                    f_mu = lambda t: t.T[0,:],
                    f_sigma = lambda t: t.T[1,:],
                )
            )
            likelihoods_component = [LikelihoodComponent(
                    observed = 'train_outputs',
                    distribution = pm.Normal,
                    var_mapping = dict(
                        mu = 'f_mu',
                        sigma = 'σ'
                    )
                )]
            responses_component = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = dict(
                        σ = lambda t: pm.math.exp(t)
                        ),
                    application_targets = dict(
                        σ = 'f_sigma',
                        ),
                    records = dict(
                        σ = False,
                        ),
                )
            )
        obj([self.diab_df.values[:,:-1]],
            [self.diab_df.values[:,[-1]]])
        obj.fit(10, tune=10, chains=2)
        trace = obj.predict(self.diab_df.values[:,:-1])
        
    def test_context_topology(self):
        import pymc as pm
        import pickle
        with GaussianProcess() as obj:
            kernel = Kernel(
                'k',
                base_kernels = {
                    'k_se' : pm.gp.cov.ExpQuad
                },
                kernel_parameters = {
                    'k_se': {
                        'ls': distribution(
                            pm.HalfCauchy, 'l', 2, 
                            transform = lambda e:e+1
                        ),
                    }
                    },
            )
            layers:list[GPLayer] = []
            l1 = GPLayer(
                [
                    GaussianSubprocess(
                        kernel = kernel,
                        mean = pm.gp.mean.Zero,
                        index = (0,i),
                        topology="A"
                    ) for i in range(5)
                ] + [
                    GaussianSubprocess(
                        kernel = kernel,
                        mean = pm.gp.mean.Zero,
                        index = (1,i),
                        topology="B",
                    ) for i in range(3)
                    ],
                layer_idx=0,
                topology = "C"
            )
            layers.append(l1)
            adaptor_component = ModelAdaptorComponent(
                var_mapping = dict(
                    f_mu = lambda t: t.T[0,:],
                    f_sigma = lambda t: t.T[1,:],
                )
            )
            likelihoods_component = [LikelihoodComponent(
                    observed = 'train_outputs',
                    distribution = pm.Normal,
                    var_mapping = dict(
                        mu = 'f_mu',
                        sigma = 'σ'
                    )
                )]
            responses_component = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = dict(
                        σ = lambda t: pm.math.exp(t)
                        ),
                    application_targets = dict(
                        σ = 'f_sigma',
                        ),
                    records = dict(
                        σ = False,
                        ),
                )
            )
        obj([self.diab_df.values[:,:-1]],
            [self.diab_df.values[:,[-1]]])
        obj.fit(draws=10, tune=10, chains=2)
        # with open("gp_test.pickle", "wb") as file:
        #     pickle.dump(obj.idata, file)
        # with open("gp_test.pickle", "rb") as file:
        #     obj.idata = pickle.load(file)  
        # obj.trained = True
        obj.predict(self.diab_df.values[:,:-1])

    def test_HSGP(self):
        import pymc as pm
        from bayesian_models.core import HSGP
        HSGP.n_basis = [7]*3
        HSGP.prop_ext_factor = 1.2
        X = self.diab_df.values[:, :3]
        Y = self.diab_df.values[:,[-1]]
        with GaussianProcess(gaussian_processor=HSGP) as obj:
            layers:list[GPLayer] = []
            with Kernel() as kernel:
                name = "k"
                base_kernels = {
                    'k_se': pm.gp.cov.ExpQuad
                }
                kernel_parameters = {
                    'k_se' : {
                        'ls' : distribution(
                            pm.HalfCauchy,'λ', 2, 
                            transform=lambda e:e+.1
                        )
                    }
                }
            l1 = GPLayer(
                [
                    GaussianSubprocess(
                        kernel = kernel,
                        mean = pm.gp.mean.Zero,
                        index = (0,i),
                    ) for i in range(2)
                ],
                layer_idx=0,
            )
            layers.append(l1)
            adaptor_component = ModelAdaptorComponent(
                var_mapping = dict(
                    f_mu = lambda t: t.T[0,:],
                    f_sigma = lambda t: t.T[1,:],
                )
            )
            likelihoods_component = [LikelihoodComponent(
                    observed = 'train_outputs',
                    distribution = pm.Normal,
                    var_mapping = dict(
                        mu = 'f_mu',
                        sigma = 'σ'
                    )
                )]
            responses_component = ResponseFunctionComponent(
                ResponseFunctions(
                    functions = dict(
                        σ = lambda t: pm.math.exp(t)
                        ),
                    application_targets = dict(
                        σ = 'f_sigma',
                        ),
                    records = dict(
                        σ = False,
                        ),
                )
            )
        obj([X],
            [Y])
        obj.fit(draws=500, tune=500, chains=2)
        # with open("hsgp_test.pickle", "wb") as file:
        #     pickle.dump(obj.idata, file)
        # with open("gp_test.pickle", "rb") as file:
        #     obj.idata = pickle.load(file)  
        # obj.trained = True
        obj.predict(self.diab_df.values[:,:-1])