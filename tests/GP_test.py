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
from bayesian_models.utilities import merge_dicts
import sklearn
import numpy as np
import pandas as pd
import warnings

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
        
    def test_dev(self):
        r'''
            Convenience test for interactive development
            Remove when ready
        '''
        import pymc as pm
        import pytensor
        N_LAYERS:int = 2
        N_NEURONS:int = 3

        with pm.Model() as DGP_model:
            inputs = pm.Data(
                'inputs', 
                self.diab_df.values[:,:-1], 
                mutable=False)
            
            outputs = pm.Data(
                'outputs', 
                self.diab_df.values[:,[-1]]
                )
            X = inputs
        with DGP_model:
            # Layers
            for i in range(N_LAYERS):
                # Sub processes
                gps = []
                fs = []
                for j in range(N_NEURONS):
                    η = pm.HalfCauchy(f'η_{i,j}', 1)
                    λ = pm.HalfCauchy(f'λ_{i,j}', 1)
                    mean_func = pm.gp.mean.Constant(0)
                    cov = η*pm.gp.cov.Matern52(
                        input_dim=X.shape[-1].eval(), ls=λ,
                        )
                    gp = pm.gp.Latent(
                        mean_func = mean_func, 
                        cov_func = cov
                        )
                    gps.append(gp)
                    f = gp.prior(f'f_{i,j}', X)
                    fs.append(f)
                f = pytensor.tensor.stack(fs).T
                X = f
            nu = pm.Deterministic('nu', pm.math.exp(f[:,[0]]))
            sigma = pm .Deterministic('sigma', f[:,[1]]**2)
            mu = pm.Deterministic('mu', f[:,[2]])
            
            y_obs = pm.StudentT(
                'y_obs', 
                observed = outputs,
                mu = mu,
                sigma= sigma,
                nu = nu,
                )
            
        with DGP_model:
            idata = pm.sample(10, tunes=10, chains=2)
        print("Hi")
        
    def test_back_layers(self):
        import pymc as pm
        import pytensor
        N_LAYERS:int = 2
        L:list[GPLayer] = []
        variables:dict = dict()
        var_catalogue:dict = dict()
        temp_params:list = []
        with pm.Model() as DGP_model:
            inputs = pm.Data(
                'inputs', 
                self.diab_df.values[:100,:-1], 
                mutable=False)
            
            outputs = pm.Data(
                'outputs', 
                self.diab_df.values[:100,[-1]]
                )
            X = inputs
            for i in range(N_LAYERS):
                p = [GaussianSubprocess(
                        kernel = pm.gp.cov.Matern12,
                        mean = pm.gp.mean.Zero,
                        kernel_hyperparameters = dict(
                            ls = distribution(pm.HalfCauchy, 'λ',1 )
                        ),
                    index = (i,k)
                ) for k in range(10)]
                L.append(GPLayer(
                    p, layer_idx = i
                ))
            for i, layer in enumerate(L):
                for k, process in enumerate(layer):
                    temp_params.append(process.__extract_basic_rvs__())
            variables = merge_dicts(variables,*temp_params)
        with DGP_model:
            for var_name, dist_obj in variables.items():
                var_catalogue[var_name] = dist_obj.dist(
                    var_name, *dist_obj.dist_args, 
                    dist_obj.dist_kwargs)
                
        with DGP_model:
            for l in  L:
                X = l(X, var_catalogue)
            f = X
            nu = pm.Deterministic('nu', pm.math.exp(f[:,[0]]))
            sigma = pm .Deterministic('sigma', f[:,[1]]**2)
            mu = pm.Deterministic('mu', f[:,[2]])
            
            y_obs = pm.StudentT(
                'y_obs', 
                observed = outputs,
                mu = mu,
                sigma= sigma,
                nu = nu,
                )
            
        with DGP_model:
            idata = pm.sample(10, tune=10, chains=2)