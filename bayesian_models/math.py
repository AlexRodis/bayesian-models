#   Copyright 2023 Alexander Rodis
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
# This module contains additional math functions and definitions, mostly
# related to neural network activations / transfer functions

import pytensor
import pymc

__all__ = ( 
        'ReLU',
        'GELU',
        'ELU',
        'SWISS',
        'SiLU'
        )


def ReLU(x, leak:float=.0):
    r'''
        `pytensor` implementation of the ReLU activation function:

        .. math::

            f(x)\triangleq max(leak, x)

        NOTE: With leak=0 this is the standard ReLU function. With
        leak a small number i.e. 1e-2 this is Leaky ReLU. Otherwise this
        is Parametric ReLU

        Args:
        ------

            - x := The input tensor

            - | leak:float=.0 := The leak parameter of the ReLU. When
                equal to 0, returns the standard ReLU, when equal to a
                small number i.e. 0.001 this is Leaky ReLU, otherwise
                it's parametric ReLu

        Returns:
        ---------

            - function := Elementwise operation

    '''

    return pytensor.tensor.switch(x<=0, leak, x)

def ELU(x, alpha:float=1.0):
    r'''
        `pytensor` implementation of the ELU activation function:

        .. math::

            f(x) \triangleq \begin{cases}
                    x & x \gt 0 \\
                    \alpha (e^x-1) &\text{if } b \\
                    \end{cases}

        Args:
        -----

            - x := The input tensor

            - | alpha:float=1.0 := The :math:`\alpha` parameter of the
                ELU function

        Returns:
        ---------

            - function := Elementwise operation
    '''

    return pytensor.tensor.switch(x<=0, 
                                  alpha*(pytensor.tensor.exp(x)-1), 
                                  x)

def SWISS(x, beta:float=1):
    r'''
        `pytensor` implementation of the Swiss activation function:

        .. math::

            f(x) \triangleq x sigmoid(\beta x)

        NOTE: This implementation is equivalent to the 'Swiss-1' activation
        function, where :math:`\beta` is **not** learned. The original
        SWISS function has this as a learnable parameter instead
    '''

    return x*pymc.math.invlogit(beta*x)

def GELU(x):
    r'''
        `pytensor` implementaion of the GELU activation function. 
        
        This function is defined as:

        .. math::
        
            X \thicksim \mathcal{N}(0,1)
            f(x) \triangleq xP(X\le x) = x\Phi (x)=x \frac 12 
            [1+erf(\frac {x}{\sqrt{2}})]

        Args:
        ------

            - x := Input tensor

        Returns:
        --------

            - function := The elementwise operation
    '''

    return x*0.5*(1+pymc.math.erf(x/pymc.math.sqrt(2)))


def SiLU(x):
    r'''
        `pytensor` implementation of the SiLU activation function. This
        function is defined as:

        .. math::

            f(x) \triangleq x \sigma (x)
        
        Args:
        -----

             - x := Input tensor

        Returns:
        ---------

            - function := The elementwise operation
    '''

    return x*pymc.math.invlogit(x)
