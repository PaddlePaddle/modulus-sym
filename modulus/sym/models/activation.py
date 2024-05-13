# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import enum
from typing import Callable
from typing import Union
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from modulus.sym.manager import JitManager, JitArchMode


class ActivationMeta(enum.EnumMeta):
    def __getitem__(self, name):
        try:
            return super().__getitem__(name.upper())
        except KeyError as error:
            raise KeyError(f"Invalid activation function {name}")


class Activation(enum.Enum, metaclass=ActivationMeta):
    ELU = enum.auto()
    LEAKY_RELU = enum.auto()
    MISH = enum.auto()
    RELU = enum.auto()
    GELU = enum.auto()
    SELU = enum.auto()
    PRELU = enum.auto()
    SIGMOID = enum.auto()
    SILU = enum.auto()
    SIN = enum.auto()
    SQUAREPLUS = enum.auto()
    SOFTPLUS = enum.auto()
    TANH = enum.auto()
    STAN = enum.auto()
    IDENTITY = enum.auto()


def identity(x: paddle.Tensor) -> paddle.Tensor:
    return x


def squareplus(x: paddle.Tensor) -> paddle.Tensor:
    b = 4
    return 0.5 * (x + paddle.sqrt(x * x + b))


def gelu(x: paddle.Tensor) -> paddle.Tensor:
    # Applies GELU approximation, slower than sigmoid but more accurate. See: https://github.com/hendrycks/GELUs
    # Standard GELU that is present in Paddle does not JIT compile!
    return 0.5 * x * (1.0 + paddle.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    # return 0.5 * x * (1 + paddle.tanh(paddle.sqrt(2 / np.pi) * (x + 0.044715 * paddle.pow(x, 3))))


# Use silu forward decomposition only when
# 1) `FLAGS_use_cinn` is set to True, or
# 2) `silu_comp` env variable is set to True.
use_silu_forward_composite = os.getenv("silu_comp", "False").lower() == "true"
use_silu_forward_composite |= bool(os.getenv("FLAGS_use_cinn", "False").lower() == "true")

if use_silu_forward_composite:
    silu_func = lambda x: x * F.sigmoid(x)
else:
    silu_func = F.silu

def custom_silu(x: paddle.Tensor) -> paddle.Tensor:
    # return paddle.nn.functional.silu(x)
    return silu_func(x)


class CustomSilu(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_silu(x)


class Identity(nn.Layer):
    """Identity activation function

    Dummy function for removing activations from a model

    Example
    -------
    >>> idnt_func = modulus.models.layers.Identity()
    >>> input = torch.randn(2, 2)
    >>> output = idnt_func(input)
    >>> torch.allclose(input, output)
    True
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Stan(nn.Layer):
    """
    Self-scalable Tanh (Stan)
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    """

    def __init__(self, out_features=1):
        super().__init__()
        self.beta = self.create_parameter(
            [out_features],
            default_initializer=paddle.nn.initializer.Constant(1),
        )

    def forward(self, x):
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return paddle.tanh(x) * (1.0 + self.beta * x)


def get_activation_fn(
    activation: Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]],
    module: bool = False,
    **kwargs,  # Optional parameters
) -> Callable[[paddle.Tensor], paddle.Tensor]:
    activation_mapping = {
        Activation.ELU: F.elu,
        Activation.LEAKY_RELU: F.leaky_relu,
        Activation.MISH: F.mish,
        Activation.RELU: F.relu,
        Activation.GELU: F.gelu,
        Activation.SELU: F.selu,
        Activation.SIGMOID: F.sigmoid,
        Activation.SILU: custom_silu,
        Activation.SIN: paddle.sin,
        Activation.SQUAREPLUS: squareplus,
        Activation.SOFTPLUS: F.softplus,
        Activation.TANH: paddle.tanh,
        Activation.IDENTITY: identity,
    }
    # Some activations have parameters in them thus must
    # be in a Module before forward call
    module_activation_mapping = {
        Activation.ELU: nn.ELU,
        Activation.LEAKY_RELU: nn.LeakyReLU,
        Activation.MISH: nn.Mish,
        Activation.RELU: nn.ReLU,
        Activation.GELU: nn.GELU,
        Activation.SELU: nn.SELU,
        Activation.PRELU: nn.PReLU,
        Activation.SIGMOID: nn.Sigmoid,
        Activation.SILU: CustomSilu,
        Activation.TANH: nn.Tanh,
        Activation.STAN: Stan,
    }

    if activation in activation_mapping and not module:
        activation_fn_ = activation_mapping[activation]
        # wraps the function because paddle.sin and F.gelu could not be scripted directly
        def activation_fn(x: Tensor) -> Tensor:
            return activation_fn_(x)

    elif activation in module_activation_mapping:
        activation_fn = module_activation_mapping[activation](**kwargs)
    else:
        activation_fn = activation

    if JitManager().enabled and JitManager().arch_mode == JitArchMode.ONLY_ACTIVATION:
        raise NotImplementedError("JIT is not supported for activation functions")
        activation_fn = paddle.jit.to_static(activation_fn)
    return activation_fn
