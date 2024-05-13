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

import logging
from typing import Callable
from typing import Optional
from typing import Union

import paddle.nn as nn
from paddle import Tensor

from modulus.sym.models.layers.weight_norm import WeightNormLinear
from modulus.sym.models.activation import Activation, get_activation_fn

logger = logging.getLogger(__name__)


class FCLayer(nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )
        self.weight_norm = weight_norm
        self.activation_par = activation_par

        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias_attr=True)
        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) -> None:
        nn.initializer.Constant(0)(self.linear.bias)
        nn.initializer.XavierUniform()(self.linear.weight)
        if self.weight_norm:
            nn.initializer.Constant(1.0)(self.linear.weight_g)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


# FC like layer for image channels
class ConvFCLayer(nn.Layer):
    def __init__(
        self,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.activation_par = activation_par

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def apply_activation(self, x: Tensor) -> Tensor:
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[Tensor] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1D(in_features, out_features, kernel_size=1, bias_attr=True)
        self.reset_parameters()

        if weight_norm:
            logger.warn("Weight norm not supported for Conv FC layers")

    def reset_parameters(self) -> None:
        nn.initializer.Constant(0)(self.conv.bias)
        nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[Tensor] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.initializer.Constant(0)(self.conv.bias)
        self.conv.bias.stop_gradient = True
        nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[Tensor] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3D(in_channels, out_channels, kernel_size=1, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.initializer.Constant(0)(self.conv.bias)
        nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x
