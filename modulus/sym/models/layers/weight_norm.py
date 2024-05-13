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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor


class WeightNormLinear(nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = paddle.create_parameter(
            (out_features, in_features), dtype=paddle.get_default_dtype()
        )
        self.weight_g = paddle.create_parameter(
            (out_features, 1), dtype=paddle.get_default_dtype()
        )
        if bias:
            self.bias = paddle.create_parameter(
                (out_features,), dtype=paddle.get_default_dtype()
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.initializer.XavierUniform()(self.weight)
        nn.initializer.Constant(1.0)(self.weight_g)
        if self.bias is not None:
            nn.initializer.Constant(0.0)(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        norm = self.weight.norm(p=2, axis=1, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return F.linear(input, weight.T, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
