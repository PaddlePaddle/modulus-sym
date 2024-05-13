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

"""
constant values used by Modulus
"""

import paddle
import numpy as np

# string used to determine derivatives
diff_str: str = "__"

# function to apply diff string
def diff(y: str, x: str, degree: int = 1) -> str:
    return diff_str.join([y] + degree * [x])


# for changing to float16 or float64
tf_dt = paddle.get_default_dtype()
np_dt = np.float32

# tensorboard naming
TF_SUMMARY = False

# Pytorch Version for which JIT will be default on
# JIT_PYTORCH_VERSION = "2.1.0a0+4136153"
JIT_PADDLE_VERSION = None

# No scaling is needed if using NO_OP_SCALE
NO_OP_SCALE = (0.0, 1.0)
# If using NO_OP_NORM, it is effectively doing no normalization
NO_OP_NORM = (-1.0, 1.0)
