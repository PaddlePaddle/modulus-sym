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

import numpy as np
import paddle
import os
from modulus.sym.eq.pdes.wave_equation import WaveEquation, HelmholtzEquation


def test_wave_equation():
    # test data for wave equation
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)

    u = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)

    c = 0.1

    u__t__t = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__x__x = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__y__y = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__z__z = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)

    wave_equation_true = u__t__t - c * c * u__x__x - c * c * u__y__y - c * c * u__z__z

    # evaluate the equation
    eq = WaveEquation(u="u", c=c, dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u__x__x": paddle.to_tensor(u__x__x, dtype="float32"),
            "u__y__y": paddle.to_tensor(u__y__y, dtype="float32"),
            "u__z__z": paddle.to_tensor(u__z__z, dtype="float32"),
            "u__t__t": paddle.to_tensor(u__t__t, dtype="float32"),
        }
    )
    eq_eval = evaluations["wave_equation"].numpy()

    # verify PDE computation
    assert np.allclose(eq_eval, wave_equation_true), "Test Failed!"


def test_helmholtz_equation():
    # test data for helmholtz equation
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)

    u = np.sin(x) * np.sin(y) * np.sin(z)

    k = 0.1

    u__x__x = -np.sin(x) * np.sin(y) * np.sin(z)
    u__y__y = -np.sin(x) * np.sin(y) * np.sin(z)
    u__z__z = -np.sin(x) * np.sin(y) * np.sin(z)

    helmholtz_equation_true = -(k**2 * u + u__x__x + u__y__y + u__z__z)

    # evaluate the equation
    eq = HelmholtzEquation(u="u", k=k, dim=3)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u": paddle.to_tensor(u, dtype="float32"),
            "u__x__x": paddle.to_tensor(u__x__x, dtype="float32"),
            "u__y__y": paddle.to_tensor(u__y__y, dtype="float32"),
            "u__z__z": paddle.to_tensor(u__z__z, dtype="float32"),
        }
    )
    eq_eval = evaluations["helmholtz"].numpy()

    # verify PDE computation
    assert np.allclose(eq_eval, helmholtz_equation_true), "Test Failed!"


if __name__ == "__main__":
    test_wave_equation()
    test_helmholtz_equation()
