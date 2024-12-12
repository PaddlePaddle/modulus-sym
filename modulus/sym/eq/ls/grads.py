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
from typing import List

Tensor = paddle.Tensor


class FirstDeriv(paddle.nn.Layer):
    """Module to compute first derivative with 2nd order accuracy using least squares method"""

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        assert (
            self.dim > 1
        ), "First Derivative through least squares method only supported for 2D and 3D inputs"

    def forward(self, coords, connectivity_tensor, y) -> List[Tensor]:
        p1 = coords[connectivity_tensor[:, :, 0]]
        p2 = coords[connectivity_tensor[:, :, 1]]
        dx = p1[:, :, 0] - p2[:, :, 0]
        dy = p1[:, :, 1] - p2[:, :, 1]

        f1 = y[connectivity_tensor[:, :, 0]]
        f2 = y[connectivity_tensor[:, :, 1]]

        du = (f1 - f2).squeeze(-1)

        result = []
        if self.dim == 2:
            w = 1 / paddle.sqrt(dx**2 + dy**2)
            w = paddle.where(paddle.isinf(w), paddle.to_tensor(1.0).to(w.place), w)
            mask = paddle.ones_like(dx)

            a1 = paddle.sum((w**2 * dx * dx) * mask, axis=1)
            b1 = paddle.sum((w**2 * dx * dy) * mask, axis=1)
            d1 = paddle.sum((w**2 * du * dx) * mask, axis=1)

            a2 = paddle.sum((w**2 * dx * dy) * mask, axis=1)
            b2 = paddle.sum((w**2 * dy * dy) * mask, axis=1)
            d2 = paddle.sum((w**2 * du * dy) * mask, axis=1)

            detA = paddle.linalg.det(
                paddle.stack(
                    [
                        paddle.stack([a1, a2], axis=1),
                        paddle.stack([b1, b2], axis=1),
                    ],
                    axis=2,
                )
            )
            dudx = (
                paddle.linalg.det(
                    paddle.stack(
                        [
                            paddle.stack([d1, d2], axis=1),
                            paddle.stack([b1, b2], axis=1),
                        ],
                        axis=2,
                    )
                )
                / detA
            )
            dudy = (
                paddle.linalg.det(
                    paddle.stack(
                        [
                            paddle.stack([a1, a2], axis=1),
                            paddle.stack([d1, d2], axis=1),
                        ],
                        axis=2,
                    )
                )
                / detA
            )
            result.append(dudx.unsqueeze(axis=1))
            result.append(dudy.unsqueeze(axis=1))
            return result
        elif self.dim == 3:
            dz = p1[:, :, 2] - p2[:, :, 2]

            w = 1 / paddle.sqrt(dx**2 + dy**2 + dz**2)
            w = paddle.where(paddle.isinf(w), paddle.to_tensor(1.0).to(w.place), w)
            mask = paddle.ones_like(dx)

            a1 = paddle.sum((w**2 * dx * dx) * mask, axis=1)
            b1 = paddle.sum((w**2 * dx * dy) * mask, axis=1)
            c1 = paddle.sum((w**2 * dx * dz) * mask, axis=1)
            d1 = paddle.sum((w**2 * du * dx) * mask, axis=1)

            a2 = paddle.sum((w**2 * dx * dy) * mask, axis=1)
            b2 = paddle.sum((w**2 * dy * dy) * mask, axis=1)
            c2 = paddle.sum((w**2 * dy * dz) * mask, axis=1)
            d2 = paddle.sum((w**2 * du * dy) * mask, axis=1)

            a3 = paddle.sum((w**2 * dx * dz) * mask, axis=1)
            b3 = paddle.sum((w**2 * dy * dz) * mask, axis=1)
            c3 = paddle.sum((w**2 * dz * dz) * mask, axis=1)
            d3 = paddle.sum((w**2 * du * dz) * mask, axis=1)

            detA = paddle.linalg.det(
                paddle.stack(
                    [
                        paddle.stack([a1, a2, a3], axis=1),
                        paddle.stack([b1, b2, b3], axis=1),
                        paddle.stack([c1, c2, c3], axis=1),
                    ],
                    axis=2,
                )
            )
            dudx = (
                paddle.linalg.det(
                    paddle.stack(
                        [
                            paddle.stack([d1, d2, d3], axis=1),
                            paddle.stack([b1, b2, b3], axis=1),
                            paddle.stack([c1, c2, c3], axis=1),
                        ],
                        axis=2,
                    )
                )
                / detA
            )
            dudy = (
                paddle.linalg.det(
                    paddle.stack(
                        [
                            paddle.stack([a1, a2, a3], axis=1),
                            paddle.stack([d1, d2, d3], axis=1),
                            paddle.stack([c1, c2, c3], axis=1),
                        ],
                        axis=2,
                    )
                )
                / detA
            )
            dudz = (
                paddle.linalg.det(
                    paddle.stack(
                        [
                            paddle.stack([a1, a2, a3], axis=1),
                            paddle.stack([b1, b2, b3], axis=1),
                            paddle.stack([d1, d2, d3], axis=1),
                        ],
                        axis=2,
                    )
                )
                / detA
            )

            result.append(dudx.unsqueeze(axis=1))
            result.append(dudy.unsqueeze(axis=1))
            result.append(dudz.unsqueeze(axis=1))
            return result
