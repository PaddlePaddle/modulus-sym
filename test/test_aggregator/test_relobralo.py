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
import numpy as np
import paddle
from paddle import nn
from modulus.sym.loss.aggregator import Relobralo


class FitToPoly(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.create_parameter(
            shape=(512, 512),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.b = self.create_parameter(
            shape=(512, 1),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        losses = {
            "loss_x": nn.functional.relu(
                x=paddle.mm(input=self.w, mat2=x1) + self.b - x1**2
            )
            .abs()
            .mean(),
            "loss_y": nn.functional.relu(
                x=paddle.mm(input=self.w, mat2=x2) + self.b - x2**2.0
            )
            .abs()
            .mean(),
            "loss_z": nn.functional.relu(
                x=paddle.mm(input=self.w, mat2=x3) + self.b + x3**2.0
            )
            .abs()
            .mean(),
        }
        return losses


def test_loss_aggregator():
    # set device
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    # Load data
    filename = os.path.join(
        os.path.dirname(__file__), "test_aggregator_data/Relobralo_data.npz"
    )
    configs = np.load(filename, allow_pickle=True)
    x_np = paddle.to_tensor(configs["x_np"][()]).to(device)
    w_np, b_np, loss_np = (
        configs["w_np"][()],
        configs["b_np"][()],
        configs["loss_np"][()],
    )
    total_steps, learning_rate = (
        configs["total_steps"][()],
        configs["learning_rate"][()],
    )

    # Instantiate the optimizer, scheduler, aggregator, and loss fucntion
    loss_function = FitToPoly()
    aggregator = Relobralo(loss_function.parameters(), 3)
    optimizer = paddle.optimizer.SGD(learning_rate, loss_function.parameters())
    tmp_lr = paddle.optimizer.lr.PiecewiseDecay(
        values=[0.3333333333333333 * optimizer.get_lr(), optimizer.get_lr()],
        boundaries=[5],
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    # Training loop
    for step in range(total_steps):
        optimizer.clear_grad()
        train_losses = loss_function(x_np)
        train_loss = aggregator(train_losses, step)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

    # check outputs
    w_out = list(loss_function.parameters())[0].cpu().detach().numpy()
    b_out = list(loss_function.parameters())[1].cpu().detach().numpy()
    loss_out = train_loss.cpu().detach().numpy()
    # print(w_out,w_np, b_out,b_np, loss_out,loss_np)
    assert np.allclose(loss_np, loss_out, rtol=1e-4, atol=1e-4)
    assert np.allclose(w_np, w_out, rtol=1e-4, atol=1e-4)
    assert np.allclose(b_np, b_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_loss_aggregator()
