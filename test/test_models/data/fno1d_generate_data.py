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
import paddle.nn as nn
import paddle.nn.functional as F


paddle.seed(0)
np.random.seed(0)
cuda_device = str("cpu").replace("cuda", "gpu")

################################################################
# 1d fourier neural operator
# Based on: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
################################################################
class SpectralConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1], dtype="complex64"
                )
            ),
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return paddle.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft(x=x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            shape=[batchsize, self.out_channels, x.shape[-1] // 2 + 1],
            dtype="complex64",
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = paddle.fft.irfft(x=out_ft, n=x.shape[-1])
        return x


class FNO1d(nn.Layer):
    def __init__(self, modes, width):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1D(self.width, self.width, 1)
        self.w1 = nn.Conv1D(self.width, self.width, 1)
        self.w2 = nn.Conv1D(self.width, self.width, 1)
        self.w3 = nn.Conv1D(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.place)
        batchsize = x.shape[0]
        x = paddle.concat(x=(x, grid), axis=-1)
        x = self.fc0(x)
        x = x.transpose(perm=[0, 2, 1])
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding]  # pad the domain if input is non-periodic
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = paddle.to_tensor(np.linspace(0, 1, size_x), dtype="float32")
        gridx = gridx.reshape(1, size_x, 1).tile(repeat_times=[batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  configurations
################################################################

modes = 16
width = 64
model = FNO1d(modes, width).to(cuda_device)

x_numpy = np.random.rand(100, 100, 1).astype(np.float32)
x_tensor = paddle.to_tensor(x_numpy).to(cuda_device)
y_tensor = model(x_tensor)
y_numpy = y_tensor.detach().numpy()
Wbs = {
    _name: _value.data.detach().numpy() for _name, _value in model.named_parameters()
}
params = {"modes": modes, "width": width, "padding": 2}
np.savez_compressed(
    "test_fno1d.npz", data_in=x_numpy, data_out=y_numpy, params=params, Wbs=Wbs
)
