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

import pytest
import paddle
from paddle import nn

from modulus.sym.models.layers import SpectralConv1d, SpectralConv2d, SpectralConv3d


class SpectralConv1d_old(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super(SpectralConv1d_old, self).__init__()

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
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            shape=[
                bsize,
                self.out_channels,
                x.shape[-1] // 2 + 1,
            ],
            dtype="complex64"
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        return x


class SpectralConv2d_old(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_old, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ),
        )
        self.weights2 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[in_channels, out_channels, self.modes1, self.modes2],
                    dtype="complex64",
                )
            ),
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft2(x)
        out_ft = paddle.zeros(
            [
                batchsize,
                self.out_channels,
                x.shape[-2],
                x.shape[-1] // 2 + 1
            ],
            dtype="complex64",
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class SpectralConv3d_old(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_old, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                    ],
                    dtype="complex64",
                )
            ),
        )
        self.weights2 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                    ],
                    dtype="complex64",
                )
            ),
        )
        self.weights3 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                    ],
                    dtype="complex64",
                )
            ),
        )
        self.weights4 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.rand(
                    shape=[
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                    ],
                    dtype="complex64",
                )
            ),
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            shape=[
                batchsize,
                self.out_channels,
                x.shape[-3],
                x.shape[-2],
                x.shape[-1] // 2 + 1,
            ],
            dtype="complex64",
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]))
        return x


@pytest.mark.skip(reason="Paddle does not support generating random complex numbers yet.")
def test_spectral_convs():

    in_channels = 2
    out_channels = 3
    modes = 4
    sc1d_old = SpectralConv1d_old(in_channels, out_channels, modes)
    # Init weights
    sc1d_old.weights1.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes]),
    )

    sc1d = SpectralConv1d(in_channels, out_channels, modes)
    # Copy to new model
    sc1d.weights1.data = paddle.stack(
        [sc1d_old.weights1.real(), sc1d_old.weights1.imag()], axis=-1
    )
    inputs = paddle.randn(shape=[5, in_channels, 32])
    # Forward pass of spectral conv
    output_old = sc1d_old(inputs)
    output = sc1d(inputs)

    assert paddle.allclose(
        output_old, y=output, rtol=1e-3, atol=1e-3
    ).item(), "Spectral conv 1d mismatch"

    sc2d_old = SpectralConv2d_old(in_channels, out_channels, modes, modes)
    sc2d_old.weights1.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes]),
    )
    sc2d_old.weights2.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes]),
    )

    sc2d = SpectralConv2d(in_channels, out_channels, modes, modes)
    # Copy to new model
    sc2d.weights1.data = paddle.stack(
        [sc2d_old.weights1.real(), sc2d_old.weights1.imag()], axis=-1
    )
    sc2d.weights2.data = paddle.stack(
        [sc2d_old.weights2.real(), sc2d_old.weights2.imag()], axis=-1
    )
    inputs = paddle.randn(shape=[5, in_channels, 32, 32])
    # Forward pass of spectral conv
    output_old = sc2d_old(inputs)
    output = sc2d(inputs)

    assert paddle.allclose(
        output_old, y=output, rtol=1e-3, atol=1e-3
    ).item(), "Spectral conv 2d mismatch"

    sc3d_old = SpectralConv3d_old(in_channels, out_channels, modes, modes, modes)
    sc3d_old.weights1.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
    )
    sc3d_old.weights2.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
    )
    sc3d_old.weights3.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
    )
    sc3d_old.weights4.data = paddle.complex(
        real=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
        imag=paddle.randn(shape=[in_channels, out_channels, modes, modes, modes]),
    )

    sc3d = SpectralConv3d(in_channels, out_channels, modes, modes, modes)
    # Copy to new model
    sc3d.weights1.data = paddle.stack(
        [sc3d_old.weights1.real(), sc3d_old.weights1.imag()], axis=-1
    )
    sc3d.weights2.data = paddle.stack(
        [sc3d_old.weights2.real(), sc3d_old.weights2.imag()], axis=-1
    )
    sc3d.weights3.data = paddle.stack(
        [sc3d_old.weights3.real(), sc3d_old.weights3.imag()], axis=-1
    )
    sc3d.weights4.data = paddle.stack(
        [sc3d_old.weights4.real(), sc3d_old.weights4.imag()], axis=-1
    )

    inputs = paddle.randn(shape=[5, in_channels, 32, 32, 32])
    # Forward pass of spectral conv
    output_old = sc3d_old(inputs)
    output = sc3d(inputs)

    assert paddle.allclose(
        output_old, y=output, rtol=1e-3, atol=1e-3
    ).item(), "Spectral conv 3d mismatch"


# test_spectral_convs()

if __name__ == "__main__":
    pytest.main()
