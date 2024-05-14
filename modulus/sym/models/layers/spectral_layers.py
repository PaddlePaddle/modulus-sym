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

from typing import List
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor


class SpectralConv1d(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
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
            [in_channels, out_channels, self.modes1, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty([in_channels, out_channels, self.modes1, 2])
            ),
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul1d(self, input: Tensor, weights: Tensor) -> Tensor:
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = paddle.as_complex(weights)

        def einsum_bix_iox(A, B):
            t = paddle.bmm(
                A.transpose([2, 0, 1]),
                B.transpose([2, 0, 1]),
            ) # [x, b, o]
            t = t.transpose([1, 2, 0])
            return t

        # return einsum_bix_iox(input, cweights)
        return paddle.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # print("SpectralConv1d")
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            [bsize, self.out_channels, x.shape[-1] // 2 + 1], dtype="complex64"
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

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
            [in_channels, out_channels, self.modes1, self.modes2, 2],
        )
        self.weights2 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, 2],
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = paddle.as_complex(weights)

        def einsum_bixy_ioxy(A, B):
            b, i, x, y = A.shape
            o = B.shape[1]
            t = paddle.bmm(
                A.transpose([2, 3, 0, 1]).reshape([x*y, b, i]),
                B.transpose([2, 3, 0, 1]).reshape([x*y, i, o]),
            ) # [xy, b, o]
            t = t.reshape([x, y, b, o])
            t = t.transpose([2, 3, 0, 1])
            return t

        # return einsum_bixy_ioxy(input, cweights)
        return paddle.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # print("SpectralConv2d")
        x_ft = paddle.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            [batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1],
            dtype="complex64",
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space

        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x

    def reset_parameters(self):
        paddle.assign(self.weights1, self.scale * paddle.rand(self.weights1.shape))
        paddle.assign(self.weights2, self.scale * paddle.rand(self.weights2.shape))


class SpectralConv3d(nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights2 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights3 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights4 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul3d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        cweights = paddle.as_complex(weights)

        def einsum_bixyz_ioxyz(x_, y_):
            b, i, x, y, z = x_.shape
            o = y_.shape[1]
            t = paddle.bmm(
                x_.transpose([2, 3, 4, 0, 1]).reshape([x*y*z, b, i]),
                y_.transpose([2, 3, 4, 0, 1]).reshape([x*y*z, i, o]),
            ) # [xyz,b,o]
            t = t.reshape([x, y, z, b, o])
            t = t.transpose([3, 4, 0, 1, 2])
            return t

        # return einsum_bixyz_ioxyz(input, cweights)
        return paddle.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # print("SpectralConv3d")
        x_ft = paddle.fft.rfftn(x, axes=[-3, -2, -1])
        out_ft = paddle.zeros(
            [
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
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * paddle.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * paddle.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * paddle.rand(self.weights4.data.shape)


# ==========================================
# Utils for PINO exact gradients
# ==========================================
def fourier_derivatives(x: Tensor, l: List[float]) -> Tuple[Tensor, Tensor]:
    # print("fourier_derivatives")
    # check that input shape maches domain length
    assert len(x.shape) - 2 == len(l), "input shape doesn't match domain dims"

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    batchsize = x.shape[0]
    n = x.shape[2:]
    dim = len(l)

    # get device
    # device = x.place

    # compute fourier transform
    x_h = paddle.fft.fftn(x, axes=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            paddle.concat(
                (
                    paddle.arange(start=0, end=nx // 2, step=1, dtype=paddle.get_default_dtype()),
                    paddle.arange(start=-nx // 2, end=0, step=1, dtype=paddle.get_default_dtype()),
                ),
                axis=0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = paddle.complex(
        real=paddle.to_tensor([0.0]),
        imag=paddle.to_tensor([1.0]),
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * x_h * (2 * pi / l[i]) for i, k_x_i in enumerate(k_x)]
    wxx_h = [
        j * k_x_i * wx_h_i * (2 * pi / l[i])
        for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, k_x))
    ]

    # inverse fourier transform out
    wx = paddle.concat(
        [
            paddle.fft.ifftn(wx_h_i, axes=list(range(2, dim + 2))).real()
            for wx_h_i in wx_h
        ],
        axis=1,
    )
    wxx = paddle.concat(
        [
            paddle.fft.ifftn(wxx_h_i, axes=list(range(2, dim + 2))).real()
            for wxx_h_i in wxx_h
        ],
        axis=1,
    )
    return (wx, wxx)


def calc_latent_derivatives(
    x: Tensor, domain_length: List[int] = 2
) -> Tuple[List[Tensor], List[Tensor]]:

    dim = len(x.shape) - 2
    # Compute derivatives of latent variables via fourier methods
    # Padd domain by factor of 2 for non-periodic domains
    padd = [(i - 1) // 2 for i in list(x.shape[2:])]
    # Scale domain length by padding amount
    domain_length = [
        domain_length[i] * (2 * padd[i] + x.shape[i + 2]) / x.shape[i + 2]
        for i in range(dim)
    ]
    padding = padd + padd
    x_p = F.pad(x, padding, mode="replicate")
    dx, ddx = fourier_derivatives(x_p, domain_length)
    # Trim padded domain
    if len(x.shape) == 3:
        dx = dx[..., padd[0] : -padd[0]]
        ddx = ddx[..., padd[0] : -padd[0]]
        dx_list = paddle.split(dx, num_or_sections=(dx.shape[1] // x.shape[1]), axis=1)
        ddx_list = paddle.split(ddx, num_or_sections=(ddx.shape[1] // x.shape[1]), axis=1)
    elif len(x.shape) == 4:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        dx_list = paddle.split(dx, num_or_sections=(dx.shape[1] // x.shape[1]), axis=1)
        ddx_list = paddle.split(ddx, num_or_sections=(ddx.shape[1] // x.shape[1]), axis=1)
    else:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        dx_list = paddle.split(dx, num_or_sections=(dx.shape[1] // x.shape[1]), axis=1)
        ddx_list = paddle.split(ddx, num_or_sections=(ddx.shape[1] // x.shape[1]), axis=1)

    return dx_list, ddx_list


def first_order_pino_grads(
    u: Tensor,
    ux: List[Tensor],
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:
    # print("first_order_pino_grads")
    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1).unsqueeze(axis=-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / paddle.cosh(u_hidden) ** 2

    # compute diff(f(g))
    # print("mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str)
    # print(f"weights_1.shape = {weights_1.shape}") # [32, 32, 1, 1]
    # print(f"diff_tanh.shape = {diff_tanh.shape}") # [8, 32, 241, 241]
    # print(f"weights_2.shape = {weights_2.shape}") # [1, 32, 1, 1]
    b, i, k, m, x, y = (
        diff_tanh.shape[0], # b
        weights_1.shape[1], # i
        weights_2.shape[0], # k
        weights_1.shape[0], # m
        diff_tanh.shape[2], # x
        diff_tanh.shape[3], # y
    )
    weights_1_tmp = paddle.broadcast_to(weights_1.transpose((1, 0, 2, 3)).unsqueeze(1).unsqueeze(0), [b, i, k, m, x, y])
    diff_tanh_tmp = paddle.broadcast_to(diff_tanh.unsqueeze(1).unsqueeze(1), [b, i, k, m, x, y])
    weights_2_tmp = paddle.broadcast_to(weights_2.unsqueeze(0).unsqueeze(0), [b, i, k, m, x, y])

    diff_fg = (weights_1_tmp * diff_tanh_tmp * weights_2_tmp).sum(axis=(2, 3))
    # print(diff_fg.shape)
    # print(torch.allclose(z, diff_fg))

    # diff_fg = paddle.einsum(
    #     "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
    #     weights_1,
    #     diff_tanh,
    #     weights_2,
    # )

    # compute diff(f(g)) * diff(g)
    def einsum_bixy_bixy(A, B):
        t = (A * B).sum(1)
        return t

    # print("4 bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg.shape, ux[0].shape)
    vx = [
        einsum_bixy_bixy(diff_fg, w)
        # paddle.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in ux
    ]
    vx = [paddle.unsqueeze(w, axis=1) for w in vx]
    return vx


def second_order_pino_grads(
    u: Tensor,
    ux: Tensor,
    uxx: Tensor,
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:
    # print("second_order_pino_grads")
    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1).unsqueeze(axis=-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / paddle.cosh(u_hidden) ** 2

    # compute diff(f(g))
    # print("1 pattern = ", "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str)
    # print(weights_1.shape)
    # print(diff_tanh.shape)
    # print(weights_2.shape)
    b, i, k, m, x, y = (
        diff_tanh.shape[0], # b
        weights_1.shape[1], # i
        weights_2.shape[0], # k
        weights_1.shape[0], # m
        diff_tanh.shape[2], # x
        diff_tanh.shape[3], # y
    )
    weights_1_tmp = paddle.broadcast_to(weights_1.transpose((1, 0, 2, 3)).unsqueeze(1).unsqueeze(0), [b, i, k, m, x, y])
    diff_tanh_tmp = paddle.broadcast_to(diff_tanh.unsqueeze(1).unsqueeze(1), [b, i, k, m, x, y])
    weights_2_tmp = paddle.broadcast_to(weights_2.unsqueeze(0).unsqueeze(0), [b, i, k, m, x, y])
    diff_fg = (weights_1_tmp * diff_tanh_tmp * weights_2_tmp).sum(axis=(2, 3))

    # diff_fg = paddle.einsum(
    #     "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
    #     weights_1,
    #     diff_tanh,
    #     weights_2,
    # )

    # compute diagonal of hessian
    # double derivative of hidden layer
    diff_diff_tanh = -2 * diff_tanh * paddle.tanh(u_hidden)

    # compute diff(g) * hessian(f) * diff(g)
    # print("bi"
    #         + dim_str
    #         + ",mi"
    #         + dim_str
    #         + ",bm"
    #         + dim_str
    #         + ",mj"
    #         + dim_str
    #         + ",bj"
    #         + dim_str
    #         + "->b"
    #         + dim_str)
    # print([w.shape for w in ux])
    # print((weights_1).shape)
    # print((weights_2 * diff_diff_tanh).shape)
    # print((weights_1).shape)
    # print([w.shape for w in ux])
    # exit()
    # def einsum_vxx1_item(a, b, c, d, e):
    #     b, i, x, y = a.shape
    #     m = d.shape[0]
    #     j = d.shape[1]
    #     full_shape = [b, m, i, j, x, y]
    #     a_tmp = paddle.broadcast_to(a.unsqueeze(2).unsqueeze(1), full_shape)
    #     b_tmp = paddle.broadcast_to(b.unsqueeze(2).unsqueeze(0), full_shape)
    #     c_tmp = paddle.broadcast_to(c.unsqueeze(2).unsqueeze(2), full_shape)
    #     d_tmp = paddle.broadcast_to(d.unsqueeze(1).unsqueeze(0), full_shape)
    #     e_tmp = paddle.broadcast_to(e.unsqueeze(1).unsqueeze(1), full_shape)
    #     result = (a_tmp * b_tmp * c_tmp * d_tmp * e_tmp).sum(axis=(1, 2, 3))
    #     return result

    vxx1 = [
        paddle.einsum(
            "bi"
            + dim_str
            + ",mi"
            + dim_str
            + ",bm"
            + dim_str
            + ",mj"
            + dim_str
            + ",bj"
            + dim_str
            + "->b"
            + dim_str,
            w,
            weights_1,
            weights_2 * diff_diff_tanh,
            weights_1,
            w,
        )
        # einsum_vxx1_item(w, weights_1, weights_2 * diff_diff_tanh, weights_1, w)
        for w in ux
    ]  # (b,x,y,t)
    # compute diff(f) * hessian(g)
    def einsum_bixy_bixy(A, B):
        t = (A * B).sum(1)
        return t

    vxx2 = [
        einsum_bixy_bixy(diff_fg, w)
        # paddle.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in uxx
    ]
    # print("3 bi" + dim_str + ",bi" + dim_str + "->b" + dim_str)
    vxx = [paddle.unsqueeze(a + b, axis=1) for a, b in zip(vxx1, vxx2)]
    return vxx
