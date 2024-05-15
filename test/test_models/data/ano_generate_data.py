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

from functools import partial
from re import S
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

paddle.seed(0)
np.random.seed(0)
cuda_device = str("cpu").replace("cuda", "gpu")

################################################################
# Baseline AFNO implementation from Jiadeeps original wind dataset implementation
# Based on: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
################################################################
def compl_mul_add_act(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor
) -> paddle.Tensor:
    tmp = paddle.einsum("bxykis,kiot->stbxyko", a, b)
    res = (
        paddle.stack(
            x=[tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]],
            axis=-1,
        )
        + c
    )
    return res


def compl_mul_add_act_c(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor
) -> paddle.Tensor:
    tmp = paddle.einsum("bxyki,kio->bxyko", a, b)
    res = tmp + c
    return res


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()

        tensor = tensor * (std * math.sqrt(2.0))
        tensor.add_(y=paddle.to_tensor(mean))

        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(
    x: paddle.Tensor, drop_prob: float = 0.0, training: bool = False
) -> paddle.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        # new
        self.w1 = self.create_parameter(
            [
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        self.num_blocks,
                        self.block_size,
                        self.block_size * self.hidden_size_factor,
                        2,
                    ]
                )
            ),
        )
        self.b1 = self.create_parameter(
            [
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                        2,
                    ]
                )
            ),
        )
        self.w2 = self.create_parameter(
            [
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                        self.block_size,
                        2,
                    ]
                )
            ),
        )
        self.b2 = self.create_parameter(
            [self.num_blocks, self.block_size, 2],
            default_initializer=nn.initializer.Assign(
                self.scale * paddle.randn(shape=[self.num_blocks, self.block_size, 2])
            ),
        )

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.astype(dtype="float32")
        B, H, W, C = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        x = paddle.fft.rfft2(x=x, axes=(1, 2), norm="ortho")
        x = x.reshape([B, H, W // 2 + 1, self.num_blocks, self.block_size])

        # new
        x = paddle.as_real(x=x)
        o2 = paddle.zeros(shape=x.shape)

        o1 = F.relu(
            x=compl_mul_add_act(
                x[
                    :,
                    total_modes - kept_modes : total_modes + kept_modes,
                    :kept_modes,
                    ...,
                ],
                self.w1,
                self.b1,
            )
        )
        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ...
        ] = compl_mul_add_act(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(x=o2, threshold=self.sparsity_threshold)
        x = paddle.as_complex(x=x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = paddle.fft.irfft2(x=x, s=(H, W), axes=(1, 2), norm="ortho")
        x = x.astype(dtype)

        return x + bias


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()

        # print("LN normalized shape", dim)
        self.norm1 = norm_layer(dim)

        self.filter = AFNO2D(
            dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # original
        self.norm2 = norm_layer(dim)
        # new
        # self.norm2 = norm_layer((h, w, dim))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNONet(nn.Layer):
    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(16, 16),
        in_chans=2,
        out_chans=2,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = self.create_parameter(
            [1, embed_dim, num_patches],
            default_initializer=nn.initializer.Assign(
                paddle.zeros(shape=[1, embed_dim, num_patches])
            ),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)
        ]
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        # new
        self.head = nn.Conv2D(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            1,
            bias_attr=False,
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init_Constant = nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # new
        x = x.reshape(
            [b, self.embed_dim, h // self.patch_size[0], w // self.patch_size[1]]
        )

        # transpose here to see if rest is OK: (B, H, W, C)
        x = x.transpose(perm=(0, 2, 3, 1))

        for blk in self.blocks:
            x = blk(x)

        # permute back: (B, C, H, W)
        x = x.transpose(perm=(0, 3, 1, 2))

        return x

    def forward(self, x):
        # new: B, C, H, W
        b, h, w = x.shape[0], x.shape[-2], x.shape[-1]

        x = self.forward_features(x)
        x = self.head(x)

        xv = x.reshape(
            [
                b,
                self.patch_size[0],
                self.patch_size[1],
                -1,
                h // self.patch_size[0],
                w // self.patch_size[1],
            ]
        )
        xvt = paddle.transpose(x=xv, perm=(0, 3, 4, 1, 5, 2))
        x = xvt.reshape([b, -1, h, w])

        return x


class PatchEmbed(nn.Layer):
    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        return x


################################################################
#  configurations
################################################################


img_size = (64, 64)
patch_size = (16, 16)
in_channels = 2
out_channels = 5
n_layers = 4
modes = 16
embed_dim = 64

model = AFNONet(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_channels,
    out_chans=out_channels,
    embed_dim=embed_dim,
    depth=n_layers,  # Number of model layers
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    num_blocks=modes,  # Number of modes
).to(cuda_device)
x_numpy = np.random.rand(2, in_channels, img_size[0], img_size[1]).astype(np.float32)
x_tensor = paddle.to_tensor(x_numpy).to(cuda_device)
y_tensor = model(x_tensor)
y_numpy = y_tensor.detach().numpy()
Wbs = {
    _name: _value.data.detach().numpy() for _name, _value in model.named_parameters()
}
params = {
    "modes": modes,
    "img_size": img_size,
    "patch_size": patch_size,
    "in_channels": in_channels,
    "out_channels": out_channels,
    "n_layers": n_layers,
    "modes": modes,
    "embed_dim": embed_dim,
}
np.savez_compressed(
    "test_ano.npz", data_in=x_numpy, data_out=y_numpy, params=params, Wbs=Wbs
)
