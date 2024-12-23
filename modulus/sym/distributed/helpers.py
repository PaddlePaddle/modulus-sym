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
import paddle.nn.functional as F
import paddle.distributed as dist


def get_memory_format(tensor):
    raise NotImplementedError("get_memory_format is not implemented")


def pad_helper(tensor, dim, new_size, mode="zero"):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = paddle.flip(
            paddle.conj(tensor_pad[rhs_slice]), axis=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous()

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert (
        dim < tensor.dim()
    ), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (
        tensor.shape[dim] % num_chunks == 0
    ), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = paddle.split(
        tensor, num_or_sections=tensor.shape[dim] // chunk_size, axis=dim
    )

    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [
        y.contiguous()
        for y in paddle.split(tensor, tensor.shape[dim0] // split_size, axis=dim0)
    ]
    x_recv = [paddle.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = dist.alltoall(x_send, x_recv, group=group, sync_op=not async_op)

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.astype(dtype="float32")
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert (
        dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    tensor_list = [paddle.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    output = paddle.concat(tensor_list, axis=dim_).contiguous()

    return output
