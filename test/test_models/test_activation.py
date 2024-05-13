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

import warnings
import paddle
import pytest
from packaging import version
from modulus.sym.manager import JitManager
from modulus.sym.utils.benchmark import profile, timeit
from modulus.sym.models.activation import Activation, get_activation_fn

# Allow fusing single node, and prevent tiny autodiff graph are inlined/reverted.
# These flags are automatically set when specifying jit_manager.enabled is True.
# User needs to set these flags manually if they would like to fuse activation
# function for standalone code.
#

skip_if_no_gpu = pytest.mark.skipif(
    not paddle.device.cuda.device_count() >= 1, reason="There is no GPU to run this test"
)

@pytest.mark.skip()
def test_activation_jit():
    jit_manager = JitManager()
    jit_manager.enabled = True
    jit_manager.arch_mode = "only_activation"

    for act in Activation:
        act_scripted = get_activation_fn(act)
        assert isinstance(
            act_scripted, (paddle.jit.ScriptFunction, paddle.jit.ScriptModule)
        )

    def sin(x):
        return paddle.sin(x)

    sin_scripted = get_activation_fn(sin)
    assert isinstance(sin_scripted, paddle.jit.ScriptFunction)


@pytest.mark.skip()
@skip_if_no_gpu
def test_activation_fused_silu():
    """
    Make sure SiLU derivative kernels are fused when jit_manager.arch_mode == "only_activation".
    We need to rely on the fused SiLU derivative kernels for AMP, because the unfused path
    may have intermediate results that overflow the FP16 dynamic range.
    """
    jit_manager = JitManager()
    jit_manager.enabled = True
    jit_manager.arch_mode = "only_activation"
    jit_manager.use_cinn = True

    silu_scripted = get_activation_fn(Activation.SILU)
    assert isinstance(silu_scripted, paddle.jit.ScriptFunction)

    device = "cuda"
    batch_size = 10000
    x = paddle.rand([batch_size, 512], device=device, requires_grad=True)
    I_N = paddle.ones_like(x)

    def run(func, order, *args):
        paddle.cuda.nvtx.range_push("forward")
        y = func(*args)
        paddle.cuda.nvtx.range_pop()

        if order >= 1:
            paddle.cuda.nvtx.range_push("1st order")
            (y__x,) = paddle.autograd.grad(y, [x], I_N, create_graph=True)
            paddle.cuda.nvtx.range_pop()

        if order >= 2:
            paddle.cuda.nvtx.range_push("2nd order")
            (y__x__x,) = paddle.autograd.grad(y__x, [x], I_N, create_graph=True)
            paddle.cuda.nvtx.range_pop()

        if order >= 3:
            paddle.cuda.nvtx.range_push("3rd order")
            (y__x__x__x,) = paddle.autograd.grad(y__x__x, [x], I_N, create_graph=True)
            paddle.cuda.nvtx.range_pop()

    def cleanup_events(event_keys):
        keys = ["cuLaunchKernel", "cudaLaunchKernel", "cudaDeviceSynchronize"]
        for evt in keys:
            if evt in event_keys:
                event_keys.remove(evt)
        return event_keys

    # benchmark
    silu = paddle.nn.functional.silu
    timeit(run, silu, 1, x, label="silu_1st", verbose=True)
    timeit(run, silu_scripted, 1, x, label="silu_scripted_1st", verbose=True)
    timeit(run, silu, 2, x, label="silu_2nd", verbose=True)
    timeit(run, silu_scripted, 2, x, label="silu_scripted_2nd", verbose=True)
    timeit(run, silu, 3, x, label="silu_3rd", verbose=True)
    timeit(run, silu_scripted, 3, x, label="silu_scripted_3rd", verbose=True)

    # profile and get the number of kernels
    verbose = False  # set to True to debug

    _, events = profile(
        run, silu_scripted, 1, x, label="silu_scripted_1st", verbose=verbose
    )
    event_keys = cleanup_events([evt.key for evt in events])
    num_kernels = len(event_keys)
    print("silu_scripted_1st num_events: ", num_kernels)
    if version.parse(paddle.__version__) >= version.parse("1.12.9"):
        # this depends on the SiLU autodiff PR: https://github.com/pypaddle/pypaddle/pull/81724
        # fwd + 1st_deriv kernels
        assert num_kernels == 2
    else:
        warnings.warn(f"Fused SiLU is not supported for paddle {paddle.__version__}")

    _, events = profile(
        run, silu_scripted, 2, x, label="silu_scripted_2nd", verbose=verbose
    )
    event_keys = cleanup_events([evt.key for evt in events])
    num_kernels = len(event_keys)
    print("silu_scripted_2nd num_events: ", num_kernels)
    if version.parse(paddle.__version__) >= version.parse("1.12.9"):
        # fwd + 1st_deriv + 2nd_deriv kernels
        assert num_kernels == 3
    else:
        warnings.warn(f"Fused SiLU is not supported for paddle {paddle.__version__}")

    _, events = profile(
        run, silu_scripted, 3, x, label="silu_scripted_3rd", verbose=verbose
    )
    event_keys = cleanup_events([evt.key for evt in events])
    num_kernels = len(event_keys)
    print("silu_scripted_3rd num_events: ", num_kernels)
    if version.parse(paddle.__version__) >= version.parse("1.12.9"):
        # fwd + 1st_deriv + 2nd_deriv + 3rd_deriv kernels
        assert num_kernels <= 7
    else:
        warnings.warn(f"Fused SiLU is not supported for paddle {paddle.__version__}")


if __name__ == "__main__":
    pytest.main()
    # test_activation_jit()
    # test_activation_fused_silu()
